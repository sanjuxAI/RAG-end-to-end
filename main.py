import os
import re
import gc
import fitz
import json
import torch
import faiss
import asyncio
import textwrap
from tqdm.auto import tqdm
from time import perf_counter as timer
import numpy as np
from spacy.lang.en import English
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import TextIteratorStreamer
import threading

#-------------------------------------------
# GLOBAL DEVICE
#-------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)


#-------------------------------------------
# LOAD EMBEDDING MODEL WITH FALLBACK
#-------------------------------------------
def load_st_model(name):
    try:
        print(f"[INFO] Loading ST model: {name}")
        return SentenceTransformer(name, device=device)
    except Exception:
        print("[WARNING] Failed. Falling back to MiniLM.")
        return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)


#-------------------------------------------
# BUILD FAISS WITH FALLBACK
#-------------------------------------------
def build_faiss(dim):
    try:
        if device == "cuda":
            res = faiss.StandardGpuResources()
            cpu_index = faiss.IndexFlatIP(dim)
            gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
            print("[INFO] Using FAISS GPU index.")
            return gpu_index
        else:
            raise Exception("No GPU available.")
    except:
        print("[INFO] Using FAISS CPU index.")
        return faiss.IndexFlatIP(dim)


#-------------------------------------------
# NEXT-GEN RAG IMPLEMENTATION
#-------------------------------------------
class RAG:
    def __init__(self,
                 llm_path,
                 embedding_model="sentence-transformers/all-mpnet-base-v2",
                 ui_callback=None):
        """
        ui_callback(event_name, payload)
        event_name: "reading_pdf", "splitting", "chunking", "embedding_batch", "indexing", "answer_stream"
        """
        self.ui = ui_callback
        self.embedding_model = load_st_model(embedding_model)

        self.llm, self.tokenizer = self.load_llm(llm_path)
        self.index = None
        self.chunks = None
        self.emb_dim = self.embedding_model.get_sentence_embedding_dimension()


    #-----------------------------------------------------
    # LLM LOAD
    #-----------------------------------------------------
    def load_llm(self, path):
        tokenizer = AutoTokenizer.from_pretrained(path)
        model = AutoModelForCausalLM.from_pretrained(
            path,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        ).to(device)
        model.eval()
        model.requires_grad_(False)
        return model, tokenizer


    #-----------------------------------------------------
    # PDF READ
    #-----------------------------------------------------
    def read_pdf(self, pdf_path):
        pages = []
        doc = fitz.open(pdf_path)

        for i, page in enumerate(tqdm(doc, desc="Reading PDF")):
            if self.ui:
                self.ui("reading_pdf", {"page": i})

            text = page.get_text("text")
            text = text.replace("\n", " ").strip()
            pages.append({"page": i, "text": text})
        return pages


    #-----------------------------------------------------
    # SENTENCE SPLIT
    #-----------------------------------------------------
    def sentence_split(self, pages):
        nlp = English()
        nlp.add_pipe("sentencizer")

        for item in tqdm(pages, desc="Splitting"):
            if self.ui:
                self.ui("splitting", {"page": item["page"]})

            item["sentences"] = [str(s) for s in nlp(item["text"]).sents]
        return pages


    #-----------------------------------------------------
    # SLIDING WINDOW CHUNKING
    #-----------------------------------------------------
    def sliding_window_chunk(self, pages, window=8, overlap=3):
        """
        Example: window=8, overlap=3 => step size = 5
        """
        step = max(1, window - overlap)
        chunks = []

        for item in tqdm(pages, desc="Chunking"):
            if self.ui:
                self.ui("chunking", {"page": item["page"]})

            sents = item["sentences"]
            for i in range(0, len(sents), step):
                chunk = " ".join(sents[i:i+window]).strip()
                chunks.append({
                    "page": item["page"],
                    "chunk_text": chunk
                })

        return chunks


    #-----------------------------------------------------
    # ASYNC BATCH EMBEDDING
    #-----------------------------------------------------
    async def embed_async(self, texts, batch_size=18):
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            if self.ui:
                self.ui("embedding_batch", {"from": i, "to": i+len(batch)})

            # non-blocking async sleep allows UI to update
            await asyncio.sleep(0)

            # encode
            emb = self.embedding_model.encode(
                batch,
                convert_to_numpy=True,
                normalize_embeddings=True,
                batch_size=len(batch),
                show_progress_bar=False
            )
            all_embeddings.append(emb)

        return np.vstack(all_embeddings)


    #-----------------------------------------------------
    # BUILD INDEX (ASYNC)
    #-----------------------------------------------------
    async def build_index(self, chunks):
        texts = [c["chunk_text"] for c in chunks]

        embeddings = await self.embed_async(texts)
        if self.ui:
            self.ui("indexing", {"count": len(embeddings)})

        index = build_faiss(self.emb_dim)
        index.add(embeddings)

        self.index = index
        self.chunks = chunks


    #-----------------------------------------------------
    # RETRIEVE
    #-----------------------------------------------------
    def retrieve(self, query, top_k=5):
        query_emb = self.embedding_model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True
        ).reshape(1, -1)

        scores, idx = self.index.search(query_emb, top_k)
        results = []
        for i, score in zip(idx[0], scores[0]):
            results.append({
                "score": float(score),
                "chunk_text": self.chunks[i]["chunk_text"],
                "page": self.chunks[i]["page"]
            })
        return results


    #-----------------------------------------------------
    # PROMPT BUILD
    #-----------------------------------------------------
    def build_prompt(self, query, context):
        ctx = "\n- ".join([c["chunk_text"] for c in context])

        prompt = f"""
Use ONLY the context to answer the user.

Context:
- {ctx}

Question:
{query}

Answer:
"""

        messages = [{"role": "user", "content": prompt}]
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


    #-----------------------------------------------------
    # STREAMING GENERATION (TYPING EFFECT)
    #-----------------------------------------------------


    def stream_answer(self, prompt, max_new_tokens=350):
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt").to(device)

        # HuggingFace streaming iterator
        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_special_tokens=True,
            skip_prompt=True
        )

        # Run generation in background thread
        thread = threading.Thread(
            target=self.llm.generate,
            kwargs=dict(
                **inputs,
                streamer=streamer,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7
            )
        )
        thread.start()

        # Yield streamed text as it arrives
        for text in streamer:
            if self.ui:
                self.ui("answer_stream", {"text": text})
            yield text



    #-----------------------------------------------------
    # ASK
    #-----------------------------------------------------
    def ask(self, query, top_k=5):
        context = self.retrieve(query, top_k)
        prompt = self.build_prompt(query, context)

        return self.stream_answer(prompt), context


    #-----------------------------------------------------
    # EXPORT METADATA
    #-----------------------------------------------------
    def export_metadata(self, file_path="metadata.json"):
        data = {
            "total_chunks": len(self.chunks),
            "embedding_dim": self.emb_dim,
            "pages_used": list({c["page"] for c in self.chunks})
        }
        with open(file_path, "w") as f:
            json.dump(data, f, indent=2)
        return file_path


    #-----------------------------------------------------
    # FULL PIPELINE
    #-----------------------------------------------------
    async def process_pdf(self, pdf_path):
        pages = self.read_pdf(pdf_path)
        pages = self.sentence_split(pages)
        chunks = self.sliding_window_chunk(pages)

        await self.build_index(chunks)
        print("[INFO] Ready for queries.")
        return True


import asyncio

# ------------------------------------------------------------
# UI Callback
# ------------------------------------------------------------
def ui_callback(event, payload):
    """
    You can print, update frontend, send WebSocket messages, etc.
    """
    if event == "reading_pdf":
        print(f"Reading PDF page {payload['page']}...")
    elif event == "splitting":
        print(f"Splitting sentences for page {payload['page']}...")
    elif event == "chunking":
        print(f"Chunking sentences for page {payload['page']}...")
    elif event == "embedding_batch":
        print(f"Embedding batch {payload['from']} → {payload['to']}...")
    elif event == "indexing":
        print("Building FAISS index...")
    elif event == "answer_stream":
        print(payload["text"], end="", flush=True)

# ------------------------------------------------------------
# MAIN FUNCTION
# ------------------------------------------------------------
async def main():
    # ----------------------------------------
    # INITIALIZE RAG SYSTEM
    # ----------------------------------------
    rag = RAG(
        llm_path="LiquidAI/LFM2-1.2B-RAG",
        embedding_model="sentence-transformers/all-mpnet-base-v2",
        ui_callback=ui_callback
    )

    # ----------------------------------------
    # PROCESS PDF
    # ----------------------------------------
    pdf_path = "./uploads/12007111.pdf"  
    print(f"\nProcessing PDF: {pdf_path}\n")
    await rag.process_pdf(pdf_path)

    # ----------------------------------------
    # ASK QUESTION
    # ----------------------------------------
    query = "Explain the central concept of the document."
    print(f"\n\nQuery: {query}\n")
    print("Answer:\n")

    stream, context = rag.ask(query, top_k=5)

    # Streaming chunks
    for token in stream:
        pass  # streaming is handled inside ui_callback

    print("\n\nContext used:")
    for c in context:
        print(f"- Page {c['page']} | Score: {c['score']:.3f}")

    # ----------------------------------------
    # EXPORT METADATA
    # ----------------------------------------
    metadata_file = rag.export_metadata("rag_metadata.json")
    print(f"\nMetadata exported: {metadata_file}")


# ------------------------------------------------------------
# RUN MAIN
# ------------------------------------------------------------
if __name__ == "__main__":
    asyncio.run(main())
