import os
import re
from collections import defaultdict
import chromadb
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from app.config import (
    EMBEDDING_MODEL, LLM_MODEL, CHROMA_DB_PATH,
    SIMILARITY_THRESHOLD, FINETUNED_THRESHOLD, ACADEMIC_TOP_K, MAX_TYPO_DISTANCE,
    ACADEMIC_KEYWORDS, COURSE_KEYWORDS,
    FINETUNED_EMBEDDING_PATH, USE_FINETUNED,
)
from app.chunker import extract_chunks


STOP_WORDS = {
    "apa", "saja", "yang", "dalam", "dan", "di", "ke", "dari", "ini", "itu",
    "adalah", "bisa", "bagaimana", "mengapa", "dimana", "kapan", "siapa",
    "apakah", "tidak", "atau", "pada", "dengan", "untuk", "akan", "telah",
    "sudah", "sedang", "saya", "kami", "kita", "mereka", "dia", "anda",
    "secara", "sebagai", "oleh", "karena", "bahwa", "yaitu", "yakni",
    "tentang", "antara", "setelah", "sebelum", "dapat", "harus", "lebih",
    "dosa", "baru",
}

SHORT_KEYWORD_MIN = 4


def _reindex_worker(pdf_files, persist_dir, status_dict, *,
                    embedding_model=None, collection=None):
    """Extract chunks in parallel, embed once in one big batch, then insert.

    Can run in-process (pass embedding_model + collection) or
    in a subprocess (loads its own model & ChromaDB).
    """
    import gc
    import hashlib
    from concurrent.futures import ThreadPoolExecutor, as_completed

    own_model = own_collection = False
    if embedding_model is None:
        from sentence_transformers import SentenceTransformer
        path = FINETUNED_EMBEDDING_PATH if USE_FINETUNED else EMBEDDING_MODEL
        print(f"[worker] Loading embedding model: {path}")
        embedding_model = SentenceTransformer(path)
        own_model = True
    if collection is None:
        chroma_client = chromadb.PersistentClient(path=persist_dir)
        collection = chroma_client.get_or_create_collection("akreditasi_docs")
        own_collection = True

    total = len(pdf_files)
    status_dict.update({
        "running": True,
        "total_files": total,
        "current_file": 0,
        "current_filename": "",
        "phase": "extracting",
        "total_chunks": 0,
        "embedded_chunks": 0,
        "error": None,
    })

    def _extract_one(pdf_file):
        if not os.path.exists(pdf_file):
            return None
        try:
            chunks, chunk_pages, course_code, chunk_meta = extract_chunks(pdf_file)
            if not chunks:
                return None
            filename = os.path.basename(pdf_file)
            path_hash = hashlib.md5(pdf_file.encode()).hexdigest()[:12]
            texts, ids, metas = [], [], []
            for i, chunk in enumerate(chunks):
                texts.append(chunk)
                ids.append(f"{path_hash}_{i}")
                meta = chunk_meta[i] if i < len(chunk_meta) else {}
                metas.append({
                    "source": filename,
                    "chunk_id": i,
                    "course_code": course_code or meta.get("course_code", ""),
                    "page_number": chunk_pages[i] if i < len(chunk_pages) else 0,
                    "chunk_type": meta.get("chunk_type", "text"),
                })
            return (filename, texts, ids, metas)
        except Exception as e:
            print(f"Error extracting {pdf_file}: {e}")
            return None

    all_texts, all_ids, all_metadatas = [], [], []
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(_extract_one, f): f for f in pdf_files}
        done_count = 0
        for future in as_completed(futures):
            done_count += 1
            result = future.result()
            if result:
                filename, texts, ids, metas = result
                all_texts.extend(texts)
                all_ids.extend(ids)
                all_metadatas.extend(metas)
                status_dict["current_file"] = done_count
                status_dict["current_filename"] = filename
                print(f"[worker] Extracted [{done_count}/{total}]: {filename} ({len(texts)} chunks)")

    if not all_texts:
        status_dict["running"] = False
        status_dict["phase"] = "done"
        print("[worker] No chunks extracted")
        return

    total_chunks = len(all_texts)
    status_dict["total_chunks"] = total_chunks

    EMBED_BATCH = 500
    num_batches = (total_chunks + EMBED_BATCH - 1) // EMBED_BATCH
    embedded_so_far = 0

    for batch_idx in range(num_batches):
        start = batch_idx * EMBED_BATCH
        end = min(start + EMBED_BATCH, total_chunks)
        batch_texts = all_texts[start:end]
        batch_ids = all_ids[start:end]
        batch_metas = all_metadatas[start:end]

        status_dict["phase"] = f"embedding batch {batch_idx + 1}/{num_batches}"
        print(f"[worker] Embedding batch {batch_idx + 1}/{num_batches} ({len(batch_texts)} chunks)...")
        batch_embeddings = embedding_model.encode(batch_texts, batch_size=128, show_progress_bar=True)
        embedded_so_far += len(batch_texts)
        status_dict["embedded_chunks"] = embedded_so_far

        status_dict["phase"] = f"indexing batch {batch_idx + 1}/{num_batches}"
        embeddings_list = batch_embeddings.tolist() if hasattr(batch_embeddings, "tolist") else list(batch_embeddings)
        del batch_embeddings
        gc.collect()

        INSERT_BATCH = 5000
        for i in range(0, len(batch_texts), INSERT_BATCH):
            i_end = min(i + INSERT_BATCH, len(batch_texts))
            collection.add(
                ids=batch_ids[i:i_end],
                documents=batch_texts[i:i_end],
                embeddings=embeddings_list[i:i_end],
                metadatas=batch_metas[i:i_end],
            )
        print(f"[worker] Inserted batch {batch_idx + 1}/{num_batches} ({len(batch_texts)} chunks)")
        del batch_texts, batch_ids, batch_metas, embeddings_list
        gc.collect()

    status_dict["running"] = False
    status_dict["phase"] = "done"
    print(f"[worker] Complete: {total_chunks} chunks from {total} files")


class RAGEngine:
    def __init__(self):
        self.embedding_model = None
        self.client = None
        self.chroma_client = None
        self.collection = None
        self.persist_dir = CHROMA_DB_PATH

    def load_embedding_model(self):
        if USE_FINETUNED:
            path = FINETUNED_EMBEDDING_PATH
            print(f"Loading fine-tuned embedding from: {path}")
        else:
            path = EMBEDDING_MODEL
            print(f"Loading base embedding: {path}")
        self.embedding_model = SentenceTransformer(path)
        print("Embedding model loaded successfully")

    def init_chroma(self, persist_directory=None):
        if persist_directory:
            self.persist_dir = persist_directory
        self.chroma_client = chromadb.PersistentClient(path=self.persist_dir)
        self.collection = self.chroma_client.get_or_create_collection("akreditasi_docs")
        print(f"ChromaDB initialized at {self.persist_dir}")

    def init_openai(self):
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        self.client = OpenAI(api_key=api_key)

    def add_documents(self, pdf_files):
        if not self.embedding_model:
            self.load_embedding_model()

        all_texts = []
        all_ids = []
        all_metadatas = []

        for pdf_file in pdf_files:
            if not os.path.exists(pdf_file):
                print(f"File not found: {pdf_file}")
                continue
            print(f"Processing: {pdf_file}")
            chunks, chunk_pages, course_code, chunk_meta = extract_chunks(pdf_file)
            for i, chunk in enumerate(chunks):
                all_texts.append(chunk)
                import hashlib
                path_hash = hashlib.md5(pdf_file.encode()).hexdigest()[:12]
                all_ids.append(f"{path_hash}_{i}")
                meta = chunk_meta[i] if i < len(chunk_meta) else {}
                all_metadatas.append({
                    "source": os.path.basename(pdf_file),
                    "chunk_id": i,
                    "course_code": course_code or meta.get("course_code", ""),
                    "page_number": chunk_pages[i] if i < len(chunk_pages) else 0,
                    "chunk_type": meta.get("chunk_type", "text"),
                })

        if all_texts:
            DB_BATCH_SIZE = 5000
            EMBED_BATCH_SIZE = 16
            print(f"Embedding {len(all_texts)} chunks (batch size {EMBED_BATCH_SIZE})...")
            embeddings_list = []
            for i in range(0, len(all_texts), EMBED_BATCH_SIZE):
                batch = all_texts[i:i+EMBED_BATCH_SIZE]
                batch_emb = self.embedding_model.encode(batch, show_progress_bar=True, batch_size=8)
                embeddings_list.extend(batch_emb.tolist())
                import gc; gc.collect()
            print(f"Adding {len(all_texts)} chunks to ChromaDB in batches...")
            for i in range(0, len(all_texts), DB_BATCH_SIZE):
                end = min(i + DB_BATCH_SIZE, len(all_texts))
                print(f"  Batch {i // DB_BATCH_SIZE + 1}/{(len(all_texts) + DB_BATCH_SIZE - 1) // DB_BATCH_SIZE} ({end - i} chunks)...")
                self.collection.add(
                    ids=all_ids[i:end],
                    documents=all_texts[i:end],
                    embeddings=embeddings_list[i:end],
                    metadatas=all_metadatas[i:end],
                )
            print(f"Added {len(all_texts)} chunks to ChromaDB")
        else:
            print("No chunks to add")
        return len(all_texts)

    def add_documents_batched(self, pdf_files, status_dict):
        """Legacy: runs _reindex_worker in-process (backward compat)."""
        _reindex_worker(pdf_files, self.persist_dir, status_dict,
                        embedding_model=self.embedding_model,
                        collection=self.collection)

    def delete_document(self, filename):
        if not self.collection:
            return
        self.collection.delete(where={"source": filename})
        count = self.collection.count()
        print(f"Deleted chunks for '{filename}'. Collection now has {count} chunks.")

    def search(self, query, top_k=ACADEMIC_TOP_K, course_code=None):
        if not self.embedding_model:
            self.load_embedding_model()

        query_embedding = self.embedding_model.encode([query])
        where_clause = {"course_code": course_code} if course_code else None

        results = self.collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=top_k,
            where=where_clause,
            include=["documents", "metadatas", "distances"],
        )
        return results

    def generate(self, prompt):
        try:
            if not self.client:
                self.init_openai()
            response = self.client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=1024,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"OpenAI API error: {e}")
            return None

    def build_prompt(self, context, question):
        return f"""Anda adalah asisten akademik program studi Ilmu Informatika.
Anda HANYA boleh menjawab berdasarkan KONTEKS di bawah ini.
JANGAN gunakan pengetahuan atau informasi dari luar konteks.
Jika konteks tidak cukup atau tidak relevan, jawab: "Tidak ditemukan dalam dokumen."
Jika konteks mengandung tabel (format pipe-separated), baca tabel tersebut untuk menjawab pertanyaan.
Setiap bagian konteks diawali dengan [Halaman X] yang menunjukkan nomor halaman sumber informasi tersebut.
Sebutkan nomor halaman di akhir setiap poin jawaban, contoh: "(Halaman 1)".
Jawab dalam bahasa Indonesia.

KONTEKS:
{context}

PERTANYAAN:
{question}

JAWABAN (hanya dari konteks di atas):"""

    def is_relevant(self, distances):
        if not distances:
            return False
        best = min(distances)
        threshold = FINETUNED_THRESHOLD if USE_FINETUNED else SIMILARITY_THRESHOLD
        print(f"Best distance: {best:.4f} (threshold: {threshold})")
        return best <= threshold

    def _is_academic_query(self, question):
        q = question.lower()
        return any(kw in q for kw in ACADEMIC_KEYWORDS)

    def _detect_course_code(self, question):
        match = re.search(r'\b(IF|TK|SI)\d{5}\b', question.upper())
        return match.group(0) if match else None

    def _extract_course_from_query(self, question):
        q = question.lower()
        for name, code in COURSE_KEYWORDS.items():
            if name in q:
                return code
        return None

    @staticmethod
    def _levenshtein_distance(a, b):
        if len(a) < len(b):
            a, b = b, a
        if len(b) == 0:
            return len(a)
        prev = range(len(b) + 1)
        for i, ca in enumerate(a):
            curr = [i + 1]
            for j, cb in enumerate(b):
                cost = 0 if ca == cb else 1
                curr.append(min(curr[j] + 1, prev[j + 1] + 1, prev[j] + cost))
            prev = curr
        return prev[len(b)]

    def _build_typo_keywords(self):
        words = set()
        for kw in ACADEMIC_KEYWORDS:
            for w in kw.split():
                w = w.strip("-").strip()
                if len(w) > 2:
                    words.add(w.lower())
        for kw in COURSE_KEYWORDS:
            for w in kw.split():
                w = w.strip("-").strip()
                if len(w) > 2:
                    words.add(w.lower())
        return words

    def _has_potential_typo(self, question):
        keywords = self._build_typo_keywords()
        q_words = re.findall(r"[a-zA-Z]+", question)
        for qw in q_words:
            qw_lower = qw.lower()
            if len(qw_lower) < 5:
                continue
            if qw_lower in STOP_WORDS:
                continue
            if qw_lower in keywords:
                continue
            for kw in keywords:
                if len(kw) < SHORT_KEYWORD_MIN:
                    continue
                d = self._levenshtein_distance(qw_lower, kw)
                if d <= MAX_TYPO_DISTANCE:
                    return True
        return False

    def suggest_correction(self, question):
        if not self._has_potential_typo(question):
            return {"original": question, "corrected": None}
        try:
            if not self.client:
                self.init_openai()
            resp = self.client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": "Anda adalah korektor ejaan untuk query akademik. Koreksi typo dalam query berikut.\n\nAturan:\n- HANYA ubah kata yang jelas salah ketik (jarak 1-2 huruf dari kata yang benar)\n- JANGAN mengubah: kata umum (apa, saja, yang, dalam, dan, itu, adalah, bisa, bagaimana, dll), singkatan yang sudah benar (LED, LKPS, IAPS, BAN-PT, LAM, dll), kata yang sudah ejaan dan konteksnya benar\n- Jika ragu, JANGAN ubah apapun\n- Kembalikan query yang sudah diperbaiki. Jika tidak ada perubahan, kembalikan query asli persis."},
                    {"role": "user", "content": question},
                ],
                temperature=0,
                max_tokens=256,
            )
            corrected = resp.choices[0].message.content.strip()
            if corrected and corrected.lower() != question.lower():
                return {"original": question, "corrected": corrected}
            return {"original": question, "corrected": None}
        except Exception as e:
            print(f"Suggest correction error: {e}")
            return {"original": question, "corrected": None}

    def generate_stream(self, prompt):
        try:
            if not self.client:
                self.init_openai()
            stream = self.client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=1024,
                stream=True,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    yield delta.content
        except Exception as e:
            print(f"OpenAI stream error: {e}")
            yield None

    def answer_question_stream(self, question, top_k=ACADEMIC_TOP_K):
        try:
            course_code = self._detect_course_code(question)
            if not course_code:
                course_code = self._extract_course_from_query(question)

            is_academic = self._is_academic_query(question)
            yield {
                "type": "debug",
                "stage": "query_analysis",
                "detail": {
                    "question": question,
                    "course_code": course_code,
                    "is_academic": is_academic,
                    "passed": bool(course_code) or is_academic,
                },
            }

            if not course_code and not is_academic:
                yield {"type": "status", "status": "irrelevant"}
                return

            results = self.search(question, top_k, course_code=course_code)
            docs = results["documents"][0] if results.get("documents") else []
            metas = results["metadatas"][0] if results.get("metadatas") else []
            dists = results.get("distances", [[]])[0] if results.get("distances") else []

            best_dist = min(dists) if dists else None
            relevant = self.is_relevant(dists) if docs else False

            chunk_previews = []
            for doc, meta, dist in zip(docs, metas, dists):
                preview = doc[:200].replace("\n", " ")
                chunk_previews.append({
                    "source": meta.get("source", "?"),
                    "page": meta.get("page_number", "?"),
                    "distance": round(dist, 4),
                    "preview": preview,
                })

            yield {
                "type": "debug",
                "stage": "retrieval",
                "detail": {
                    "total_chunks": len(chunk_previews),
                    "best_distance": best_dist,
                    "threshold": FINETUNED_THRESHOLD if USE_FINETUNED else SIMILARITY_THRESHOLD,
                    "is_relevant": relevant,
                    "chunks": chunk_previews,
                },
            }

            if not docs or not relevant:
                yield {"type": "status", "status": "irrelevant"}
                return

            context_parts = []
            for doc, meta in zip(docs, metas):
                page = meta.get("page_number", "?")
                context_parts.append(f"[Halaman {page}]\n{doc}")
            context = "\n\n".join(context_parts)
            prompt = self.build_prompt(context, question)

            yield {
                "type": "debug",
                "stage": "prompt",
                "detail": {
                    "context_length": len(context),
                    "prompt_length": len(prompt),
                    "context_preview": context[:500],
                },
            }

            for token in self.generate_stream(prompt):
                if token is None:
                    break
                yield {"type": "token", "token": token}

            sources = {}
            for m in metas:
                src = m["source"]
                if src not in sources:
                    sources[src] = set()
                sources[src].add(m.get("page_number", 0))
                if len(sources) >= 3:
                    break
            sources = {src: sorted(pages) for src, pages in sources.items()}
            yield {"type": "status", "status": "sources", "sources": sources}
            yield {"type": "done"}

        except Exception as e:
            import traceback
            traceback.print_exc()
            yield {"type": "error", "message": str(e)}
            yield {"type": "done"}

    def answer_question(self, question, top_k=ACADEMIC_TOP_K):
        try:
            course_code = self._detect_course_code(question)
            if not course_code:
                course_code = self._extract_course_from_query(question)

            if not course_code and not self._is_academic_query(question):
                print(f"Non-academic query rejected: {question}")
                return {
                    "answer": "Maaf, pertanyaan Anda tidak memiliki korelasi dengan data yang tersedia.",
                    "sources": {},
                    "context": "",
                }

            results = self.search(question, top_k, course_code=course_code)
            docs = results["documents"][0] if results.get("documents") else []
            metas = results["metadatas"][0] if results.get("metadatas") else []
            dists = results.get("distances", [[]])[0] if results.get("distances") else []

            if not docs or not self.is_relevant(dists):
                return {
                    "answer": "Maaf, pertanyaan Anda tidak memiliki korelasi dengan data yang tersedia.",
                    "sources": {},
                    "context": "",
                }

            context_parts = []
            for doc, meta in zip(docs, metas):
                page = meta.get("page_number", "?")
                context_parts.append(f"[Halaman {page}]\n{doc}")
            context = "\n\n".join(context_parts)
            prompt = self.build_prompt(context, question)
            answer = self.generate(prompt)

            if answer and len(answer.strip()) > 0:
                sources = {}
                for m in metas:
                    src = m["source"]
                    if src not in sources:
                        sources[src] = set()
                    sources[src].add(m.get("page_number", 0))
                    if len(sources) >= 3:
                        break
                sources = {src: sorted(pages) for src, pages in sources.items()}
                return {"answer": answer, "sources": sources, "context": context}

            return {
                "answer": "Maaf, data tidak ditemukan.",
                "sources": {},
                "context": context,
            }

        except Exception as e:
            import traceback
            print(f"Error: {e}")
            traceback.print_exc()
            return {
                "answer": f"Error: {type(e).__name__}: {str(e)}",
                "sources": {},
                "context": "",
            }
