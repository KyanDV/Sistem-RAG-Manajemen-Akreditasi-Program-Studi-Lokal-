import argparse
import gc
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from openai import OpenAI

from app.config import (
    ACADEMIC_TOP_K,
    CHROMA_DB_PATH,
    EMBEDDING_MODEL,
    FINETUNED_EMBEDDING_PATH,
    LLM_MODEL,
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUT_DIR = os.path.join(DATA_DIR, "bertscore")
VALIDATION_PATH = os.path.join(DATA_DIR, "training", "validation_triplets.json")
RESULTS_LEGACY = os.path.join(BASE_DIR, "bertscore_results.json")

COLLECTION = "akreditasi_docs"
COLLECTION_BASE = "akreditasi_docs_base"

ZHIPU_BASE_URL = os.environ.get("ZHIPU_BASE_URL", "https://open.bigmodel.cn/api/paas/v4/")
ZHIPU_MODEL = os.environ.get("ZHIPU_MODEL", "glm-4.7-flash")

BATCH = 10


def zhipu_key():
    key = os.environ.get("ZHIPU_API_KEY", "").strip()
    if not key:
        raise SystemExit("ZHIPU_API_KEY belum diset (export ZHIPU_API_KEY=...)")
    return key


def _client_for(llm):
    if llm == "openai":
        return OpenAI()
    return OpenAI(api_key=zhipu_key(), base_url=ZHIPU_BASE_URL)


def _model_for(llm):
    return LLM_MODEL if llm == "openai" else ZHIPU_MODEL


def _max_tokens_for(llm):
    return 1024 if llm == "openai" else 2048


def _chat(client, model, messages, temperature, max_tokens=1024, retries=6):
    last = None
    for i in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            content = resp.choices[0].message.content
            if not (content or "").strip():
                raise RuntimeError("empty completion")
            return content
        except Exception as e:
            last = e
            time.sleep(3 * (i + 1))
    raise RuntimeError(f"LLM call failed: {last}")


def _save(name, data):
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, name), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
    print(f"Saved {name}")


def _load(name):
    path = os.path.join(OUT_DIR, name)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _load_json_any(path):
    with open(path, encoding="utf-8") as f:
        text = f.read()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return [json.loads(line) for line in text.splitlines() if line.strip()]


def load_validation():
    return _load_json_any(VALIDATION_PATH)


def _norm_text(value):
    if isinstance(value, list):
        return "\n".join(str(v) for v in value)
    return str(value)


def _gold_prompt(context, question):
    return f"""Berdasarkan konteks dokumen berikut, buatlah jawaban yang lengkap, ringkas, dan akurat dalam bahasa Indonesia untuk pertanyaan yang diberikan. Jawab hanya dari konteks.

KONTEKS:
{context}

PERTANYAAN:
{question}

JAWABAN:"""


def stage_ping(args):
    client = _client_for("zhipu")
    out = _chat(
        client, ZHIPU_MODEL,
        [{"role": "user", "content": "Sebutkan 3 contoh dokumen akreditasi program studi dalam satu baris."}],
        temperature=0.3, max_tokens=2048,
    )
    print(f"Zhipu OK ({ZHIPU_MODEL}): {out[:120]!r}")


def stage_gold(args):
    existing = _load("gold_answers.json")
    if existing and len(existing) == 100:
        print("gold_answers.json sudah ada, dipakai ulang.")
        return
    if os.path.exists(RESULTS_LEGACY):
        legacy = _load_json_any(RESULTS_LEGACY)
        if isinstance(legacy, dict):
            pq = legacy.get("per_query") or legacy.get("per-query") or []
            if isinstance(pq, list) and len(pq) == 100 and all(
                isinstance(i, dict) and i.get("gold") for i in pq
            ):
                _save("gold_answers.json", [{"query": i["query"], "gold": i["gold"]} for i in pq])
                print("Gold diambil dari bertscore_results.json (per_query).")
                return
        elif isinstance(legacy, list) and len(legacy) == 100 and all(
            isinstance(i, dict) and i.get("gold") for i in legacy
        ):
            _save("gold_answers.json", [{"query": i["query"], "gold": i["gold"]} for i in legacy])
            print("Gold diambil dari bertscore_results.json lama.")
            return
    client = _client_for("openai")
    triplets = load_validation()
    golds = []
    for i, t in enumerate(triplets):
        prompt = _gold_prompt(_norm_text(t["positive"]), t["query"])
        gold = _chat(client, LLM_MODEL, [{"role": "user", "content": prompt}], temperature=0.0)
        golds.append({"query": t["query"], "gold": gold})
        print(f"[{i+1}/100] gold ok")
        time.sleep(0.2)
    _save("gold_answers.json", golds)


def stage_retrieve(args):
    import chromadb
    from sentence_transformers import SentenceTransformer

    embedding = args.embedding
    top_k = args.top_k
    if embedding == "finetuned":
        model = SentenceTransformer(FINETUNED_EMBEDDING_PATH)
        col_name = COLLECTION
    else:
        model = SentenceTransformer(EMBEDDING_MODEL)
        col_name = COLLECTION_BASE

    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    names = [c.name for c in client.list_collections()]
    print("Collections:", names)
    if col_name not in names:
        raise SystemExit(f"Collection '{col_name}' tidak ada. Jalankan stage reindex-base (jika base) atau cek VM.")
    col = client.get_collection(col_name)

    triplets = load_validation()
    queries = [t["query"] for t in triplets]
    out = []
    for i in range(0, len(queries), BATCH):
        qs = queries[i:i + BATCH]
        embs = model.encode(qs).tolist()
        res = col.query(
            query_embeddings=embs,
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        for q, docs, metas, dists in zip(qs, res["documents"], res["metadatas"], res["distances"]):
            out.append({
                "query": q,
                "docs": docs,
                "pages": [m.get("page_number", 0) for m in metas],
                "distances": dists,
            })
        del embs
        gc.collect()
        print(f"[{min(i + BATCH, len(queries))}/{len(queries)}] retrieved ({embedding})")
    _save(f"retrieved_{embedding}.json", out)


def stage_reindex_base(args):
    import torch

    torch.set_num_threads(min(2, os.cpu_count() or 2))
    import chromadb
    from sentence_transformers import SentenceTransformer

    from app.chunker import extract_chunks
    from app.config import DATA_DIRS

    model = SentenceTransformer(EMBEDDING_MODEL)
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    ref_col = client.get_collection(COLLECTION)
    ref = ref_col.get(include=["metadatas"])
    allowed = {m.get("source", "") for m in ref["metadatas"]}
    print(f"Korpus rujukan {COLLECTION}: {len(ref['metadatas'])} chunks, {len(allowed)} source", flush=True)
    col = client.get_or_create_collection(COLLECTION_BASE)
    try:
        client.delete_collection(COLLECTION_BASE)
        print(f"Collection {COLLECTION_BASE} dihapus (fresh).", flush=True)
    except Exception as e:
        print(f"delete_collection skip: {e}", flush=True)
    col = client.get_or_create_collection(COLLECTION_BASE)

    files = []
    seen = set()
    for d in DATA_DIRS:
        for root, _, fnames in os.walk(d):
            for fn in sorted(fnames):
                if fn.lower().endswith(".pdf") and fn in allowed and fn not in seen:
                    seen.add(fn)
                    files.append(os.path.join(root, fn))
    print(f"{len(files)} pdf unik ditemukan (scoped ke korpus rujukan)", flush=True)

    added = 0
    for p in files:
        chunks, pages, code, metas = extract_chunks(p)
        batch = []
        for c, pg, meta in zip(chunks, pages, metas):
            batch.append((c, pg, meta))
        if not batch:
            print(f"  SKIP file tanpa chunk: {os.path.basename(p)} (code={code})", flush=True)
            continue
        embs = model.encode([c for c, _, _ in batch]).tolist()
        col.add(
            ids=[f"{os.path.basename(p)}#{added + j}" for j in range(len(batch))],
            embeddings=embs,
            documents=[c for c, _, _ in batch],
            metadatas=[{"source": os.path.basename(p), "page_number": pg, "chunk_type": m["chunk_type"], "course_code": m.get("course_code", "")} for _, pg, m in batch],
        )
        added += len(batch)
        print(f"  {os.path.basename(p)}: {len(batch)} chunks (total {added})", flush=True)
        gc.collect()
    print(f"Done. Collection {COLLECTION_BASE}: {added} chunks", flush=True)


def _build_context(item):
    parts = []
    for doc, page in zip(item["docs"], item["pages"]):
        parts.append(f"[Halaman {page}] {doc}")
    return "\n\n".join(parts)


def stage_generate(args):
    retrieved = _load(f"retrieved_{args.embedding}.json")
    if not retrieved:
        raise SystemExit("File retrieved belum ada. Jalankan stage retrieve dulu.")
    golds = _load("gold_answers.json")
    if not golds:
        raise SystemExit("gold_answers.json belum ada. Jalankan stage gold dulu.")

    client = _client_for(args.llm)
    model = _model_for(args.llm)
    answers = []
    for i, item in enumerate(retrieved):
        context = _build_context(item)
        prompt = f"""Anda adalah asisten akademik program studi Ilmu Informatika.
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
{item['query']}

JAWABAN (hanya dari konteks di atas):"""
        try:
            answer = _chat(client, model, [{"role": "user", "content": prompt}], temperature=0.1, max_tokens=_max_tokens_for(args.llm))
        except Exception as e:
            print(f"[{i+1}/{len(retrieved)}] {args.llm}/{args.embedding} FAIL: {e}")
            answer = ""
        answers.append({"query": item["query"], "answer": answer})
        print(f"[{i+1}/{len(retrieved)}] {args.llm}/{args.embedding} ok")
        time.sleep(args.sleep)
    _save(f"answers_{args.embedding}_{args.llm}.json", answers)


def stage_score(args):
    from bert_score import BERTScorer

    golds = _load("gold_answers.json")
    if not golds:
        raise SystemExit("gold_answers.json belum ada.")
    refs = [g["gold"] for g in golds]

    scorer = BERTScorer(
        lang="id",
        model_type="microsoft/deberta-xlarge-mnli",
        device="cpu",
        batch_size=BATCH,
    )
    if getattr(scorer, "_model_max_length", None):
        scorer._model_max_length = 512
    for _attr in ("_tokenizer", "_model_tokenizer", "tokenizer"):
        _tok = getattr(scorer, _attr, None)
        if _tok is not None:
            _tok.model_max_length = 512

    matrix = {}
    for name in sorted(os.listdir(OUT_DIR)):
        if not name.startswith("answers_") or not name.endswith(".json"):
            continue
        comb = name[len("answers_"):-len(".json")]
        data = _load(name)
        pairs = [(d["answer"], refs[i]) for i, d in enumerate(data) if d.get("answer")]
        if len(pairs) != len(refs):
            print(f"WARN {comb}: hanya {len(pairs)}/{len(refs)} jawaban valid")
        cands = [c for c, _ in pairs]
        refs_sub = [r for _, r in pairs]
        P, R, F = scorer.score(cands, refs_sub, verbose=False, batch_size=2)
        matrix[comb] = {
            "n": len(pairs),
            "precision": round(P.mean().item() * 100, 2),
            "recall": round(R.mean().item() * 100, 2),
            "f1": round(F.mean().item() * 100, 2),
        }
        print(f"{comb}: P={matrix[comb]['precision']} R={matrix[comb]['recall']} F1={matrix[comb]['f1']}")
        del P, R, F, cands, refs_sub
        gc.collect()

    with open(os.path.join(OUT_DIR, "bertscore_matrix.json"), "w", encoding="utf-8") as f:
        json.dump(matrix, f, ensure_ascii=False, indent=1)
    print("\n=== MATRIK BERTSCORE ===")
    for comb, m in matrix.items():
        print(f"{comb:28s} P={m['precision']:6.2f}  R={m['recall']:6.2f}  F1={m['f1']:6.2f}")


def stage_inspect(args):
    import gc

    import chromadb

    from sentence_transformers import SentenceTransformer

    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    names = [c.name for c in client.list_collections()]
    print("Collections:", names)
    for n in names:
        c = client.get_collection(n)
        print(f"  {n}: {c.count()} chunks")

    col = client.get_collection(COLLECTION)
    sample_q = ["Siapa dosen pengampu mata kuliah kalkulus?",
                "Bagaimana komponen penilaian pada RPS?"]
    for label, model_path in [("base", EMBEDDING_MODEL), ("finetuned", FINETUNED_EMBEDDING_PATH)]:
        model = SentenceTransformer(model_path)
        embs = model.encode(sample_q).tolist()
        res = col.query(query_embeddings=embs, n_results=3, include=["distances"])
        print(f"query {label}: top1 dist = {[round(d[0], 3) for d in res['distances']]}")
        del model, embs, res
        gc.collect()

    for path in (VALIDATION_PATH, RESULTS_LEGACY):
        if os.path.exists(path):
            data = _load_json_any(path)
            first = data[0] if isinstance(data, list) and data else data
            print(f"\n{os.path.basename(path)}: list({len(data)}) keys={list(first.keys()) if isinstance(first, dict) else type(first)}")
        else:
            print(f"\n{os.path.basename(path)}: tidak ada")

    print(f"ZHIPU_API_KEY set: {bool(os.environ.get('ZHIPU_API_KEY'))}")
    print(f"OPENAI_API_KEY set: {bool(os.environ.get('OPENAI_API_KEY'))}")


def main():
    parser = argparse.ArgumentParser(description="Evaluasi BERTScore 2x2 (embedding x LLM)")
    sub = parser.add_subparsers(dest="stage", required=True)

    p = sub.add_parser("ping")
    p.set_defaults(func=stage_ping)

    p = sub.add_parser("gold")
    p.set_defaults(func=stage_gold)

    p = sub.add_parser("retrieve")
    p.add_argument("--embedding", choices=["base", "finetuned"], required=True)
    p.add_argument("--top-k", type=int, default=3)
    p.set_defaults(func=stage_retrieve)

    p = sub.add_parser("reindex-base")
    p.set_defaults(func=stage_reindex_base)

    p = sub.add_parser("generate")
    p.add_argument("--embedding", choices=["base", "finetuned"], required=True)
    p.add_argument("--llm", choices=["openai", "zhipu"], required=True)
    p.add_argument("--sleep", type=float, default=0.3)
    p.set_defaults(func=stage_generate)

    p = sub.add_parser("score")
    p.set_defaults(func=stage_score)

    p = sub.add_parser("inspect")
    p.set_defaults(func=stage_inspect)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
