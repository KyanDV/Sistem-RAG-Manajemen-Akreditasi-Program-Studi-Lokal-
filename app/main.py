import json
import os
import threading
from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel
from typing import Optional, List
from app.rag_engine import RAGEngine
from app.config import DATA_DIRS, CHROMA_DB_PATH, EMBEDDING_MODEL, FINETUNED_EMBEDDING_PATH, USE_FINETUNED

app = FastAPI(title="RAG Akreditasi API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(SCRIPT_DIR, "static")
if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class QuestionRequest(BaseModel):
    question: str
    top_k: Optional[int] = 12


class AnswerResponse(BaseModel):
    answer: str
    sources: dict


rag_engine = None
reindex_status = {"running": False}
_reindex_lock = threading.Lock()

_index_queue = []
_index_lock = threading.Lock()
_index_status = {"total": 0, "completed": 0, "current_file": "", "errors": [], "warnings": []}

def _index_worker():
    while True:
        item = None
        with _index_lock:
            if _index_queue:
                item = _index_queue.pop(0)
                if item:
                    _index_status["current_file"] = item[1]
        if item:
            filepath, filename = item
            # Safety: skip if already indexed (same filename, has chunks)
            try:
                coll_data = rag_engine.collection.get(where={"source": filename})
                if coll_data and len(coll_data.get("ids", [])) > 0:
                    print(f"Skipping {filename}: already in ChromaDB ({len(coll_data['ids'])} chunks)")
                    with _index_lock:
                        _index_status["completed"] += 1
                        _index_status["warnings"].append(f"{filename}: dilewati (sudah terindex)")
                        if _index_status["completed"] >= _index_status["total"]:
                            _index_status["current_file"] = ""
                    continue
            except Exception:
                pass
            try:
                rag_engine.delete_document(filename)
                chunk_count = rag_engine.add_documents([filepath])
                if chunk_count == 0:
                    with _index_lock:
                        _index_status["warnings"].append(f"{filename}: PDF tidak memiliki teks yang bisa diekstrak")
            except Exception as e:
                with _index_lock:
                    _index_status["errors"].append(f"{filename}: {str(e)}")
            with _index_lock:
                _index_status["completed"] += 1
                if _index_status["completed"] >= _index_status["total"]:
                    _index_status["current_file"] = ""
        else:
            with _index_lock:
                # Clear stale status if worker thread crashed mid-process
                if _index_status["current_file"] and _index_status["completed"] >= _index_status["total"]:
                    _index_status["current_file"] = ""
                    _index_status["errors"].append(f"Auto-recovered: cleared stale current_file")
                elif _index_status["current_file"] and _index_status["completed"] < _index_status["total"]:
                    # Still processing — don't auto-recover, let it finish
                    pass
            import time
            time.sleep(0.5)

def _run_reindex_thread(pdf_files, engine, status_dict):
    """Reindex files one-by-one to avoid OOM. Skips files already in ChromaDB."""
    import gc
    try:
        total = len(pdf_files)
        status_dict["total_files"] = total

        existing = set()
        try:
            all_data = engine.collection.get(include=[])
            for meta in all_data.get("metadatas") or []:
                if meta and "source" in meta:
                    existing.add(meta["source"])
        except Exception:
            pass

        indexed = 0
        for idx, pdf_path in enumerate(pdf_files):
            fname = os.path.basename(pdf_path)
            status_dict["current_file"] = idx + 1
            status_dict["current_filename"] = fname
            status_dict["phase"] = f"processing {idx + 1}/{total}"

            if fname in existing:
                print(f"[reindex] Skipping {fname} (already indexed)")
                continue

            try:
                count = engine.add_documents([pdf_path])
                indexed += 1
                existing.add(fname)
                if count == 0:
                    print(f"[reindex] Warning: {fname} has no extractable text (scanned PDF?)")
                gc.collect()
            except Exception as e:
                print(f"[reindex] Error processing {fname}: {e}")

        status_dict["running"] = False
        status_dict["phase"] = "done"
        print(f"[reindex] Complete: {indexed} new files indexed ({total - indexed} skipped)")
    except Exception as e:
        import traceback
        traceback.print_exc()
        status_dict["running"] = False
        status_dict["phase"] = "error"
        status_dict["error"] = str(e)


@app.on_event("startup")
async def startup():
    global rag_engine, reindex_status
    rag_engine = RAGEngine()
    rag_engine.init_chroma("/home/kyan67verado/rag_app/data/chroma_db")
    rag_engine.load_embedding_model()
    rag_engine.init_openai()
    reindex_status = {"running": False}
    t = threading.Thread(target=_index_worker, daemon=True)
    t.start()
    print("RAG Engine v2 initialized")


@app.get("/")
async def root():
    return {"message": "RAG Akreditasi API v2", "status": "ok"}


@app.get("/health")
async def health():
    status = "initialized" if rag_engine else "not initialized"
    return {"status": "healthy", "engine": status}


@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    if not rag_engine:
        raise HTTPException(status_code=500, detail="Engine not initialized")
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    try:
        result = rag_engine.answer_question(request.question, request.top_k)
        return AnswerResponse(answer=result["answer"], sources=result.get("sources", []))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask/stream")
async def ask_question_stream(body: QuestionRequest):
    if not rag_engine:
        raise HTTPException(status_code=500, detail="Engine not initialized")
    if not body.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    async def event_generator():
        for event in rag_engine.answer_question_stream(body.question, body.top_k):
            if event["type"] in ("token", "status", "debug"):
                yield f"data: {json.dumps(event)}\n\n"
            elif event["type"] == "done":
                yield f"data: {json.dumps(event)}\n\n"
                return
            elif event["type"] == "error":
                yield f"data: {json.dumps(event)}\n\n"
                return

    return StreamingResponse(event_generator(), media_type="text/event-stream")


class SuggestRequest(BaseModel):
    question: str


@app.post("/suggest")
async def suggest_correction(body: SuggestRequest):
    if not rag_engine:
        raise HTTPException(status_code=500, detail="Engine not initialized")
    if not body.question.strip():
        return {"original": body.question, "corrected": None}
    try:
        result = rag_engine.suggest_correction(body.question)
        return result
    except Exception as e:
        return {"original": body.question, "corrected": None, "error": str(e)}


@app.get("/reindex/status")
async def get_reindex_status():
    return dict(reindex_status)

@app.post("/reindex")
async def reindex():
    if not rag_engine:
        raise HTTPException(status_code=500, detail="Engine not initialized")
    if reindex_status.get("running"):
        raise HTTPException(status_code=409, detail="Reindex already in progress")

    try:
        pdf_files = []
        for base_dir in DATA_DIRS:
            if not os.path.exists(base_dir):
                continue
            for root, dirs, files in os.walk(base_dir):
                for f in files:
                    if f.endswith(".pdf"):
                        pdf_files.append(os.path.join(root, f))

        if not pdf_files:
            raise HTTPException(status_code=400, detail="No PDF files found")

        status = reindex_status
        status["running"] = True
        status["total_files"] = len(pdf_files)
        status["current_file"] = 0
        status["current_filename"] = ""
        status["phase"] = "starting"
        status["total_chunks"] = 0
        status["embedded_chunks"] = 0
        status["error"] = None

        t = threading.Thread(target=_run_reindex_thread,
                             args=(pdf_files, rag_engine, status),
                             daemon=True)
        t.start()

        return {"status": "started", "message": f"Indexing {len(pdf_files)} PDFs in background"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/files")
async def list_files():
    files = []
    for base_dir in DATA_DIRS:
        if not os.path.exists(base_dir):
            continue
        for root, dirs, fnames in os.walk(base_dir):
            for f in fnames:
                if f.endswith('.pdf'):
                    files.append({"filename": f})
    seen = set()
    unique = []
    for f in files:
        if f["filename"] not in seen:
            seen.add(f["filename"])
            unique.append(f)
    return {"files": unique}


@app.get("/download/{filename}")
async def download_file(filename: str):
    from urllib.parse import unquote
    filename = unquote(filename)
    for base_dir in DATA_DIRS:
        if not os.path.exists(base_dir):
            continue
        for root, dirs, fnames in os.walk(base_dir):
            for f in fnames:
                if f == filename:
                    filepath = os.path.join(root, f)
                    return FileResponse(
                        filepath,
                        media_type='application/pdf',
                        headers={"Content-Disposition": f"attachment; filename={filename}"}
                    )
    raise HTTPException(status_code=404, detail="File not found")


@app.get("/view/{filename}")
async def view_file(filename: str):
    from urllib.parse import unquote
    filename = unquote(filename)
    for base_dir in DATA_DIRS:
        if not os.path.exists(base_dir):
            continue
        for root, dirs, fnames in os.walk(base_dir):
            for f in fnames:
                if f == filename:
                    filepath = os.path.join(root, f)
                    return FileResponse(
                        filepath,
                        media_type='application/pdf',
                        headers={"Content-Disposition": f"inline; filename={filename}"}
                    )
    raise HTTPException(status_code=404, detail="File not found")


@app.get("/documents")
async def list_documents():
    if not rag_engine or not rag_engine.collection:
        return {"count": 0, "documents": []}
    try:
        data = rag_engine.collection.get()
        sources = list(set(m["source"] for m in data["metadatas"]))
        return {"count": len(data["ids"]), "documents": sources}
    except Exception as e:
        return {"count": 0, "documents": [], "error": str(e)}


@app.get("/documents/detail")
async def list_documents_detail():
    if not rag_engine or not rag_engine.collection:
        return {"files": []}
    try:
        data = rag_engine.collection.get()
        meta_by_source = {}
        for m in data["metadatas"]:
            src = m["source"]
            if src not in meta_by_source:
                meta_by_source[src] = {"chunks": 0, "pages": set()}
            meta_by_source[src]["chunks"] += 1
            if m.get("page_number"):
                meta_by_source[src]["pages"].add(m["page_number"])

        result = []
        for src, info in meta_by_source.items():
            file_size = 0
            for base_dir in DATA_DIRS:
                if not os.path.exists(base_dir):
                    continue
                for root, dirs, fnames in os.walk(base_dir):
                    for f in fnames:
                        if f == src:
                            fp = os.path.join(root, f)
                            file_size = os.path.getsize(fp)
                            break
            result.append({
                "filename": src,
                "chunks": info["chunks"],
                "pages": len(info["pages"]),
                "size": file_size,
            })
        return {"files": result}
    except Exception as e:
        return {"files": [], "error": str(e)}


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    if not rag_engine:
        raise HTTPException(status_code=500, detail="Engine not initialized")

    filename = os.path.basename(file.filename or "unknown.pdf")

    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    content = await file.read()
    if len(content) > 50 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large (max 50MB)")

    if not content.startswith(b"%PDF"):
        raise HTTPException(status_code=400, detail="Invalid PDF file (wrong magic bytes)")

    save_path = None
    for base_dir in DATA_DIRS:
        if not os.path.exists(base_dir):
            continue
        candidate = os.path.join(base_dir, filename)
        if os.path.exists(candidate):
            save_path = candidate
            break

    if not save_path:
        save_dir = None
        for d in DATA_DIRS:
            if os.path.exists(d):
                save_dir = d
                break
        if not save_dir:
            raise HTTPException(status_code=500, detail="No data directory found")
        save_path = os.path.join(save_dir, filename)
        with open(save_path, "wb") as f:
            f.write(content)

    with _index_lock:
        # Check for duplicates in queue
        already_queued = any(item[1] == filename for item in _index_queue)
        if already_queued or _index_status["current_file"] == filename:
            return {"status": "skipped", "filename": filename, "reason": "already queued"}
        _index_status["total"] += 1
        _index_queue.append((save_path, filename))
    return {"status": "accepted", "filename": filename}


@app.get("/indexing_status")
async def get_indexing_status():
    with _index_lock:
        status = dict(_index_status, queue_size=len(_index_queue))
        # Auto-recover: if queue empty, no active file, but completed < total
        if status["queue_size"] == 0 and not status["current_file"] and status["completed"] < status["total"]:
            _index_status["completed"] = status["total"]
            status["completed"] = status["total"]
            _index_status["errors"].append(f"Auto-recovered: completed set to {status['total']}")
        return status


@app.delete("/file/{filename}")
async def delete_file(filename: str):
    from urllib.parse import unquote
    if not rag_engine:
        raise HTTPException(status_code=500, detail="Engine not initialized")
    filename = unquote(filename)

    rag_engine.delete_document(filename)

    deleted = False
    for base_dir in DATA_DIRS:
        if not os.path.exists(base_dir):
            continue
        for root, dirs, fnames in os.walk(base_dir):
            for f in fnames:
                if f == filename:
                    os.remove(os.path.join(root, f))
                    deleted = True
                    break
    if not deleted:
        raise HTTPException(status_code=404, detail=f"File '{filename}' not found on disk")

    return {"status": "ok", "filename": filename, "message": "File deleted and removed from index"}


if __name__ == "__main__":
    import uvicorn
    import os
    uvicorn.run(app, host="0.0.0.0", port=8000)
