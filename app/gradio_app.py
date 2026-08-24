import gradio as gr
import httpx
import json
import uuid
import sys
import os
import asyncio
from urllib.parse import quote

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from app.chat_store import ChatStore

API_URL = "http://localhost:8000"


def _format_size(size):
    if size < 1024:
        return f"{size} B"
    elif size < 1024 * 1024:
        return f"{size / 1024:.1f} KB"
    return f"{size / 1024 / 1024:.1f} MB"


async def refresh_docs():
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{API_URL}/documents/detail", timeout=3.0)
        if resp.status_code == 200:
            data = resp.json()
            rows = [[f["filename"], f["chunks"], f["pages"], _format_size(f["size"])] for f in data.get("files", [])]
            choices = [(f["filename"], f["filename"]) for f in data.get("files", [])]
            return rows, gr.update(choices=choices, value=None)
        return [], gr.update(choices=[], value=None)
    except Exception:
        return [], gr.update(choices=[], value=None)


async def upload_doc(files):
    if not files:
        yield "❌ Pilih file terlebih dahulu", ""
        return
    if not isinstance(files, list):
        files = [files]
    pdfs = [f for f in files if f.name.lower().endswith(".pdf")]
    if not pdfs:
        yield "❌ Tidak ada file PDF ditemukan", ""
        return

    yield "", ""

    success = 0
    errors = []
    warnings = []
    initial_completed = 0
    try:
        async with httpx.AsyncClient() as client:
            r0 = await client.get(f"{API_URL}/indexing_status", timeout=5)
            if r0.status_code == 200:
                initial_completed = r0.json().get("completed", 0)
    except Exception:
        pass

    async with httpx.AsyncClient() as client:
        for f in pdfs:
            try:
                with open(f.name, "rb") as fh:
                    resp = await client.post(
                        f"{API_URL}/upload",
                        files={"file": (os.path.basename(f.name), fh, "application/pdf")},
                        timeout=300,
                    )
                if resp.status_code == 200:
                    success += 1
                else:
                    detail = resp.json().get("detail", "Gagal upload")
                    errors.append(f"{os.path.basename(f.name)}: {detail}")
            except Exception as e:
                errors.append(f"{os.path.basename(f.name)}: {str(e)}")

    if success == 0:
        msg = "❌ Gagal" + (f"\n" + "\n".join(errors[:3]) if errors else "")
        yield msg, ""
        return

    non_pdf = len(files) - len(pdfs)
    msg_prefix = f"{success} PDF terkirim"
    if non_pdf:
        msg_prefix += f" ({non_pdf} non-PDF diabaikan)"

    target_completed = initial_completed + success
    yield msg_prefix, '<div class="folder-upload-progress"><span id="progress-text">Menunggu indeks...</span></div>'

    async with httpx.AsyncClient() as client:
        while True:
            await asyncio.sleep(2)
            try:
                resp = await client.get(f"{API_URL}/indexing_status", timeout=5)
                if resp.status_code != 200:
                    continue
                s = resp.json()
                done = s.get("completed", 0)
                current = s.get("current_file", "")
                errs = s.get("errors", [])
                warns = s.get("warnings", [])

                if done < target_completed:
                    progress_html = f'<div class="folder-upload-progress"><span id="progress-text">📄 Mengindeks {current}...</span></div>'
                    yield msg_prefix, progress_html
                else:
                    if errs:
                        new_errors = [e for e in errs if e not in errors]
                        errors.extend(new_errors)
                    if warns:
                        new_warnings = [w for w in warns if w not in warnings]
                        warnings.extend(new_warnings)
                    break
            except Exception:
                continue

    msg = f"✅ {success} PDF berhasil diindex"
    parts = []
    if warnings:
        parts.append("⚠ " + "\n".join(warnings[:2]))
    if errors:
        parts.append("❌ " + "\n".join(errors[:2]))
    if parts:
        msg += "\n" + "\n".join(parts)
    yield msg, ""


async def delete_doc(filename):
    if not filename:
        return "❌ Pilih file terlebih dahulu", gr.update(), gr.update()
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.delete(f"{API_URL}/file/{quote(filename)}", timeout=60)
        if resp.status_code == 200:
            rows, choices = await refresh_docs()
            return "✅ Berhasil dihapus dari database dan disk", rows, choices
        detail = resp.json().get("detail", "Gagal hapus")
        return f"❌ {detail}", gr.update(), gr.update()
    except Exception as e:
        return f"❌ Error: {str(e)}", gr.update(), gr.update()

EXAMPLES = [
    "Apa saja instrumen akreditasi yang perlu dikumpulkan?",
    "Bagaimana cara menyusun Laporan Evaluasi Diri (LED)?",
    "Apa saja komponen yang dinilai dalam akreditasi?",
    "Bagaimana alur proses akreditasi program studi?",
    "Apa saja standar akreditasi yang harus dipenuhi?",
    "Apa saja elemen yang dinilai pada kriteria visi, misi, tujuan, dan strategi?",
    "Elemen apa saja yang dinilai pada kriteria mahasiswa?",
    "Apa saja yang dinilai pada kriteria luaran dan capaian tridarma?",
    "Apa saja yang dinilai dalam kriteria relevansi pendidikan?",
    "Berapa lama masa berlaku SK akreditasi dan bagaimana perpanjangannya?",
]

EXAMPLE_MESSAGES = [
    {"text": ex, "display_text": ex} for ex in EXAMPLES
]


def _ensure_chatbot_format(messages):
    if not messages:
        return messages
    if isinstance(messages[0], list):
        result = []
        for user_msg, bot_msg in messages:
            result.append({"role": "user", "content": user_msg})
            result.append({"role": "assistant", "content": bot_msg})
        return result
    return messages


def format_sources(src_dict):
    if not src_dict:
        return ""
    result = "\n\n**Dokumen Referensi:**\n"
    for src_name, pages in src_dict.items():
        view_url = f"/api/view/{quote(src_name)}"
        if pages:
            view_url += f"#page={pages[0]}"
        page_str = f" (halaman {', '.join(str(p) for p in pages)})" if pages else ""
        result += f'- <a href="{view_url}" target="_blank">{src_name}</a>{page_str}\n'
    return result


def escape_html(text):
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def format_debug_html(debug_events):
    if not debug_events:
        return ""
    parts = []
    for ev in debug_events:
        stage = ev.get("stage", "?")
        d = ev.get("detail", {})
        if stage == "query_analysis":
            passed = d.get("passed", False)
            status = "✅ Lanjut" if passed else "❌ Ditolak"
            academic = "✅ Ya" if d.get("is_academic") else "❌ Tidak"
            parts.append(f"""<details style="background:#f5f5f5;border:1px solid #ddd;border-radius:6px;padding:8px 10px;margin-bottom:8px;color:#1a1a1a">
<summary style="font-weight:600;cursor:pointer;color:#1a1a1a">Query Analysis</summary>
<table style="width:100%;border-collapse:collapse;font-size:0.82rem;color:#1a1a1a">
<tr><td style="padding:2px 6px;font-weight:500;color:#333;white-space:nowrap">Pertanyaan</td><td style="padding:2px 6px;color:#1a1a1a">{escape_html(d.get('question', ''))}</td></tr>
<tr><td style="padding:2px 6px;font-weight:500;color:#333;white-space:nowrap">Course Code</td><td style="padding:2px 6px;color:#1a1a1a">{escape_html(str(d.get('course_code', '—') or '—'))}</td></tr>
<tr><td style="padding:2px 6px;font-weight:500;color:#333;white-space:nowrap">Academic Query</td><td style="padding:2px 6px;color:#1a1a1a">{academic}</td></tr>
<tr><td style="padding:2px 6px;font-weight:500;color:#333;white-space:nowrap">Status</td><td style="padding:2px 6px;color:#1a1a1a;font-weight:600">{status}</td></tr>
</table>
</details>""")
        elif stage == "retrieval":
            relevant = d.get("is_relevant", False)
            rv = "✅ Relevan" if relevant else "❌ Tidak relevan"
            rows = ""
            chunks_list = d.get("chunks", [])
            for i, c in enumerate(chunks_list, 1):
                rows += f"<tr><td style='padding:2px 6px;color:#1a1a1a'>{i}</td><td style='padding:2px 6px;color:#1a1a1a'>{escape_html(str(c.get('source', '?')))}</td><td style='padding:2px 6px;color:#1a1a1a'>{c.get('page', '?')}</td><td style='padding:2px 6px;color:#1a1a1a'>{c.get('distance', '?')}</td><td style='padding:2px 6px;color:#1a1a1a;font-size:0.78rem'>{escape_html(str(c.get('preview', ''))[:120])}</td></tr>"
            parts.append(f"""<details style="background:#f5f5f5;border:1px solid #ddd;border-radius:6px;padding:8px 10px;margin-bottom:8px;color:#1a1a1a">
<summary style="font-weight:600;cursor:pointer;color:#1a1a1a">Retrieval</summary>
<table style="width:100%;border-collapse:collapse;font-size:0.82rem;color:#1a1a1a">
<tr><td style="padding:2px 6px;font-weight:500;color:#333;white-space:nowrap">Total Chunks</td><td style="padding:2px 6px;color:#1a1a1a">{d.get('total_chunks', 0)}</td></tr>
<tr><td style="padding:2px 6px;font-weight:500;color:#333;white-space:nowrap">Best Distance</td><td style="padding:2px 6px;color:#1a1a1a">{d.get('best_distance', '—')}</td></tr>
<tr><td style="padding:2px 6px;font-weight:500;color:#333;white-space:nowrap">Threshold</td><td style="padding:2px 6px;color:#1a1a1a">{d.get('threshold', '—')}</td></tr>
<tr><td style="padding:2px 6px;font-weight:500;color:#333;white-space:nowrap">Relevant</td><td style="padding:2px 6px;color:#1a1a1a;font-weight:600">{rv}</td></tr>
</table>
<div style="margin-top:6px;font-weight:500;color:#1a1a1a">Chunks:</div>
<table style="width:100%;border-collapse:collapse;font-size:0.82rem;color:#1a1a1a">
<tr><th style="text-align:left;padding:2px 6px;border-bottom:1px solid #ccc;color:#333">#</th><th style="text-align:left;padding:2px 6px;border-bottom:1px solid #ccc;color:#333">Source</th><th style="text-align:left;padding:2px 6px;border-bottom:1px solid #ccc;color:#333">Page</th><th style="text-align:left;padding:2px 6px;border-bottom:1px solid #ccc;color:#333">Distance</th><th style="text-align:left;padding:2px 6px;border-bottom:1px solid #ccc;color:#333">Preview</th></tr>
{rows}
</table>
</details>""")
        elif stage == "prompt":
            parts.append(f"""<details style="background:#f5f5f5;border:1px solid #ddd;border-radius:6px;padding:8px 10px;margin-bottom:8px;color:#1a1a1a">
<summary style="font-weight:600;cursor:pointer;color:#1a1a1a">Prompt</summary>
<table style="width:100%;border-collapse:collapse;font-size:0.82rem;color:#1a1a1a">
<tr><td style="padding:2px 6px;font-weight:500;color:#333;white-space:nowrap">Context Length</td><td style="padding:2px 6px;color:#1a1a1a">{d.get('context_length', 0)} chars</td></tr>
<tr><td style="padding:2px 6px;font-weight:500;color:#333;white-space:nowrap">Prompt Length</td><td style="padding:2px 6px;color:#1a1a1a">{d.get('prompt_length', 0)} chars</td></tr>
</table>
<div style="margin-top:6px;font-weight:500;color:#1a1a1a">Context Preview:</div>
<pre style="font-size:0.78rem;white-space:pre-wrap;background:#fff;padding:6px;border-radius:4px;border:1px solid #eee;color:#1a1a1a;max-height:200px;overflow-y:auto">{escape_html(str(d.get('context_preview', '')))}</pre>
</details>""")
    return "\n".join(parts)


def init_app(user_id):
    store = ChatStore(user_id)
    chats = store.list()
    if chats:
        chat = chats[0]
        messages = _ensure_chatbot_format(chat.messages)
        choices = [(c.title, c.id) for c in chats]
        return store, chat.id, messages, gr.update(choices=choices, value=chat.id)
    else:
        chat = store.create()
        choices = [(chat.title, chat.id)]
        return store, chat.id, [], gr.update(choices=choices, value=chat.id)


def new_chat(store):
    chat = store.create()
    choices = [(c.title, c.id) for c in store.list()]
    return chat.id, [], gr.update(choices=choices, value=chat.id)


def delete_chat(current_id, store):
    if current_id:
        store.delete(current_id)
    chats = store.list()
    if chats:
        first = chats[0]
        messages = _ensure_chatbot_format(first.messages)
        return (
            first.id,
            messages,
            gr.update(choices=[(c.title, c.id) for c in chats], value=first.id),
        )
    else:
        chat = store.create()
        return (
            chat.id,
            [],
            gr.update(choices=[(chat.title, chat.id)], value=chat.id),
        )


def select_chat(chat_id, store):
    if not chat_id:
        return None, []
    chat = store.get(chat_id)
    if not chat:
        return None, []
    messages = _ensure_chatbot_format(chat.messages)
    return chat.id, messages


async def respond(message, history, store, current_id):
    if not message.strip():
        yield history, "", gr.update(), current_id, ""
        return

    if not current_id:
        chat = store.create()
        current_id = chat.id

    history = history or []
    history.append({"role": "user", "content": message})
    yield history, "", gr.update(), current_id, ""

    dots_msg = {"role": "assistant", "content": '<span class="typing-indicator"><span></span><span></span><span></span></span>'}
    history.append(dots_msg)
    yield history, "", gr.update(), current_id, ""

    corrected = None
    question = message
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(f"{API_URL}/suggest", json={"question": message}, timeout=30)
        if resp.status_code == 200:
            result = resp.json()
            if result.get("corrected"):
                corrected = result["corrected"]
                question = corrected
    except Exception:
        pass

    partial = ""
    sources = {}
    debug_events = []
    has_content = False
    correction_note = ""
    if corrected:
        correction_note = f"\n\n> 💡 **Maksud Anda:** _{corrected}_\n\n---\n\n"

    def set_content(history, content):
        nonlocal has_content
        history[-1]["content"] = correction_note + content
        has_content = True
        return history

    try:
        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST",
                f"{API_URL}/ask/stream",
                json={"question": question},
                timeout=300,
            ) as response:
                async for line in response.aiter_lines():
                    if not line or not line.startswith("data: "):
                        continue

                    try:
                        data = json.loads(line[6:])
                        t = data.get("type")

                        if t == "debug":
                            debug_events.append(data)
                            continue

                        if t == "status" and data["status"] == "irrelevant":
                            set_content(history, "Maaf, pertanyaan Anda tidak memiliki korelasi dengan data yang tersedia.")
                            yield history, "", gr.update(), current_id, ""
                            break

                        if t == "status" and data["status"] == "sources":
                            sources = data.get("sources", {})
                            content = partial + format_sources(sources)
                            set_content(history, content)
                            yield history, "", gr.update(), current_id, ""
                            continue

                        if t == "token":
                            partial += data["token"]
                            content = partial + format_sources(sources)
                            set_content(history, content)
                            yield history, "", gr.update(), current_id, ""

                        if t == "done":
                            break

                    except json.JSONDecodeError:
                        continue

    except Exception as e:
        set_content(history, f"Error: {e}")
        yield history, "", gr.update(), current_id, ""
        return

    store.update(current_id, history)
    choices = [(c.title, c.id) for c in store.list()]
    debug_html = format_debug_html(debug_events) if debug_events else ""
    yield history, "", gr.update(choices=choices, value=current_id), current_id, debug_html


CUSTOM_CSS = """
@media (prefers-color-scheme: light) {
  :root {
    --block-label-text-color: #000 !important;
    --block-title-text-color: #000 !important;
    --body-text-color: #000 !important;
    --body-text-color-subdued: #1a1a1a !important;
  }
  #header .desc { color: #000 !important; }
  #sidebar h3 { color: #000 !important; }
}

body { margin: 0 !important; padding: 0 !important; }
[class*="gradio-container"] { max-width: 100% !important; margin: 0 !important; padding: 0 !important; font-size: 1.1rem; }
#header { padding: 20px 0 12px; border-bottom: 2px solid var(--border-color-primary, #eee); margin-bottom: 10px; }
#header .top { display: flex; align-items: center; gap: 16px; }
#header img { height: 60px; }
#header h1 { margin: 0; font-size: 1.6rem; }
#header .desc { margin: 6px 0 0 0; font-size: 1rem; font-style: italic; }
#sidebar { border-right: 1px solid var(--border-color-primary, #ddd); padding: 0 !important; margin: 0 !important; }
.main-row { gap: 0 !important; }

#sidebar h3 { margin: 0 0 8px 0; font-size: 1rem; }
#content-area { padding: 0 24px 16px 16px; }

.typing-indicator { display: inline-flex; align-items: center; gap: 5px; padding: 8px 4px; }
.typing-indicator span { width: 8px; height: 8px; background: #888; border-radius: 50%; animation: typing-bounce 1.4s ease-in-out infinite both; }
.typing-indicator span:nth-child(1) { animation-delay: -0.32s; }
.typing-indicator span:nth-child(2) { animation-delay: -0.16s; }
.typing-indicator span:nth-child(3) { animation-delay: 0s; }
@keyframes typing-bounce { 0%, 80%, 100% { transform: scale(0); } 40% { transform: scale(1); } }

@media (max-width: 600px) {
  [class*="gradio-container"] { font-size: 1rem; }
  #header { padding: 12px 0 10px; }
  #header .top { gap: 10px; }
  #header img { height: 36px; }
  #header h1 { font-size: 1.15rem; }
  #header .desc { font-size: 0.85rem; }
  #content-area { padding: 0 8px 8px 8px; }
}

@media (max-width: 600px) {
  [class*="gradio-container"] { font-size: 1rem; }
  #header { padding: 12px 0 10px; }
  #header .top { gap: 10px; }
  #header img { height: 36px; }
  #header h1 { font-size: 1.15rem; }
  #header .desc { font-size: 0.85rem; }
  #content-area { padding: 0 8px 8px 8px; }
}

.folder-upload-progress {
  padding: 8px 12px;
  background: var(--block-background-fill);
  border: 1px solid var(--border-color-primary);
  border-radius: var(--block-radius, 8px);
  display: flex;
  align-items: center;
  gap: 12px;
  color: var(--body-text-color);
  font-size: var(--text-md);
}
.folder-upload-progress .cancel-upload-btn {
  margin-left: auto;
  padding: 4px 12px;
  border: 1px solid var(--border-color-primary);
  border-radius: var(--block-radius, 6px);
  background: var(--button-secondary-background-fill);
  color: var(--button-secondary-text-color);
  cursor: pointer;
  font-size: 13px;
  font-family: inherit;
}
.folder-upload-progress .cancel-upload-btn:hover {
  background: var(--button-secondary-background-fill-hover);
}
.hidden-refresh {
  position: fixed !important;
  top: -9999px !important;
  left: -9999px !important;
  width: 1px !important;
  height: 1px !important;
  opacity: 0 !important;
}
"""


with gr.Blocks(title="RAG Akreditasi Program Studi", fill_height=True) as demo:
    user_id = gr.BrowserState(str(uuid.uuid4()))
    store = gr.State()
    current_id = gr.State()
    doc_state = gr.State([])

    with gr.Tabs():
        with gr.TabItem("💬 Chat"):
            with gr.Row(equal_height=False, elem_classes=["main-row"]):
                with gr.Column(scale=0, min_width=220, elem_id="sidebar"):
                    gr.HTML("<h3>Riwayat Chat</h3>")
                    new_btn = gr.Button("+ Chat Baru", variant="secondary", size="sm")
                    delete_chat_btn = gr.Button("\U0001f5d1 Hapus", size="sm", variant="stop")
                    chat_selector = gr.Radio(
                        label="",
                        choices=[],
                        interactive=True,
                        container=False,
                    )

                with gr.Column(scale=1, elem_id="content-area"):
                    gr.HTML("""
                        <div id="header">
                            <div class="top">
                                <img src="/api/static/logo.png" alt="UKDC">
                                <h1>RAG Akreditasi Program Studi</h1>
                            </div>
                            <div class="desc">Asisten virtual yang membantu dosen baru dalam menjawab pertanyaan seputar akreditasi Program Studi Ilmu Informatika berdasarkan dokumen resmi.</div>
                        </div>
                    """)
                    chatbot = gr.Chatbot(
                        label="",
                        scale=1,
                        min_height=400,
                        sanitize_html=False,
                        render_markdown=True,
                        examples=EXAMPLE_MESSAGES,
                        buttons=[],
                    )
                    msg = gr.Textbox(
                        label="",
                        placeholder="Tanyakan sesuatu tentang akreditasi...",
                        lines=1,
                        container=False,
                    )
                    debug_output = gr.HTML("", visible=True)

        with gr.TabItem("📁 Dokumen"):
            doc_table = gr.Dataframe(
                headers=["Nama File", "Chunks", "Halaman", "Ukuran"],
                interactive=False,
            )
            with gr.Row():
                upload_btn = gr.UploadButton("📄 Pilih PDF", file_types=[".pdf"], file_count="multiple", variant="primary", size="sm")
                folder_upload_btn = gr.Button("📁 Upload Folder", variant="secondary", size="sm")
            upload_progress = gr.HTML("", visible=True, elem_id="upload-progress")
            refresh_trigger = gr.Button("Refresh", elem_id="refresh-trigger", elem_classes="hidden-refresh")
            upload_status = gr.Markdown("")
            with gr.Row():
                doc_selector = gr.Dropdown(label="Pilih file untuk dihapus", choices=[], interactive=True)
                delete_btn = gr.Button("🗑 Hapus", variant="stop", size="sm")
            delete_status = gr.Markdown("")
            delete_modal = gr.HTML("""<div id="delete-modal-overlay" style="display:none;position:fixed;top:0;left:0;width:100%;height:100%;background:var(--neutral-800, rgba(0,0,0,0.5));z-index:9999;align-items:center;justify-content:center;">
<div style="background:var(--block-background-fill, white);border-radius:var(--block-radius, 12px);padding:24px;min-width:360px;box-shadow:var(--shadow-drop-lg, 0 4px 20px rgba(0,0,0,0.2));">
<h3 style="margin:0 0 8px 0;font-size:16px;color:var(--block-label-text-color, #d32f2f);">Konfirmasi Hapus</h3>
<p id="modal-delete-text" style="margin:0 0 20px 0;font-size:14px;color:var(--body-text-color, #333);"></p>
<span id="pending-delete-file" style="display:none;"></span>
<div style="display:flex;gap:10px;justify-content:flex-end;">
<button onclick="document.getElementById('delete-modal-overlay').style.display='none'" style="padding:8px 20px;border:var(--button-secondary-border-color, 1px solid #ccc);border-radius:var(--button-secondary-border-radius, 6px);background:var(--button-secondary-background-fill, white);color:var(--button-secondary-text-color, #333);cursor:pointer;font-size:14px;font-family:inherit;">Batal</button>
<button onclick="(async function(){var o=document.getElementById('delete-modal-overlay');var f=document.getElementById('pending-delete-file').textContent;if(!f)return;o.style.display='none';try{await fetch('/api/file/'+encodeURIComponent(f),{method:'DELETE'});}catch(e){}var r=document.getElementById('refresh-trigger');if(r)r.click();})()" style="padding:8px 20px;border:var(--button-primary-border-color, none);border-radius:var(--button-primary-border-radius, 6px);background:var(--button-primary-background-fill, #d32f2f);color:var(--button-primary-text-color, white);cursor:pointer;font-size:14px;font-weight:600;font-family:inherit;">Ya, Hapus</button>
</div>
</div>
</div>""")

    demo.load(
        init_app,
        inputs=[user_id],
        outputs=[store, current_id, chatbot, chat_selector],
    )

    new_btn.click(
        new_chat,
        inputs=[store],
        outputs=[current_id, chatbot, chat_selector],
    )

    delete_chat_btn.click(
        delete_chat,
        inputs=[current_id, store],
        outputs=[current_id, chatbot, chat_selector],
    )

    chat_selector.change(
        select_chat,
        inputs=[chat_selector, store],
        outputs=[current_id, chatbot],
    )

    def fill_from_example(sel_data: gr.SelectData):
        val = sel_data.value
        if isinstance(val, dict):
            return val.get("text", "")
        return str(val)

    chatbot.example_select(
        fill_from_example,
        None,
        [msg],
        queue=False,
    ).then(
        respond,
        [msg, chatbot, store, current_id],
        [chatbot, msg, chat_selector, current_id, debug_output],
    )

    msg.submit(
        respond,
        inputs=[msg, chatbot, store, current_id],
        outputs=[chatbot, msg, chat_selector, current_id, debug_output],
    )

    demo.load(
        refresh_docs,
        outputs=[doc_table, doc_selector],
    )

    upload_btn.upload(
        upload_doc,
        inputs=[upload_btn],
        outputs=[upload_status, upload_progress],
        queue=True,
    ).then(
        refresh_docs,
        outputs=[doc_table, doc_selector],
    )

    def upload_folder_done():
        return ""

    refresh_trigger.click(
        refresh_docs,
        outputs=[doc_table, doc_selector],
    )

    folder_upload_btn.click(
        upload_folder_done,
        inputs=[],
        outputs=[],
        js="""() => {
    window.__cancelUpload = function(btn) {
        window.cancelFlag = true;
        btn.disabled = true;
        btn.textContent = 'Membatalkan...';
    };
    var input = document.createElement('input');
    input.type = 'file';
    input.webkitdirectory = true;
    input.multiple = true;
    input.style.cssText = 'display:none!important';
    document.body.appendChild(input);
    input.addEventListener('change', async function() {
        var files = Array.from(input.files);
        var pdfs = files.filter(function(f) { return f.name.toLowerCase().endsWith('.pdf'); });
        var total = pdfs.length;
        window.cancelFlag = false;
        var progressEl = document.getElementById('upload-progress');
        if (progressEl && total > 0) {
            progressEl.innerHTML = '<div class=\"folder-upload-progress\"><span id=\"progress-text\">Ditemukan ' + total + ' file. Mulai upload...</span><button class=\"cancel-upload-btn\" onclick=\"window.__cancelUpload(this)\">Cancel</button></div>';
        }
        var uploaded = 0, failed = 0;
        for (var i = 0; i < total; i++) {
            if (window.cancelFlag) break;
            var f = pdfs[i];
            if (progressEl) {
                var pt = progressEl.querySelector('#progress-text');
                if (pt) pt.textContent = 'Uploading ' + (i+1) + '/' + total + '...';
            }
            var fd = new FormData();
            fd.append('file', f, f.webkitRelativePath || f.name);
            try {
                var r = await fetch('/api/upload', { method: 'POST', body: fd });
                if (r.ok) uploaded++;
                else failed++;
            } catch(e) { failed++; }
        }
        input.removeEventListener('change', this);
        document.body.removeChild(input);
        window.cancelFlag = false;
        if (progressEl && total > 0) {
            progressEl.innerHTML = '<div class=\"folder-upload-progress\" id=\"progress-bar\"><span id=\"progress-text\">' + uploaded + ' terupload. Mengindeks ' + uploaded + '/' + uploaded + '...</span></div>';
        }
                var pollCount = 0;
                var pollTimer = setInterval(async function() {
                    try {
                        var r = await fetch('/api/indexing_status');
                        var s = await r.json();
                        pollCount++;
                        var done = s.completed || 0;
                        var totalIdx = s.total || 0;
                        var current = s.current_file || '';
                        var errors = s.errors || [];
                        var warnings = s.warnings || [];
                        if (progressEl) {
                            var pt = progressEl.querySelector('#progress-text');
                            if (pt) {
                                if (totalIdx > 0 && done < totalIdx) {
                                    pt.textContent = 'Mengindeks ' + done + '/' + totalIdx + ' (' + current + ')...';
                                } else {
                                    var msg = 'Selesai mengindeks ' + done + ' file';
                                    var extras = [];
                                    if (warnings.length > 0) {
                                        extras.push(warnings[0]);
                                    }
                                    if (errors.length > 0) {
                                        extras.push(errors[errors.length - 1]);
                                    }
                                    if (extras.length > 0) {
                                        msg += '\\n' + extras.join('\\n');
                                    }
                                    pt.textContent = msg;
                                }
                            }
                        }
                        if (totalIdx > 0 && done >= totalIdx) {
                            clearInterval(pollTimer);
                            var btn = document.getElementById('refresh-trigger');
                            if (btn) btn.click();
                        }
                    } catch(e) {
                        if (progressEl) {
                            var pt = progressEl.querySelector('#progress-text');
                            if (pt) pt.textContent = 'Menunggu indeks...';
                        }
                    }
                }, 2000);
    });
    input.click();
}""",
    )

    def show_delete_modal(filename):
        if not filename:
            return "❌ Pilih file terlebih dahulu", gr.update()
        html = f"""<div id="delete-modal-overlay" style="display:flex;position:fixed;top:0;left:0;width:100%;height:100%;background:var(--neutral-800, rgba(0,0,0,0.5));z-index:9999;align-items:center;justify-content:center;">
<div style="background:var(--block-background-fill, white);border-radius:var(--block-radius, 12px);padding:24px;min-width:360px;box-shadow:var(--shadow-drop-lg, 0 4px 20px rgba(0,0,0,0.2));">
<h3 style="margin:0 0 8px 0;font-size:16px;color:var(--block-label-text-color, #d32f2f);">Konfirmasi Hapus</h3>
<p style="margin:0 0 20px 0;font-size:14px;color:var(--body-text-color, #333);">Yakin ingin menghapus <strong>{filename}</strong>?</p>
<span id="pending-delete-file" style="display:none;">{filename}</span>
<div style="display:flex;gap:10px;justify-content:flex-end;">
<button onclick="document.getElementById('delete-modal-overlay').style.display='none'" style="padding:8px 20px;border:var(--button-secondary-border-color, 1px solid #ccc);border-radius:var(--button-secondary-border-radius, 6px);background:var(--button-secondary-background-fill, white);color:var(--button-secondary-text-color, #333);cursor:pointer;font-size:14px;font-family:inherit;">Batal</button>
<button onclick="(async function(){{var o=document.getElementById('delete-modal-overlay');var f=document.getElementById('pending-delete-file').textContent;if(!f)return;o.style.display='none';try{{var b=window.location.protocol+'//'+window.location.hostname+':8000';await fetch(b+'/file/'+encodeURIComponent(f),{{method:'DELETE'}});}}catch(e){{}}var r=document.getElementById('refresh-trigger');if(r)r.click();}})()" style="padding:8px 20px;border:var(--button-primary-border-color, none);border-radius:var(--button-primary-border-radius, 6px);background:var(--button-primary-background-fill, #d32f2f);color:var(--button-primary-text-color, white);cursor:pointer;font-size:14px;font-weight:600;font-family:inherit;">Ya, Hapus</button>
</div>
</div>
</div>"""
        return "", html

    delete_btn.click(
        show_delete_modal,
        [doc_selector],
        [delete_status, delete_modal],
    )


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1, max_size=20).launch(
        server_name="0.0.0.0",
        server_port=7860,
        css=CUSTOM_CSS,
    )
