"""
Ollama UI — lightweight OpenWebUI alternative built with Streamlit.
Run: streamlit run app.py
"""

import streamlit as st
import ollama
import json
import uuid
import os
import base64
from datetime import datetime
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
CHATS_DIR = Path.home() / ".ollama_ui" / "chats"
CHATS_DIR.mkdir(parents=True, exist_ok=True)

st.set_page_config(
    page_title="Ollama UI",
    page_icon="🦙",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ---- General ---- */
[data-testid="stSidebar"] { background: #0d0d0d; }
[data-testid="stSidebar"] .stButton > button { text-align: left; }
.stChatMessage [data-testid="stMarkdownContainer"] pre {
    background: #1a1a2e;
    border: 1px solid #2d2d4e;
    border-radius: 6px;
}
/* ---- Sidebar chat list buttons ---- */
div[data-testid="stSidebarContent"] .chat-item-active button {
    background: #1e3a5f !important;
    border-left: 3px solid #4fa3e0 !important;
}
/* ---- Token counter ---- */
.token-badge {
    font-size: 11px;
    color: #888;
    font-family: monospace;
}
/* ---- Message action row ---- */
.msg-actions {
    display: flex;
    gap: 6px;
    margin-top: 4px;
    opacity: 0;
    transition: opacity 0.2s;
}
/* ---- Scrollable sidebar chat list ---- */
.sidebar-chat-list {
    max-height: 55vh;
    overflow-y: auto;
}
/* ---- Status pill ---- */
.status-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    display: inline-block;
    margin-right: 5px;
}
.status-ok  { background: #4caf50; }
.status-err { background: #f44336; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Persistence helpers
# ─────────────────────────────────────────────────────────────────────────────
def load_all_chats() -> dict:
    chats = {}
    for f in CHATS_DIR.glob("*.json"):
        try:
            data = json.loads(f.read_text())
            chats[data["id"]] = data
        except Exception:
            pass
    return chats

def save_chat(chat: dict):
    path = CHATS_DIR / f"{chat['id']}.json"
    path.write_text(json.dumps(chat, indent=2, ensure_ascii=False))

def delete_chat_file(chat_id: str):
    path = CHATS_DIR / f"{chat_id}.json"
    if path.exists():
        path.unlink()

# ─────────────────────────────────────────────────────────────────────────────
# Ollama helpers
# ─────────────────────────────────────────────────────────────────────────────
def fetch_models() -> list[str]:
    try:
        result = ollama.list()
        return [m["model"] for m in result.get("models", [])]
    except Exception:
        return []

def ollama_online() -> bool:
    try:
        ollama.list()
        return True
    except Exception:
        return False

def make_chat(model: str) -> dict:
    return {
        "id": str(uuid.uuid4()),
        "title": "New Chat",
        "model": model,
        "system_prompt": "",
        "messages": [],          # list of {role, content}
        "params": {
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 40,
            "num_ctx": 4096,
            "repeat_penalty": 1.1,
        },
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
    }

def auto_title(text: str) -> str:
    title = text.strip().split("\n")[0]
    return (title[:50] + "…") if len(title) > 50 else title

def export_markdown(chat: dict) -> str:
    lines = [f"# {chat['title']}", f"> Model: `{chat['model']}`", ""]
    if chat.get("system_prompt"):
        lines += [f"**System:** {chat['system_prompt']}", ""]
    for msg in chat["messages"]:
        role = "**You**" if msg["role"] == "user" else f"**{chat['model']}**"
        lines += [f"{role}:", msg["content"], ""]
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Attachment helpers
# ─────────────────────────────────────────────────────────────────────────────
IMAGE_EXTS = {"png", "jpg", "jpeg", "webp", "gif"}
TEXT_EXTS  = {"txt", "md", "py", "js", "ts", "c", "cpp", "h", "java",
              "sh", "zsh", "bash", "json", "yaml", "yml", "toml", "csv",
              "html", "css", "xml", "rst", "log"}

def extract_text_from_attachment(att: dict) -> str:
    """Return plain text to inject into message context for non-image files."""
    raw = base64.b64decode(att["b64"])
    ext = att["name"].rsplit(".", 1)[-1].lower()
    name = att["name"]

    if ext == "pdf":
        import io
        text = None
        errors = []

        # 1. pymupdf — most robust, handles widest range of PDF encodings
        try:
            import fitz  # pymupdf
            doc = fitz.open(stream=raw, filetype="pdf")
            pages = [page.get_text() for page in doc]
            doc.close()
            text = "\n\n".join(p for p in pages if p.strip())
        except ImportError:
            errors.append("pymupdf not installed")
        except Exception as e:
            errors.append(f"pymupdf: {e}")

        # 2. pdfplumber — good for tables / complex layouts
        if not text:
            try:
                import pdfplumber
                with pdfplumber.open(io.BytesIO(raw)) as pdf:
                    pages = [p.extract_text() or "" for p in pdf.pages]
                text = "\n\n".join(p for p in pages if p.strip())
            except ImportError:
                errors.append("pdfplumber not installed")
            except Exception as e:
                errors.append(f"pdfplumber: {e}")

        # 3. pypdf — lightweight fallback
        if not text:
            try:
                import pypdf
                reader = pypdf.PdfReader(io.BytesIO(raw))
                pages = [p.extract_text() or "" for p in reader.pages]
                text = "\n\n".join(p for p in pages if p.strip())
            except ImportError:
                errors.append("pypdf not installed")
            except Exception as e:
                errors.append(f"pypdf: {e}")

        if text and text.strip():
            return (
                f"[PDF attached: {name}]\n"
                f"The following is the full extracted text from this PDF. "
                f"Answer the user based on this content.\n\n"
                f"{text.strip()}"
            )
        else:
            err_detail = "; ".join(errors) if errors else "unknown"
            return (
                f"[PDF attached: {name}]\n"
                f"PDF text extraction FAILED ({err_detail}). "
                f"This is likely a scanned/image-based PDF with no text layer. "
                f"Do NOT guess content from the filename. "
                f"Tell the user the PDF could not be read and suggest they copy-paste the text."
            )

    if ext in TEXT_EXTS:
        try:
            return (
                f"[File attached: {name}]\n"
                f"Full file content below:\n\n"
                f"{raw.decode('utf-8', errors='replace')}"
            )
        except Exception as e:
            return f"[File: {name}]\n(read error: {e})"

    return f"[Attachment: {name}] (unsupported type for text extraction)"


# ─────────────────────────────────────────────────────────────────────────────
# Session state bootstrap
# ─────────────────────────────────────────────────────────────────────────────
if "chats" not in st.session_state:
    st.session_state.chats = load_all_chats()
if "current_id" not in st.session_state:
    ids = list(st.session_state.chats)
    st.session_state.current_id = ids[0] if ids else None
if "models" not in st.session_state:
    st.session_state.models = fetch_models()
if "generating" not in st.session_state:
    st.session_state.generating = False
if "edit_idx" not in st.session_state:
    st.session_state.edit_idx = None
if "pull_progress" not in st.session_state:
    st.session_state.pull_progress = ""
if "pending_attachments" not in st.session_state:
    # list of {"name": str, "mime": str, "b64": str}
    st.session_state.pending_attachments = []

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    # Header + status
    online = ollama_online()
    status_color = "🟢" if online else "🔴"
    st.markdown(f"## 🦙 Ollama UI &nbsp;{status_color}", unsafe_allow_html=True)

    if not online:
        st.error("Ollama not reachable. Is it running on localhost:11434?")

    # Model selector
    models = st.session_state.models
    default_model = models[0] if models else ""
    selected_model = st.selectbox(
        "Model",
        options=models if models else ["(no models)"],
        key="model_selector",
    )

    # New chat
    if st.button("＋  New Chat", use_container_width=True, type="primary"):
        if models:
            chat = make_chat(selected_model)
            st.session_state.chats[chat["id"]] = chat
            st.session_state.current_id = chat["id"]
            st.session_state.edit_idx = None
            save_chat(chat)
            st.rerun()

    st.divider()

    # Search
    search_q = st.text_input("🔍 Search", placeholder="Search chats…", label_visibility="collapsed")

    # Chat list
    sorted_chats = sorted(
        st.session_state.chats.items(),
        key=lambda x: x[1].get("updated_at", ""),
        reverse=True,
    )

    for cid, chat in sorted_chats:
        title = chat.get("title", "Untitled")
        if search_q and search_q.lower() not in title.lower():
            skip = True
            # also search message content
            for m in chat.get("messages", []):
                if search_q.lower() in m.get("content", "").lower():
                    skip = False
                    break
            if skip:
                continue

        col_a, col_b = st.columns([5, 1])
        is_active = cid == st.session_state.current_id
        btn_type = "primary" if is_active else "secondary"

        with col_a:
            icon = "💬 " if is_active else "   "
            label = f"{icon}{title[:28]}{'…' if len(title) > 28 else ''}"
            if st.button(label, key=f"sel_{cid}", use_container_width=True, type=btn_type):
                st.session_state.current_id = cid
                st.session_state.edit_idx = None
                st.rerun()
        with col_b:
            if st.button("✕", key=f"del_{cid}", help="Delete chat"):
                delete_chat_file(cid)
                del st.session_state.chats[cid]
                if st.session_state.current_id == cid:
                    remaining = [k for k in st.session_state.chats if k != cid]
                    st.session_state.current_id = remaining[0] if remaining else None
                st.rerun()

    st.divider()

    # ── Parameters ──
    with st.expander("⚙️  Parameters", expanded=False):
        if st.session_state.current_id:
            chat = st.session_state.chats[st.session_state.current_id]
            p = chat["params"]
            p["temperature"]    = st.slider("Temperature",    0.0, 2.0, p.get("temperature", 0.7),    0.05)
            p["top_p"]          = st.slider("Top P",          0.0, 1.0, p.get("top_p", 0.9),          0.05)
            p["top_k"]          = st.slider("Top K",          1,   200, p.get("top_k", 40))
            p["repeat_penalty"] = st.slider("Repeat Penalty", 0.5, 2.0, p.get("repeat_penalty", 1.1), 0.05)
            p["num_ctx"]        = st.select_slider(
                "Context Length",
                options=[2048, 4096, 8192, 16384, 32768, 65536],
                value=p.get("num_ctx", 4096),
            )
            save_chat(chat)
        else:
            st.caption("Open a chat to set params.")

    # ── System Prompt ──
    with st.expander("🔧  System Prompt", expanded=False):
        if st.session_state.current_id:
            chat = st.session_state.chats[st.session_state.current_id]
            new_sp = st.text_area(
                "System prompt",
                value=chat.get("system_prompt", ""),
                height=140,
                label_visibility="collapsed",
                placeholder="You are a helpful assistant…",
            )
            if new_sp != chat.get("system_prompt", ""):
                chat["system_prompt"] = new_sp
                save_chat(chat)
        else:
            st.caption("Open a chat first.")

    # ── Model Pull ──
    with st.expander("📦  Pull Model", expanded=False):
        pull_name = st.text_input("Model name", placeholder="e.g. llama3:8b", key="pull_name_input")
        if st.button("Pull", use_container_width=True):
            if pull_name.strip():
                with st.spinner(f"Pulling {pull_name}…"):
                    try:
                        for chunk in ollama.pull(pull_name.strip(), stream=True):
                            status = chunk.get("status", "")
                            if "total" in chunk and "completed" in chunk:
                                pct = int(chunk["completed"] / chunk["total"] * 100)
                                st.session_state.pull_progress = f"{status} {pct}%"
                        st.session_state.models = fetch_models()
                        st.success(f"✅ Pulled {pull_name}")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Pull failed: {e}")

    # ── Debug ──
    with st.expander("🛠  Debug", expanded=False):
        st.session_state.show_debug = st.toggle("Show API payload", value=st.session_state.get("show_debug", False))

    # ── Model Info ──
    with st.expander("ℹ️  Model Info", expanded=False):
        if models and st.session_state.current_id:
            chat = st.session_state.chats[st.session_state.current_id]
            try:
                info = ollama.show(chat["model"])
                details = info.get("details", {})
                st.markdown(f"""
**Family:** `{details.get('family', 'N/A')}`  
**Params:** `{details.get('parameter_size', 'N/A')}`  
**Quant:** `{details.get('quantization_level', 'N/A')}`  
**Format:** `{details.get('format', 'N/A')}`
""")
            except Exception as e:
                st.caption(f"Could not load info: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# Main area
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state.current_id is None:
    st.markdown("""
    <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;height:70vh;gap:12px;opacity:0.6;">
        <div style="font-size:80px">🦙</div>
        <div style="font-size:28px;font-weight:700;">Ollama UI</div>
        <div style="font-size:16px;color:#888;">Create a new chat or select one from the sidebar.</div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── Active chat ──
chat = st.session_state.chats[st.session_state.current_id]

# Header row
hcol1, hcol2, hcol3, hcol4, hcol5 = st.columns([5, 1, 1, 1, 1])

with hcol1:
    new_title = st.text_input(
        "title", value=chat["title"], label_visibility="collapsed", key="chat_title_input"
    )
    if new_title != chat["title"]:
        chat["title"] = new_title
        save_chat(chat)

with hcol2:
    st.download_button(
        "⬇ JSON",
        data=json.dumps(chat, indent=2, ensure_ascii=False),
        file_name=f"{chat['title']}.json",
        mime="application/json",
        use_container_width=True,
    )

with hcol3:
    st.download_button(
        "⬇ MD",
        data=export_markdown(chat),
        file_name=f"{chat['title']}.md",
        mime="text/markdown",
        use_container_width=True,
    )

with hcol4:
    if st.button("🔄 Regen", use_container_width=True, help="Regenerate last response"):
        msgs = chat["messages"]
        if msgs and msgs[-1]["role"] == "assistant":
            chat["messages"] = msgs[:-1]
            save_chat(chat)
        st.rerun()

with hcol5:
    if st.button("🗑 Clear", use_container_width=True, help="Clear all messages"):
        chat["messages"] = []
        save_chat(chat)
        st.session_state.edit_idx = None
        st.rerun()

# Meta info
n_msgs = len(chat["messages"])
n_chars = sum(len(m["content"]) for m in chat["messages"])
st.caption(
    f"Model: `{chat['model']}` &nbsp;·&nbsp; "
    f"Messages: `{n_msgs}` &nbsp;·&nbsp; "
    f"Chars: `{n_chars:,}` &nbsp;·&nbsp; "
    f"Ctx: `{chat['params']['num_ctx']:,}` tokens"
)

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# Message display
# ─────────────────────────────────────────────────────────────────────────────
for i, msg in enumerate(chat["messages"]):
    with st.chat_message(msg["role"]):
        # Edit mode for a user message
        if st.session_state.edit_idx == i and msg["role"] == "user":
            edited = st.text_area(
                "Edit message", value=msg["content"], key=f"edit_area_{i}", height=100
            )
            ec1, ec2 = st.columns([1, 1])
            with ec1:
                if st.button("✅ Save & Resend", key=f"save_edit_{i}"):
                    chat["messages"] = chat["messages"][:i]
                    chat["messages"].append({"role": "user", "content": edited})
                    st.session_state.edit_idx = None
                    save_chat(chat)
                    st.rerun()
            with ec2:
                if st.button("✕ Cancel", key=f"cancel_edit_{i}"):
                    st.session_state.edit_idx = None
                    st.rerun()
        else:
            # Render attachments stored with the message
            for att in msg.get("attachments", []):
                kind = att.get("kind", "image" if att["mime"].startswith("image/") else "text")
                if kind == "image":
                    img_bytes = base64.b64decode(att["b64"])
                    st.image(img_bytes, caption=att["name"], use_container_width=False, width=420)
                elif kind == "pdf":
                    with st.expander(f"📕 {att['name']}"):
                        extracted = extract_text_from_attachment(att)
                        prefix = f"[PDF: {att['name']}]\n"
                        st.text(extracted[len(prefix):] if extracted.startswith(prefix) else extracted)
                else:
                    with st.expander(f"📄 {att['name']}"):
                        extracted = extract_text_from_attachment(att)
                        prefix = f"[File: {att['name']}]\n"
                        st.code(extracted[len(prefix):] if extracted.startswith(prefix) else extracted)
            st.markdown(msg["content"])
            # Action buttons (subtle)
            ac1, ac2, ac3, *_ = st.columns([1, 1, 1, 10])
            with ac1:
                if msg["role"] == "user":
                    if st.button("✏️", key=f"edit_btn_{i}", help="Edit & resend"):
                        st.session_state.edit_idx = i
                        st.rerun()
            with ac2:
                st.button("📋", key=f"copy_{i}", help="(copy text above manually — clipboard API not available in Streamlit)")
            with ac3:
                if st.button("🗑", key=f"delmsg_{i}", help="Delete this message"):
                    chat["messages"].pop(i)
                    save_chat(chat)
                    st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# Attachment uploader
# ─────────────────────────────────────────────────────────────────────────────
ACCEPT_TYPES = list(IMAGE_EXTS) + ["pdf"] + list(TEXT_EXTS)

with st.container():
    uploaded_files = st.file_uploader(
        "📎 Attach files",
        type=ACCEPT_TYPES,
        accept_multiple_files=True,
        label_visibility="collapsed",
        key="file_uploader",
    )

    # Sync uploader into pending_attachments
    if uploaded_files:
        existing_names = {a["name"] for a in st.session_state.pending_attachments}
        for f in uploaded_files:
            if f.name not in existing_names:
                raw = f.read()
                ext = f.name.rsplit(".", 1)[-1].lower()
                kind = "image" if ext in IMAGE_EXTS else ("pdf" if ext == "pdf" else "text")
                st.session_state.pending_attachments.append({
                    "name": f.name,
                    "mime": f.type or f"application/{ext}",
                    "kind": kind,          # "image" | "pdf" | "text"
                    "b64":  base64.b64encode(raw).decode(),
                })

    # Preview pending attachments
    if st.session_state.pending_attachments:
        cols = st.columns(min(len(st.session_state.pending_attachments), 6))
        for idx, att in enumerate(st.session_state.pending_attachments):
            with cols[idx % 6]:
                if att["kind"] == "image":
                    st.image(base64.b64decode(att["b64"]), caption=att["name"], width=100)
                elif att["kind"] == "pdf":
                    st.markdown(f"📕 `{att['name']}`")
                else:
                    st.markdown(f"📄 `{att['name']}`")
                if st.button("✕", key=f"rm_att_{idx}", help="Remove"):
                    st.session_state.pending_attachments.pop(idx)
                    st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# Chat input & streaming
# ─────────────────────────────────────────────────────────────────────────────
has_attachments = bool(st.session_state.pending_attachments)
n_att = len(st.session_state.pending_attachments)
placeholder_text = f"Message… ({n_att} file(s) attached 📎)" if has_attachments else "Message…"
user_input = st.chat_input(placeholder_text, disabled=st.session_state.generating)

if user_input and not st.session_state.generating:
    attachments = list(st.session_state.pending_attachments)  # snapshot
    st.session_state.pending_attachments = []                  # clear queue

    # Build stored message (includes attachments for history rendering)
    user_msg = {
        "role": "user",
        "content": user_input.strip(),
        "attachments": attachments,
    }
    chat["messages"].append(user_msg)

    # Auto-title on first message
    if chat["title"] == "New Chat" and len(chat["messages"]) == 1:
        chat["title"] = auto_title(user_input)

    chat["updated_at"] = datetime.now().isoformat()
    save_chat(chat)

    # Build API message list
    # For multimodal messages, pass images as bytes to Ollama
    api_messages = []
    if chat.get("system_prompt", "").strip():
        api_messages.append({"role": "system", "content": chat["system_prompt"]})

    for m in chat["messages"]:
        content_parts = [m["content"]]
        img_b64_list = []
        for att in m.get("attachments", []):
            kind = att.get("kind", "image" if att["mime"].startswith("image/") else "text")
            if kind == "image":
                # Pass base64 string directly — ollama lib handles encoding internally
                img_b64_list.append(att["b64"])
            else:
                # PDF and text files: inject extracted text as context
                content_parts.insert(0, extract_text_from_attachment(att))
        api_msg = {"role": m["role"], "content": "\n\n".join(content_parts)}
        if img_b64_list:
            api_msg["images"] = img_b64_list
        api_messages.append(api_msg)

    # Display user bubble immediately
    with st.chat_message("user"):
        for att in attachments:
            kind = att.get("kind", "image" if att["mime"].startswith("image/") else "text")
            if kind == "image":
                st.image(base64.b64decode(att["b64"]), caption=att["name"], width=420)
            elif kind == "pdf":
                st.caption(f"📕 {att['name']}")
            else:
                st.caption(f"📄 {att['name']}")
        st.markdown(user_input)

    # Debug: show what's being sent (toggle in sidebar via session state)
    if st.session_state.get("show_debug"):
        with st.expander("🛠 API payload (debug)", expanded=False):
            debug_view = []
            for m in api_messages:
                entry = {"role": m["role"], "content": m["content"][:200] + "..." if len(m["content"]) > 200 else m["content"]}
                if "images" in m:
                    entry["images"] = f"[{len(m['images'])} image(s), b64 len={[len(i) for i in m['images']]}]"
                debug_view.append(entry)
            st.json(debug_view)

    # Stream assistant response
    with st.chat_message("assistant"):
        resp_placeholder = st.empty()
        full_response = ""
        st.session_state.generating = True

        try:
            stream = ollama.chat(
                model=chat["model"],
                messages=api_messages,
                stream=True,
                options={
                    "temperature":    chat["params"]["temperature"],
                    "top_p":          chat["params"]["top_p"],
                    "top_k":          chat["params"]["top_k"],
                    "num_ctx":        chat["params"]["num_ctx"],
                    "repeat_penalty": chat["params"].get("repeat_penalty", 1.1),
                },
            )
            for chunk in stream:
                token = chunk.get("message", {}).get("content", "")
                full_response += token
                resp_placeholder.markdown(full_response + "▌")
            resp_placeholder.markdown(full_response)

        except Exception as e:
            error_msg = f"⚠️ **Error:** `{e}`"
            resp_placeholder.error(error_msg)
            full_response = error_msg

    st.session_state.generating = False
    chat["messages"].append({"role": "assistant", "content": full_response})
    chat["updated_at"] = datetime.now().isoformat()
    save_chat(chat)
    st.rerun()
