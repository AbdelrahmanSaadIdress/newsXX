"""
news_ui.py
==========
Gradio front-end for the NewsX API.

Streaming is real — tokens are yielded to Gradio one at a time as they
arrive from the server. The SSE parser reads chunk_size=1 so no byte is
held back in requests' internal 512-byte buffer.

Tabs
----
  ① Ask the News  — conversational RAG chat (fresh ask → follow-up)
  ② Deep Dive     — article digest pipeline with streaming + translation
"""

from __future__ import annotations

import json
from typing import Generator

import gradio as gr
import requests

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

API_BASE = "http://localhost:8000/api/v1"


# ─────────────────────────────────────────────────────────────────────────────
# True-streaming SSE parser
# ─────────────────────────────────────────────────────────────────────────────

def _iter_sse_tokens(response) -> Generator[str, None, None]:
    """
    Parse an SSE stream byte-by-byte so every token is yielded the instant
    it arrives — no 512-byte buffering from requests' default iter_lines().

    Each SSE frame from the server is:
        data: <payload>\\n\\n

    We accumulate raw bytes into a line buffer. On every \\n we flush the
    line: if it starts with "data: " we yield the payload immediately.
    Blank lines (SSE frame separators) are silently dropped.
    """
    buf = b""
    for byte in response.iter_content(chunk_size=1):
        if not byte:
            continue
        if byte == b"\n":
            line = buf.decode("utf-8", errors="replace")
            buf = b""
            if line.startswith("data: "):
                yield line[len("data: "):]
        else:
            buf += byte


# ─────────────────────────────────────────────────────────────────────────────
# API helpers
# ─────────────────────────────────────────────────────────────────────────────

def api_create_session() -> str | None:
    try:
        r = requests.post(f"{API_BASE}/sessions", timeout=10)
        r.raise_for_status()
        return r.json().get("session_id")
    except Exception as exc:
        print(f"[SESSION] create failed: {exc}")
        return None


def api_ask_fresh(question: str) -> Generator[tuple[str, str], None, None]:
    """Yields (answer_so_far, sources_html) — one yield per token."""
    answer = ""
    sources: list[dict] = []
    try:
        with requests.post(
            f"{API_BASE}/ask",
            json={"question": question},
            stream=True,
            timeout=120,
        ) as resp:
            resp.raise_for_status()
            for payload in _iter_sse_tokens(resp):
                if payload.startswith("[DONE]"):
                    body = json.loads(payload[len("[DONE]"):].strip())
                    sources = body.get("source_links", [])
                    break
                answer += payload
                yield answer, _render_sources(sources)
    except Exception as exc:
        yield f"⚠️ Error: {exc}", ""
        return
    yield answer, _render_sources(sources)


def api_ask_followup(question: str, session_id: str) -> Generator[tuple[str, str], None, None]:
    """Yields (answer_so_far, sources_html) — one yield per token."""
    answer = ""
    sources: list[dict] = []
    try:
        with requests.post(
            f"{API_BASE}/ask/followup",
            json={"question": question, "session_id": session_id},
            stream=True,
            timeout=120,
        ) as resp:
            resp.raise_for_status()
            for payload in _iter_sse_tokens(resp):
                if payload.startswith("[DONE]"):
                    body = json.loads(payload[len("[DONE]"):].strip())
                    sources = body.get("source_links", [])
                    break
                answer += payload
                yield answer, _render_sources(sources)
    except Exception as exc:
        yield f"⚠️ Error: {exc}", ""
        return
    yield answer, _render_sources(sources)


def api_digest(url: str) -> Generator[tuple[str, str, str], None, None]:
    """Yields (meta_html, article_so_far, sources_html) — one yield per token."""
    meta_html = ""
    article = ""
    sources_html = ""
    try:
        with requests.post(
            f"{API_BASE}/article/digest",
            json={"url": url},
            stream=True,
            timeout=300,
        ) as resp:
            resp.raise_for_status()
            for payload in _iter_sse_tokens(resp):
                if payload.startswith("[META]"):
                    meta = json.loads(payload[len("[META]"):].strip())
                    meta_html = _render_meta(meta)
                    yield meta_html, article, sources_html
                elif payload.startswith("[DONE]"):
                    body = json.loads(payload[len("[DONE]"):].strip())
                    sources_html = _render_sources(body.get("source_links", []))
                    break
                else:
                    article += payload
                    yield meta_html, article, sources_html
    except Exception as exc:
        yield "", f"⚠️ Error: {exc}", ""
        return
    yield meta_html, article, sources_html


def api_translate(article: str, lang: str) -> tuple[str, str]:
    r = requests.post(
        f"{API_BASE}/article/translate",
        json={"article": article, "lang": lang},
        timeout=120,
    )
    r.raise_for_status()
    data = r.json()
    return data.get("translated_title", ""), data.get("translated_content", "")


# ─────────────────────────────────────────────────────────────────────────────
# HTML renderers
# ─────────────────────────────────────────────────────────────────────────────

def _render_sources(sources: list[dict]) -> str:
    if not sources:
        return ""
    items = "".join(
        f'<li><a href="{s["url"]}" target="_blank">{s.get("title") or s["url"]}</a></li>'
        for s in sources
    )
    return (
        '<div class="sources-box">'
        '<span class="sources-label">📰 Sources</span>'
        f'<ul>{items}</ul>'
        '</div>'
    )


def _render_meta(meta: dict) -> str:
    title    = meta.get("story_title", "")
    category = meta.get("story_category", "").upper()
    keywords = meta.get("story_keywords", [])
    summary  = meta.get("story_summary", [])
    entities = meta.get("story_entities", [])

    kw_chips = "".join(f'<span class="chip">{k}</span>' for k in keywords)
    bullets  = "".join(f"<li>{b}</li>" for b in summary)

    entity_html = ""
    if entities:
        by_type: dict[str, list[str]] = {}
        for e in entities:
            t = e.get("entity_type", "other")
            v = e.get("entity_value", "")
            by_type.setdefault(t, []).append(v)
        rows = ""
        for etype, vals in by_type.items():
            pills = "".join(f'<span class="entity-pill">{v}</span>' for v in vals)
            rows += (
                f'<div class="entity-row">'
                f'<span class="entity-type">{etype}</span>{pills}'
                f'</div>'
            )
        entity_html = (
            '<div class="entity-section">'
            '<div class="meta-label">Entities</div>'
            f'{rows}</div>'
        )

    return (
        '<div class="meta-card">'
        f'<div class="meta-category">{category}</div>'
        f'<div class="meta-title">{title}</div>'
        '<div class="meta-label">Keywords</div>'
        f'<div class="meta-keywords">{kw_chips}</div>'
        '<div class="meta-label">Key Points</div>'
        f'<ul class="meta-bullets">{bullets}</ul>'
        f'{entity_html}'
        '</div>'
    )


def _render_translation(title: str, content: str, lang_label: str) -> str:
    flag = "🇬🇧" if lang_label == "English" else "🇫🇷"
    return (
        '<div class="translation-card">'
        f'<div class="trans-header">{flag} {lang_label} Translation</div>'
        f'<div class="trans-title">{title}</div>'
        f'<div class="trans-content">{content}</div>'
        '</div>'
    )


# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@500;700&family=Source+Serif+4:ital,wght@0,300;0,400;0,600;1,400&family=DM+Sans:wght@400;500&display=swap');

:root {
  --cream:      #FAF7F2;
  --warm-white: #FDF9F4;
  --parchment:  #F2EBE0;
  --rust:       #C1440E;
  --rust-light: #E05C2A;
  --rust-pale:  #FAE9E1;
  --ink:        #1C1410;
  --ink-mid:    #3D2B1F;
  --ink-soft:   #6B5240;
  --gold:       #B8860B;
  --sage:       #5A7A5C;
  --border:     #D9CEBF;
  --shadow:     rgba(28,20,16,0.10);
}

body, .gradio-container {
  background: var(--cream) !important;
  font-family: 'DM Sans', sans-serif !important;
  color: var(--ink) !important;
}
.gradio-container { max-width: 1200px !important; margin: 0 auto !important; }

.masthead {
  text-align: center; padding: 36px 0 20px;
  border-bottom: 3px double var(--rust); margin-bottom: 8px;
}
.masthead-date { font-size: 11px; letter-spacing: 3px; text-transform: uppercase; color: var(--ink-soft); margin-bottom: 6px; }
.masthead-name { font-family: 'Playfair Display', serif; font-size: 52px; font-weight: 700; color: var(--ink); letter-spacing: -1px; line-height: 1; }
.masthead-tagline { font-family: 'Source Serif 4', serif; font-style: italic; font-size: 14px; color: var(--ink-soft); margin-top: 6px; }

.tabs > .tab-nav { border-bottom: 2px solid var(--rust) !important; background: transparent !important; }
.tabs > .tab-nav button {
  font-family: 'DM Sans', sans-serif !important; font-size: 12px !important;
  font-weight: 500 !important; letter-spacing: 2px !important; text-transform: uppercase !important;
  color: var(--ink-soft) !important; padding: 10px 24px !important; border: none !important;
  border-bottom: 3px solid transparent !important; background: transparent !important; transition: all .2s !important;
}
.tabs > .tab-nav button.selected { color: var(--rust) !important; border-bottom-color: var(--rust) !important; }
.tabs > .tab-nav button:hover { color: var(--rust-light) !important; }

.section-rule { display: flex; align-items: center; gap: 12px; margin: 20px 0 12px; }
.section-rule span { font-size: 10px; letter-spacing: 3px; text-transform: uppercase; color: var(--ink-soft); white-space: nowrap; font-family: 'DM Sans', sans-serif; }
.section-rule::before, .section-rule::after { content: ''; flex: 1; height: 1px; background: var(--border); }

.chat-wrap {
  background: var(--warm-white) !important; border: 1px solid var(--border) !important;
  border-radius: 4px !important; box-shadow: 0 2px 12px var(--shadow) !important;
}

textarea, input[type=text] {
  font-family: 'Source Serif 4', serif !important; font-size: 15px !important;
  background: var(--warm-white) !important; border: 1px solid var(--border) !important;
  border-radius: 4px !important; color: var(--ink) !important;
}
textarea:focus, input[type=text]:focus {
  border-color: var(--rust) !important;
  box-shadow: 0 0 0 3px var(--rust-pale) !important; outline: none !important;
}

button.primary, .gr-button-primary {
  font-family: 'DM Sans', sans-serif !important; font-size: 12px !important;
  font-weight: 500 !important; letter-spacing: 2px !important; text-transform: uppercase !important;
  background: var(--rust) !important; color: #fff !important; border: none !important;
  border-radius: 3px !important; padding: 10px 24px !important; transition: background .2s !important;
}
button.primary:hover { background: var(--rust-light) !important; }

button.secondary, .gr-button-secondary {
  font-family: 'DM Sans', sans-serif !important; font-size: 11px !important;
  font-weight: 500 !important; letter-spacing: 2px !important; text-transform: uppercase !important;
  background: transparent !important; color: var(--ink-mid) !important;
  border: 1px solid var(--border) !important; border-radius: 3px !important;
  padding: 9px 20px !important; transition: all .2s !important;
}
button.secondary:hover { border-color: var(--rust) !important; color: var(--rust) !important; }

.btn-en { border-color: var(--sage) !important; color: var(--sage) !important; }
.btn-en:hover { background: var(--sage) !important; color: #fff !important; }
.btn-fr { border-color: var(--gold) !important; color: var(--gold) !important; }
.btn-fr:hover { background: var(--gold) !important; color: #fff !important; }

.status-badge {
  display: inline-flex; align-items: center; gap: 6px;
  font-family: 'DM Sans', sans-serif; font-size: 11px; font-weight: 500;
  letter-spacing: 1px; text-transform: uppercase; padding: 4px 12px;
  border-radius: 20px; background: var(--parchment); color: var(--ink-soft); border: 1px solid var(--border);
}
.status-badge.active { background: var(--rust-pale); color: var(--rust); border-color: var(--rust); }
.status-badge.session { background: #EDF5EE; color: var(--sage); border-color: var(--sage); }

.sources-box { margin-top: 16px; padding: 14px 18px; background: var(--parchment); border-left: 3px solid var(--gold); border-radius: 0 4px 4px 0; }
.sources-label { font-family: 'DM Sans', sans-serif; font-size: 10px; letter-spacing: 2px; text-transform: uppercase; color: var(--gold); display: block; margin-bottom: 8px; font-weight: 600; }
.sources-box ul { margin: 0; padding-left: 18px; }
.sources-box li { margin: 4px 0; }
.sources-box a { font-family: 'Source Serif 4', serif; font-size: 13px; color: var(--rust); text-decoration: none; }
.sources-box a:hover { text-decoration: underline; }

.meta-card { background: var(--warm-white); border: 1px solid var(--border); border-top: 3px solid var(--rust); border-radius: 0 0 4px 4px; padding: 20px; font-family: 'DM Sans', sans-serif; }
.meta-category { font-size: 9px; letter-spacing: 3px; color: var(--rust); font-weight: 600; margin-bottom: 8px; }
.meta-title { font-family: 'Playfair Display', serif; font-size: 17px; font-weight: 700; color: var(--ink); line-height: 1.4; margin-bottom: 14px; padding-bottom: 14px; border-bottom: 1px solid var(--border); }
.meta-label { font-size: 9px; letter-spacing: 3px; text-transform: uppercase; color: var(--ink-soft); font-weight: 600; margin: 12px 0 7px; }
.meta-keywords { display: flex; flex-wrap: wrap; gap: 5px; margin-bottom: 4px; }
.chip { font-size: 11px; padding: 3px 10px; background: var(--rust-pale); color: var(--rust); border-radius: 20px; border: 1px solid #f0c4b4; white-space: nowrap; }
.meta-bullets { margin: 0; padding-left: 18px; font-family: 'Source Serif 4', serif; font-size: 13px; line-height: 1.7; color: var(--ink-mid); }
.meta-bullets li { margin-bottom: 5px; }
.entity-section { margin-top: 14px; padding-top: 14px; border-top: 1px solid var(--border); }
.entity-row { display: flex; flex-wrap: wrap; align-items: center; gap: 5px; margin-bottom: 7px; }
.entity-type { font-size: 9px; letter-spacing: 1.5px; text-transform: uppercase; color: var(--ink-soft); min-width: 80px; font-weight: 600; }
.entity-pill { font-size: 11px; padding: 2px 9px; background: var(--parchment); color: var(--ink-mid); border-radius: 3px; border: 1px solid var(--border); }

.article-output textarea {
  font-family: 'Source Serif 4', serif !important; font-size: 16px !important;
  line-height: 1.85 !important; color: var(--ink) !important;
  background: var(--warm-white) !important; padding: 28px 32px !important;
}

.translation-card { background: var(--warm-white); border: 1px solid var(--border); border-radius: 4px; overflow: hidden; margin-top: 4px; }
.trans-header { font-family: 'DM Sans', sans-serif; font-size: 10px; font-weight: 600; letter-spacing: 3px; text-transform: uppercase; padding: 10px 20px; background: var(--parchment); color: var(--ink-soft); border-bottom: 1px solid var(--border); }
.trans-title { font-family: 'Playfair Display', serif; font-size: 20px; font-weight: 700; padding: 20px 24px 10px; color: var(--ink); border-bottom: 1px solid var(--border); }
.trans-content { font-family: 'Source Serif 4', serif; font-size: 15px; line-height: 1.85; padding: 20px 24px; color: var(--ink-mid); }

footer { display: none !important; }
"""


# ─────────────────────────────────────────────────────────────────────────────
# Build UI
# ─────────────────────────────────────────────────────────────────────────────

def build_app() -> gr.Blocks:
    with gr.Blocks(title="NewsX") as app:

        session_id_state     = gr.State(value=None)
        turn_count_state     = gr.State(value=0)
        digest_article_state = gr.State(value="")

        gr.HTML("""
        <div class="masthead">
          <div class="masthead-date" id="today-date"></div>
          <div class="masthead-name">NewsX</div>
          <div class="masthead-tagline">Intelligence from the news — delivered in prose</div>
        </div>
        <script>
          const d = new Date();
          document.getElementById('today-date').textContent =
            d.toLocaleDateString('en-US',{weekday:'long',year:'numeric',month:'long',day:'numeric'}).toUpperCase();
        </script>
        """)

        with gr.Tabs():

            # ── TAB 1: Ask the News ──────────────────────────────────────────
            with gr.TabItem("✦  Ask the News"):

                with gr.Row():
                    session_badge = gr.HTML(
                        '<span class="status-badge">⏳ Starting session…</span>'
                    )

                gr.HTML('<div class="section-rule"><span>Conversation</span></div>')

                chatbot = gr.Chatbot(
                    height=460,
                    show_label=False,
                    elem_classes=["chat-wrap"],
                    avatar_images=(None, "https://i.imgur.com/7yUvePI.png"),
                    render_markdown=True,
                    layout="bubble",
                    placeholder=(
                        "<div style='text-align:center;padding:40px;"
                        "font-family:Source Serif 4,serif;color:#6B5240;"
                        "font-style:italic;'>Ask anything about today's news…</div>"
                    ),
                )

                sources_html_chat = gr.HTML("")

                with gr.Row():
                    question_box = gr.Textbox(
                        placeholder="What's happening in the world today?",
                        label="Your question",
                        lines=2,
                        scale=5,
                    )
                    with gr.Column(scale=1, min_width=120):
                        send_btn  = gr.Button("Send",  variant="primary")
                        clear_btn = gr.Button("Clear", variant="secondary")

                def _bootstrap():
                    sid = api_create_session()
                    if sid:
                        badge = f'<span class="status-badge session">🔑 Session active · {sid[:8]}…</span>'
                        return sid, 0, badge
                    return None, 0, '<span class="status-badge">⚠️ Session failed — check API</span>'

                app.load(fn=_bootstrap, outputs=[session_id_state, turn_count_state, session_badge])

                def _send(question, history, session_id, turn_count):
                    if not question.strip():
                        yield history, "", session_id, turn_count
                        return

                    history = list(history or [])
                    history.append({"role": "user",      "content": question})
                    history.append({"role": "assistant", "content": "▌"})
                    yield history, "", session_id, turn_count

                    answer      = ""
                    sources_out = ""

                    stream_fn = (
                        api_ask_fresh(question)
                        if (turn_count == 0 or session_id is None)
                        else api_ask_followup(question, session_id)
                    )

                    for token_answer, srcs in stream_fn:
                        answer      = token_answer
                        sources_out = srcs
                        history[-1]["content"] = answer + " ▌"
                        yield history, sources_out, session_id, turn_count

                    history[-1]["content"] = answer
                    yield history, sources_out, session_id, turn_count + 1

                send_btn.click(
                    fn=_send,
                    inputs=[question_box, chatbot, session_id_state, turn_count_state],
                    outputs=[chatbot, sources_html_chat, session_id_state, turn_count_state],
                ).then(fn=lambda: gr.update(value=""), outputs=question_box)

                question_box.submit(
                    fn=_send,
                    inputs=[question_box, chatbot, session_id_state, turn_count_state],
                    outputs=[chatbot, sources_html_chat, session_id_state, turn_count_state],
                ).then(fn=lambda: gr.update(value=""), outputs=question_box)

                def _clear():
                    sid = api_create_session()
                    badge = (
                        f'<span class="status-badge session">🔑 New session · {sid[:8]}…</span>'
                        if sid else '<span class="status-badge">⚠️ Session failed</span>'
                    )
                    return [], "", sid, 0, badge

                clear_btn.click(
                    fn=_clear,
                    outputs=[chatbot, sources_html_chat, session_id_state, turn_count_state, session_badge],
                )

            # ── TAB 2: Deep Dive ────────────────────────────────────────────
            with gr.TabItem("✦  Deep Dive"):

                gr.HTML('<div class="section-rule"><span>Article URL</span></div>')

                with gr.Row():
                    url_box    = gr.Textbox(
                        placeholder="https://www.ajnet.me/news/…",
                        label="Article URL",
                        scale=5,
                    )
                    digest_btn = gr.Button("✦ Analyse", variant="primary", scale=1, min_width=130)

                digest_status = gr.HTML("")

                gr.HTML('<div class="section-rule"><span>Analysis</span></div>')

                with gr.Row(equal_height=False):

                    with gr.Column(scale=1, min_width=280):
                        meta_panel = gr.HTML(
                            '<div style="padding:20px;color:var(--ink-soft);'
                            'font-family:DM Sans,sans-serif;font-size:12px;'
                            'letter-spacing:1px;text-align:center;">'
                            'Story details will appear here</div>'
                        )

                    with gr.Column(scale=2):
                        article_box = gr.Textbox(
                            lines=22,
                            interactive=False,
                            elem_classes=["article-output"],
                            placeholder="Your deep-dive article will stream here…",
                            show_label=False,
                        )
                        sources_html_digest = gr.HTML("")

                gr.HTML('<div class="section-rule"><span>Translation</span></div>')

                with gr.Row():
                    btn_en = gr.Button("🇬🇧  Translate to English", variant="secondary", elem_classes=["btn-en"])
                    btn_fr = gr.Button("🇫🇷  Traduire en Français",  variant="secondary", elem_classes=["btn-fr"])

                translation_output = gr.HTML("")

                def _digest(url):
                    if not url.strip():
                        yield (
                            '<span class="status-badge">⚠️ Please enter a URL</span>',
                            '<div style="padding:20px;color:var(--ink-soft);font-family:DM Sans,sans-serif;font-size:12px;">—</div>',
                            "", "", "",
                        )
                        return

                    yield (
                        '<span class="status-badge active">⏳ Analysing article…</span>',
                        '<div style="padding:20px;color:var(--ink-soft);font-family:DM Sans,sans-serif;font-size:12px;">Fetching…</div>',
                        "", "", "",
                    )

                    collected = ""
                    meta_h    = ""
                    src_h     = ""

                    for meta_h, art, src_h in api_digest(url):
                        collected = art
                        yield (
                            '<span class="status-badge active">✦ Generating…</span>',
                            meta_h or '<div style="padding:20px;color:var(--ink-soft);font-family:DM Sans,sans-serif;font-size:12px;">Loading metadata…</div>',
                            collected,
                            src_h,
                            collected,
                        )

                    yield (
                        '<span class="status-badge session">✓ Complete</span>',
                        meta_h, collected, src_h, collected,
                    )

                digest_btn.click(
                    fn=_digest,
                    inputs=[url_box],
                    outputs=[digest_status, meta_panel, article_box, sources_html_digest, digest_article_state],
                ).then(fn=lambda: "", outputs=translation_output)

                def _translate(article: str, lang: str) -> str:
                    if not article.strip():
                        return '<div style="padding:16px;color:var(--ink-soft);font-size:13px;">⚠️ No article to translate yet. Run a Deep Dive first.</div>'
                    try:
                        title, content = api_translate(article, lang)
                        label = "English" if lang == "en" else "French"
                        return _render_translation(title, content, label)
                    except Exception as exc:
                        return f'<div style="padding:16px;color:var(--rust);font-size:13px;">⚠️ Translation error: {exc}</div>'

                btn_en.click(fn=lambda art: _translate(art, "en"), inputs=[digest_article_state], outputs=[translation_output])
                btn_fr.click(fn=lambda art: _translate(art, "fr"), inputs=[digest_article_state], outputs=[translation_output])

    return app


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    demo = build_app()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
        share=False,
        css=CSS,
    )