# NewsX — AI-Powered News Intelligence Platform

> **An end-to-end NLP pipeline** that scrapes, analyses, stores, and synthesises Arabic news — powered by a fine-tuned LLM, retrieval-augmented generation, and a streaming FastAPI backend.

---

## 📺 Demo

[![Watch the demo](/docs/thumbnail.png)](https://canva.link/iz0yyxvvyp109nz)

---

## 📬 Telegram Daily Digest

![Telegram Bot Screenshot](/docs/telegram_bot.png)

> Every day at a scheduled time, NewsX automatically scrapes the latest articles, analyses each one through the fine-tuned model, and publishes a concise Arabic digest directly to a Telegram channel.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Key Features](#key-features)
- [Fine-tuning Pipeline](#fine-tuning-pipeline)
- [RAG Pipeline](#rag-pipeline)
- [Article Deep-Dive Pipeline](#article-deep-dive-pipeline)
- [API Reference](#api-reference)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Configuration](#configuration)
---

## Overview

NewsX is a production-grade news intelligence platform built around three core ideas:

1. **Automated daily intelligence** — A scheduled scraper harvests articles every day, runs each one through a fine-tuned language model for structured analysis (title, keywords, summary bullets, category, named entities), persists results to MongoDB, and publishes a curated digest to Telegram.

2. **Conversational RAG interface** — Users can ask any natural-language question and receive grounded answers synthesised from articles stored in ChromaDB. Multi-turn sessions are supported, with rolling conversation summaries persisted in MongoDB to maintain context across turns.

3. **On-demand article deep-dive** — Given any article URL, the system scrapes and analyses the page on the fly, retrieves semantically related chunks from other articles, and streams a freshly written, context-enriched piece back to the user. The generated article can then be translated into English or French.

---

## Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                         FastAPI Backend                        │
│                                                                │
│  ┌──────────────┐  ┌─────────────────┐  ┌───────────────────┐  │
│  │  /scrape     │  │  /ask  (RAG)    │  │  /article         │  │
│  │  endpoints   │  │  /ask/followup  │  │  /digest          │  │
│  └──────┬───────┘  └────────┬────────┘  │  /translate       │  │
│         │                   │           └────────┬──────────┘  │
└─────────┼───────────────────┼────────────────────┼─────────────┘
          │                   │                    │
          ▼                   ▼                    ▼
  ┌───────────────┐   ┌──────────────┐    ┌──────────────────┐
  │  Selenium /   │   │  LangGraph   │    │   A_TDeps        │
  │  PyQuery      │   │  Retrieval   │    │  (Fine-tuned     │
  │  Scraper      │   │  Graph       │    │   Qwen2.5 +      │
  └───────┬───────┘   └──────┬───────┘    │   LoRA adapter)  │
          │                  │            └────────┬─────────┘
          ▼                  ▼                     │
  ┌───────────────────────────────────────────────────────────┐
  │                      Data Layer                           │
  │                                                           │
  │   MongoDB                ChromaDB               MongoDB   │
  │  (articles,            (text chunks,            (sessions,│
  │   analysis,             embeddings)              history) │
  │   statistics)                                             │
  └───────────────────────────────────────────────────────────┘
```

---

## Key Features

### 🤖 Fine-Tuned Language Model
- Base model: `Qwen/Qwen2.5-1.5B-Instruct` fine-tuned with **LoRA** (rank 64) via **LLaMA-Factory**
- Training data generated through **knowledge distillation** from GPT-4o-mini (2,700+ examples)
- Two specialised tasks baked into the adapter: **structured extraction** (NewsDetails schema) and **translation** (Arabic → English / French)
- Training tracked with **Weights & Biases**; adapter published to **Hugging Face Hub**

### 🔍 Retrieval-Augmented Generation
- **LangGraph** state graph orchestrates a two-node pipeline: `retrieve → grade_chunks`
- Every retrieved chunk is individually graded for relevance by an LLM before being included in the prompt — reducing noise and hallucination risk
- Conversation memory is managed through a **rolling compressed summary** (≤ 200 words) updated after each turn, so context never stale-bloats the context window
- Full turn-by-turn history persisted to MongoDB for auditability

### 📰 Article Deep-Dive Pipeline
- Phase 0 — URL analysis with Mongo cache: avoids re-analysing articles already seen
- Phase 1 — Per-bullet ChromaDB retrieval: each summary bullet is independently embedded and queried, returning supporting evidence from *other* articles
- Phase 2 — Streaming generation: a professionally structured article is streamed token-by-token via SSE, backed by a `asyncio.Queue` bridge between the synchronous OpenAI generator and the async event loop
- Optional translation of the generated article via the fine-tuned model

### ⚡ Streaming Architecture
All generation endpoints return `text/event-stream` responses. The Queue bridge pattern used throughout ensures the FastAPI event loop is **never blocked**: the sync OpenAI/model generator runs in a thread pool and pushes tokens into a queue; the event loop consumes and forwards them to the client immediately.

### 📊 Daily Analytics
- Each article's category is recorded in a dedicated MongoDB statistics collection, enabling daily and historical category-distribution reporting
- Daily Telegram digests are auto-generated and posted by the scheduler using APScheduler with a CronTrigger

---

## Fine-tuning Pipeline

```
Raw Arabic news articles (2,400 samples)
          │
          ▼
   GPT-4o-mini (teacher)
   ├── Task 1: Structured extraction → NewsDetails JSON
   └── Task 2: Translation → English / French JSON
          │
          ▼
   sft.jsonl  (2,766 labelled examples)
          │
          ▼
   LLaMA-Factory — SFT with LoRA
   ├── Model  : Qwen/Qwen2.5-1.5B-Instruct
   ├── Rank   : 64   |  Target: all layers
   ├── Epochs : 3    |  LR: 1e-4  (cosine schedule)
   └── cutoff_len: 3500 tokens
          │
          ▼
   bakrianoo/news-analyzer  (Hugging Face Hub)
```

The distillation loop tracks cumulative token cost in real time, printing a running estimate every three samples.

---

## RAG Pipeline

```
User question
      │
      ▼
embed_text()  ──►  ChromaDB.query()  ──►  raw_chunks
                                               │
                                               ▼
                                    grade_chunks (LLM, concurrent)
                                               │
                                    ┌──────────┴────────────┐
                                    │  relevant_chunks      │
                                    └──────────┬────────────┘
                                               ▼
                             build_answer_prompt()
                             (+ rolling summary + last exchange)
                                               │
                                               ▼
                              generate_text_stream()  →  SSE tokens
                                               │
                                    ┌──────────┴────────────┐
                                    │  extract_links        │
                                    │  summarize_history    │  (followup only)
                                    │  update_session()     │
                                    └───────────────────────┘
```

---

## Article Deep-Dive Pipeline

```
Article URL
     │
     ▼
MongoDB cache lookup
     ├── HIT  → load analysis directly
     └── MISS → scrape page → generate_analysis() → persist to Mongo
     │
     ▼
For each summary bullet (concurrent):
     embed_text()  →  ChromaDB.query()  →  exclude source URL
     │
     ▼
build_digest_prompt()
(overview + per-bullet facts + supporting context chunks)
     │
     ▼
stream_digest()  →  SSE token stream  →  [DONE] frame with source links
     │
     ▼
(Optional) generate_english_translation() / generate_french_translation()
```

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/api/v1/` | Health check / version |
| `POST` | `/api/v1/scrape/bulk-finetune` | Scrape N articles to JSONL for fine-tuning |
| `POST` | `/api/v1/scrape/daily` | Scrape today's articles → Mongo + ChromaDB + Telegram |
| `POST` | `/api/v1/scrape/url` | Scrape a specific article URL |
| `POST` | `/api/v1/sessions` | Create a new conversation session |
| `DELETE` | `/api/v1/sessions/{session_id}` | Delete a session |
| `GET`  | `/api/v1/sessions/{session_id}/history` | Retrieve full turn history |
| `POST` | `/api/v1/ask` | Stateless RAG question (SSE stream) |
| `POST` | `/api/v1/ask/followup` | Session-aware follow-up question (SSE stream) |
| `POST` | `/api/v1/article/digest` | Full deep-dive pipeline for a URL (SSE stream) |
| `POST` | `/api/v1/article/translate` | Translate generated article (en / fr) |

### SSE Wire Format

**`/ask` and `/ask/followup`**
```
data: <token>\n\n
...
data: [DONE] {"source_links": [...], "session_id": "..."}\n\n
```

**`/article/digest`**
```
data: [META] {"story_title": ..., "story_keywords": [...], ...}\n\n
data: <token>\n\n
...
data: [DONE] {"source_links": [...]}\n\n
```

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| **API framework** | FastAPI + Uvicorn |
| **Orchestration** | LangGraph |
| **LLM fine-tuning** | LLaMA-Factory, PEFT (LoRA), Transformers |
| **Base model** | Qwen2.5-1.5B-Instruct |
| **Generation (cloud)** | OpenAI API (GPT-4o-mini) |
| **Embeddings** | OpenAI `text-embedding-*` |
| **Vector store** | ChromaDB (cosine similarity, persistent) |
| **Document store** | MongoDB (articles, analysis, statistics, sessions) |
| **Async sessions** | Motor (async MongoDB driver) |
| **Scraping** | Selenium (Firefox / geckodriver) + PyQuery |
| **Scheduling** | APScheduler (CronTrigger) |
| **Experiment tracking** | Weights & Biases |
| **Model hosting** | Hugging Face Hub |
| **Notifications** | Telegram Bot API |
| **UI** | Gradio |
| **Schema validation** | Pydantic v2 |

---

## Project Structure

```
newsX/
│
├── routes/
│   ├── welcome.py              # Health check
│   ├── scraping.py             # Scraping endpoints + scheduler setup
│   ├── rag_router.py           # /ask and /ask/followup (streaming)
│   ├── article_router.py       # /article/digest and /article/translate
│   └── session_router.py       # Session CRUD
│
├── models/
│   ├── analyzingANDtranslating/
│   │   ├── analyze_Trans_deps.py   # Fine-tuned model wrapper (A_TDeps)
│   │   ├── article_nodes.py        # Deep-dive pipeline nodes
│   │   └── article_state.py        # ArticleState Pydantic model
│   └── ask/
│       ├── rag_deps.py             # RAGDeps (embeddings + generation + ChromaDB)
│       ├── rag_graph.py            # LangGraph retrieval graph
│       ├── rag_nodes.py            # retrieve, grade_chunks, stream_answer, …
│       ├── rag_state.py            # RAGState Pydantic model
│       └── session_store.py        # MongoSessionStore (Motor)
│
├── schema/
│   ├── AnalyizingSchema.py         # NewsDetails, Entity, StoryCategory
│   └── TranslatingSchema.py        # TranslatedStory
│
├── Scraper/
│   ├── BaseScrapingModel.py        # Selenium Firefox base
│   ├── BulkScraping.py             # BulkScraper (feed + daily mode)
│   └── PageScraping.py             # PageScraper (single URL)
│
├── stores/
│   └── llm/
│       ├── LLM_Factory.py          # Factory: openai | huggingface
│       ├── providers/
│       │   ├── OpenAIProvider.py   # Blocking + streaming generation, embeddings
│       │   └── HugginFaceProvider.py
│       └── LLMEnums.py
│
├── helpers/
│   └── Config.py                   # Pydantic Settings (.env)
│
├── notebooks/
│   └── llm_finetuning.ipynb        # Full distillation + fine-tuning walkthrough
│
├── news_ui.py                      # Gradio front-end
├── requirements.txt
└── README.md
```

---

## Getting Started

### Prerequisites

- Python 3.10+
- MongoDB running locally or a connection URI
- Firefox + geckodriver (see note below)
- OpenAI API key
- Hugging Face token (for pushing the fine-tuned adapter)

### Firefox / geckodriver on WSL / Linux

```bash
wget https://github.com/mozilla/geckodriver/releases/download/v0.35.0/geckodriver-v0.35.0-linux64.tar.gz
tar -xzf geckodriver-v0.35.0-linux64.tar.gz
mkdir -p driver && mv geckodriver driver/geckodriver
chmod +x driver/geckodriver

sudo add-apt-repository -y ppa:mozillateam/ppa
echo 'Package: *
Pin: release o=LP-PPA-mozillateam
Pin-Priority: 1001' | sudo tee /etc/apt/preferences.d/mozilla-firefox
sudo apt update && sudo apt install -y firefox
```

### Installation

```bash
git clone https://github.com/<your-username>/newsX.git
cd newsX
pip install -r requirements.txt
```

### Environment Variables

Copy `.env.example` to `.env` and fill in your values:

```env
APP_NAME=NewsX
APP_VERSION=1.0.0

MONGO_URL=mongodb://localhost:27017
MONGO_DB=newsX
MONGO_COL_1=articles
MONGO_COL_STATS=statistics
MONGO_COL_ANALYSIS=analysis

CHROMA_PATH=./chroma_store
CHROMA_COLL=news_chunks

OPENAI_API_KEY=sk-...
OPENAI_API_URL=
OPENAI_GENERATION_MODEL_ID=gpt-4o-mini
OPENAI_EMBEDDING_MODEL_ID=text-embedding-3-small

PROVIDERS=openai

ANALYZER_MODEL_NAME=Qwen/Qwen2.5-1.5B-Instruct
ADAPTER_NAME=AbdoSaad24/news-analyzer

BotToken=<telegram-bot-token>
chatID=<telegram-chat-id>

LANGCHAIN_API_KEY=
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
LANGCHAIN_PROJECT=newsX
```

### Run the API

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Run the UI

```bash
python news_ui.py
# → http://localhost:7860
```

---

## Configuration

| Variable | Description |
|----------|-------------|
| `ANALYZER_MODEL_NAME` | Base HuggingFace model ID |
| `ADAPTER_NAME` | LoRA adapter repo on Hugging Face Hub |
| `OPENAI_GENERATION_MODEL_ID` | Model used for RAG grading, summarisation, and daily digest generation |
| `OPENAI_EMBEDDING_MODEL_ID` | Embedding model used for both indexing and retrieval |
| `CHROMA_PATH` | Local path for the persistent ChromaDB store |
| `MONGO_COL_ANALYSIS` | Collection where per-URL analysis results are cached |
| `MONGO_COL_STATS` | Collection for per-article category statistics |
---

*Built end-to-end — scraping, distillation, fine-tuning, RAG, streaming, and UI — as a demonstration of production ML engineering.*