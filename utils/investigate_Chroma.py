"""
ChromaDB Inspector
──────────────────
A clean, professional CLI tool to investigate what lives inside your ChromaDB.

Usage:
    python chroma_inspector.py --path ./chroma_db
    python chroma_inspector.py --path ./chroma_db --collection my_col
    python chroma_inspector.py --path ./chroma_db --search "some query text"
"""

import argparse
import json
import sys
from typing import Optional

try:
    import chromadb
except ImportError:
    print("[ERROR] chromadb is not installed. Run: pip install chromadb")
    sys.exit(1)

# ── ANSI Colors ───────────────────────────────────────────────────────────────
R  = "\033[0m"          # reset
B  = "\033[1m"          # bold
DIM= "\033[2m"          # dim
C  = "\033[96m"         # cyan
G  = "\033[92m"         # green
Y  = "\033[93m"         # yellow
RE = "\033[91m"         # red
M  = "\033[95m"         # magenta
W  = "\033[97m"         # white

def hr(char="─", width=65, color=DIM):
    print(f"{color}{char * width}{R}")

def header(title: str):
    print()
    hr("═")
    print(f"{B}{C}  {title}{R}")
    hr("═")

def section(title: str):
    print()
    print(f"{B}{Y}▸ {title}{R}")
    hr()

def kv(key: str, value, key_color=C, val_color=W):
    print(f"  {key_color}{key:<22}{R} {val_color}{value}{R}")

def badge(text: str, color=G):
    return f"{color}[{text}]{R}"

# ── Core Inspector ────────────────────────────────────────────────────────────

def connect(path: str) -> chromadb.PersistentClient:
    print(f"\n{DIM}Connecting to ChromaDB at:{R} {C}{path}{R}")
    try:
        client = chromadb.PersistentClient(path=path)
        print(f"{G}✓ Connected{R}")
        return client
    except Exception as e:
        print(f"{RE}✗ Failed to connect: {e}{R}")
        sys.exit(1)


def list_collections(client: chromadb.PersistentClient):
    header("Collections Overview")
    collections = client.list_collections()

    if not collections:
        print(f"  {Y}No collections found.{R}")
        return []

    print(f"  Found {B}{G}{len(collections)}{R} collection(s):\n")

    for i, col in enumerate(collections, 1):
        col_obj = client.get_collection(col.name)
        count   = col_obj.count()
        meta    = col.metadata or {}

        print(f"  {B}{i}.{R} {M}{col.name}{R}")
        kv("Documents", f"{count:,}", val_color=G if count > 0 else Y)
        kv("Metadata",  json.dumps(meta) if meta else "—", val_color=DIM)
        print()

    return collections


def inspect_collection(client: chromadb.PersistentClient, name: str, sample_size: int = 5):
    header(f"Inspecting: {name}")

    try:
        col = client.get_collection(name)
    except Exception as e:
        print(f"{RE}Collection '{name}' not found: {e}{R}")
        return

    total = col.count()
    kv("Collection name", name)
    kv("Total chunks",    f"{total:,}")
    kv("Distance metric", col.metadata.get("hnsw:space", "l2"))

    if total == 0:
        print(f"\n  {Y}Collection is empty.{R}")
        return

    # ── Sample documents ──────────────────────────────────────────────────
    section(f"Sample Documents (first {min(sample_size, total)})")

    results = col.get(
        limit            = min(sample_size, total),
        include          = ["documents", "metadatas", "embeddings"],
    )

    ids       = results.get("ids", [])
    docs      = results.get("documents", [])
    metas     = results.get("metadatas", [])
    embeddings= results.get("embeddings", [])

    for i, (doc_id, doc, meta, emb) in enumerate(zip(ids, docs, metas, embeddings), 1):
        print(f"\n  {B}{badge(str(i))} {C}{doc_id}{R}")
        hr("·", 60, DIM)

        # Metadata
        if meta:
            print(f"  {B}Metadata:{R}")
            for k, v in meta.items():
                kv(f"  {k}", v, key_color=DIM)

        # Document preview
        preview = doc[:280].replace("\n", " ") if doc else "—"
        print(f"\n  {B}Content preview:{R}")
        print(f"  {DIM}{preview}{'...' if len(doc or '') > 280 else ''}{R}")

        # Embedding info
        if emb:
            dims = len(emb)
            sample_vals = ", ".join(f"{v:.4f}" for v in emb[:4])
            print(f"\n  {B}Embedding:{R} {DIM}{dims} dims — [{sample_vals}, ...]{R}")

    # ── Metadata field analysis ───────────────────────────────────────────
    section("Metadata Field Analysis")

    all_results = col.get(include=["metadatas"])
    all_metas   = all_results.get("metadatas", [])

    if all_metas:
        fields: dict = {}
        for m in all_metas:
            for k, v in (m or {}).items():
                if k not in fields:
                    fields[k] = set()
                fields[k].add(str(v))

        for field, values in fields.items():
            unique_count = len(values)
            sample       = list(values)[:4]
            sample_str   = ", ".join(f'"{v}"' for v in sample)
            if unique_count > 4:
                sample_str += f", ... (+{unique_count - 4} more)"
            kv(field, f"{unique_count} unique → {DIM}{sample_str}{R}")

    # ── Chunk distribution per source ─────────────────────────────────────
    section("Chunks per Source (mongo_id)")

    mongo_ids = {}
    for m in all_metas:
        mid = (m or {}).get("mongo_id", "unknown")
        mongo_ids[mid] = mongo_ids.get(mid, 0) + 1

    if mongo_ids:
        sorted_ids = sorted(mongo_ids.items(), key=lambda x: x[1], reverse=True)
        top = sorted_ids[:10]
        for mid, count in top:
            bar  = "█" * min(count, 30)
            kv(mid[:28], f"{G}{bar}{R} {count}", key_color=DIM)
        if len(sorted_ids) > 10:
            print(f"\n  {DIM}... and {len(sorted_ids) - 10} more sources{R}")


def search_collection(
    client:     chromadb.PersistentClient,
    name:       str,
    query:      str,
    n_results:  int = 5,
    embed_fn=None,
):
    header(f"Semantic Search in: {name}")
    print(f"  {DIM}Query:{R} {W}{query}{R}\n")

    try:
        col = client.get_collection(name, embedding_function=embed_fn)
    except Exception as e:
        print(f"{RE}Could not get collection: {e}{R}")
        return

    try:
        results = col.query(
            query_texts = [query],
            n_results   = min(n_results, col.count()),
            include     = ["documents", "metadatas", "distances"],
        )
    except Exception as e:
        print(f"{RE}Search failed: {e}{R}")
        print(f"{Y}Tip: If the collection has no embedding_function, pass --embed-model.{R}")
        return

    ids       = results["ids"][0]
    docs      = results["documents"][0]
    metas     = results["metadatas"][0]
    distances = results["distances"][0]

    for i, (doc_id, doc, meta, dist) in enumerate(zip(ids, docs, metas, distances), 1):
        similarity = 1 - dist   # cosine distance → similarity
        color      = G if similarity > 0.7 else Y if similarity > 0.4 else RE

        print(f"  {B}{badge(str(i))} {C}{doc_id}{R}")
        kv("Similarity", f"{color}{similarity:.4f}{R} ({dist:.4f} distance)")

        if meta:
            kv("Title",      meta.get("title", "—"),      val_color=W)
            kv("Chunk",      f"{meta.get('chunk_index','?')} / {meta.get('total_chunks','?')}",val_color=DIM)
            kv("Day",        meta.get("current_day", "—"), val_color=DIM)

        preview = (doc or "")[:250].replace("\n", " ")
        print(f"  {DIM}{preview}{'...' if len(doc or '') > 250 else ''}{R}")
        hr("·", 60, DIM)
        print()


def stats_summary(client: chromadb.PersistentClient):
    header("Database Summary")

    collections = client.list_collections()
    total_docs  = 0

    for col in collections:
        c = client.get_collection(col.name)
        total_docs += c.count()

    kv("Total collections", len(collections), val_color=G)
    kv("Total chunks",      f"{total_docs:,}", val_color=G)


# ── CLI ───────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="ChromaDB Inspector — investigate your vector store",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument("--path",        required=True,  help="Path to ChromaDB directory")
    p.add_argument("--collection",  default=None,   help="Inspect a specific collection")
    p.add_argument("--search",      default=None,   help="Run a semantic search query")
    p.add_argument("--n",           type=int, default=5, help="Number of results to show (default: 5)")
    p.add_argument("--sample",      type=int, default=5, help="Sample size for inspection (default: 5)")
    p.add_argument("--embed-model", default=None,   help="SentenceTransformer model for search (optional)")
    return p


def main():
    parser = build_parser()
    args   = parser.parse_args()

    client = connect(args.path)

    # ── Always show overview ──────────────────────────────────────────────
    collections = list_collections(client)
    stats_summary(client)

    # ── Deep inspect a collection ─────────────────────────────────────────
    target = args.collection
    if not target and collections:
        # default to first collection if only one exists
        if len(collections) == 1:
            target = collections[0].name
            print(f"\n{DIM}Auto-selected collection: {target}{R}")

    if target:
        inspect_collection(client, target, sample_size=args.sample)

        # ── Search ────────────────────────────────────────────────────────
        if args.search:
            embed_fn = None
            if args.embed_model:
                try:
                    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
                    embed_fn = SentenceTransformerEmbeddingFunction(model_name=args.embed_model)
                    print(f"{G}✓ Embedding model loaded: {args.embed_model}{R}")
                except ImportError:
                    print(f"{Y}sentence-transformers not installed. Trying without embed_fn.{R}")

            search_collection(client, target, args.search, n_results=args.n, embed_fn=embed_fn)

    print()
    hr("═")
    print(f"{DIM}  Done.{R}")
    print()


if __name__ == "__main__":
    main()




# # See all collections + summary
# python chroma_inspector.py --path assets/chroma_db

# # Deep inspect a specific collection
# python chroma_inspector.py --path assets/chroma_db --collection my_collection

# # Search inside a collection
# python chroma_inspector.py --path assets/chroma_db --collection my_collection --search "your query here"

# # Search with a custom embedding model
# python chroma_inspector.py --path assets/chroma_db --collection my_collection \
#   --search "your query" --embed-model "all-MiniLM-L6-v2"

# # Control how many results/samples to show
# python chroma_inspector.py --path assets/chroma_db --collection my_collection --sample 10 --n 8