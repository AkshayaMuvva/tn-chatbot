"""
setup_rag.py - One-time script to build the ChromaDB vector index.

Run this ONCE before starting the chatbot:
    python setup_rag.py

Optional: force rebuild even if index exists:
    python setup_rag.py --force
"""
import sys
import io
import time
from pathlib import Path

# Force UTF-8 output on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from rag.chunker import create_chunks, get_all_college_names
from rag.embedder import build_index, is_index_ready


def main():
    force = "--force" in sys.argv

    print("=" * 60)
    print("[RAG SETUP] TN Engineering College RAG Index Builder")
    print("=" * 60)

    if is_index_ready() and not force:
        print("[OK] ChromaDB index already exists and is ready!")
        print("     Use --force to rebuild: python setup_rag.py --force")
        return

    print("\n[INFO] Reading college data from CSV...")
    t0 = time.time()
    chunks = create_chunks()
    colleges = get_all_college_names()
    print(f"[OK] Created {len(chunks)} chunks from {len(colleges)} colleges")
    print(f"     Sample: {', '.join(colleges[:5])}...")

    print("\n[INFO] Loading embedding model (all-MiniLM-L6-v2)...")
    print("     First run downloads ~80MB from HuggingFace (cached after).")

    build_index(chunks, force_rebuild=force)

    elapsed = time.time() - t0
    print(f"\n[DONE] Setup complete in {elapsed:.1f}s")
    print("[NEXT] Run the chatbot: streamlit run app.py")


if __name__ == "__main__":
    main()
