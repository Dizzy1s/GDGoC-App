# Helpers/clear_long_term.py
"""
Delete the entire FAISS index and recreate a placeholder index so
the rest of the code never crashes on empty store.
"""

import sys, pathlib, shutil
# ── add project root to import path ───────────────────────────────────────
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# ───────────────────────────────────────────────────────────────────────────

import long_term

def main() -> None:
    idx = pathlib.Path(long_term.vectorstore_path).resolve()
    if idx.exists():
        shutil.rmtree(idx)
        print(f"Deleted folder: {idx}")
    else:
        print("No FAISS index folder found – nothing to delete.")

    long_term.init_vectorstore()        # writes placeholder
    print("Long-term memory cleared (placeholder index created).")

if __name__ == "__main__":
    main()
