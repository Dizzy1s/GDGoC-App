# Helpers/clean_faiss_index.py
"""
Deduplicate & prune long-term memories, keeping only meaningful ones.
Run from anywhere (folder-aware).
"""

import sys, pathlib, shutil, os
# ── make project root importable ───────────────────────────────────────────
ROOT = pathlib.Path(__file__).resolve().parents[1]   # one level up from Helpers/
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# ───────────────────────────────────────────────────────────────────────────

import long_term
from importance import is_important
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

def main() -> None:
    long_term.init_vectorstore()
    vs = long_term.vectorstore
    path = pathlib.Path(long_term.vectorstore_path).resolve()
    print("→ FAISS folder:", path)

    if vs.index.ntotal == 0:
        print("Index empty – nothing to clean.")
        return

    docs = vs.similarity_search("", k=vs.index.ntotal)
    kept, seen = [], set()

    for d in docs:
        txt = d.page_content.strip()
        if txt in seen:
            continue
        
        sender = d.metadata.get("sender", "user")
        if sender == "user":
            if not is_important(txt):
                continue
        else:
            if not is_important(txt):      # plus any NPC-specific rule if you want
                continue

        kept.append(Document(page_content=txt, metadata=d.metadata))
        seen.add(txt)

    print(f"Kept {len(kept)} of {len(docs)} memories.")

    tmp = path.parent / "faiss_tmp"
    if tmp.exists():
        shutil.rmtree(tmp, ignore_errors=True)

    clean_vs = FAISS.from_documents(kept, long_term.embedding)
    clean_vs.save_local(str(tmp))

    shutil.rmtree(path, ignore_errors=True)
    os.rename(tmp, path)
    print("Clean index written.")

if __name__ == "__main__":
    main()
