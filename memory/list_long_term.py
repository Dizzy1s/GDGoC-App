# run this file to list the content of the long term memory index
import pathlib, long_term

def main():
    long_term.init_vectorstore()
    vs   = long_term.vectorstore
    path = pathlib.Path(long_term.vectorstore_path).resolve()
    print("→ FAISS folder:", path)

    # remove placeholder
    total = vs.index.ntotal
    if total == 1:
        docs = vs.similarity_search("", k=1)
        if docs[0].page_content == "__PLACEHOLDER__":
            print("No long-term memories stored yet.")
            return

    docs = vs.similarity_search("", k=total)
    buckets = {}
    for d in docs:
        if d.page_content == "__PLACEHOLDER__":
            continue
        npc = d.metadata.get("npc_id", "UNKNOWN")
        buckets.setdefault(npc, []).append(d.page_content)

    if not buckets:
        print("No long-term memories stored yet.")
        return

    for npc, mems in buckets.items():
        print(f"\n=== {npc} – {len(mems)} memories ===")
        for i, m in enumerate(mems, 1):
            print(f"{i:>3}. {m}")

if __name__ == "__main__":
    main()
