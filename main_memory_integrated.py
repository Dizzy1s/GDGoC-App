"""
Entry-point that boots the memory-enhanced multi-NPC CLI.

Run:
    python main_memory_integrated.py
"""

import sys, pathlib

# ── ensure repo root + memory package are on import path ──────────────────
ROOT = pathlib.Path(__file__).parent.resolve()
MEMORY_DIR = ROOT / "memory"
if str(MEMORY_DIR) not in sys.path:
    sys.path.insert(0, str(MEMORY_DIR))

# ── import and launch the real chat loop ──────────────────────────────────
from memory.main import chat_loop, init_vectorstore, npc_list
from memory.short_term import clear_short_term  # optional: start fresh

if __name__ == "__main__":
    init_vectorstore()
    # start each session with empty short-term cache
    for npc in npc_list:
        clear_short_term(npc.name)
    chat_loop()
