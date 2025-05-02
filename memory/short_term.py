# short_term.py
"""
Short-term memory with on-disk persistence.

• Keeps the last MAX_TURNS exchanges per NPC.
• Writes to `history.json` after every update.
• Loads the cache on first import, so state survives restarts.
"""

from __future__ import annotations
import json
import os
from typing import Dict, List

from utils_memory import Message

# --- Tunables --------------------------------------------------------------

MAX_TURNS: int = 6          # how many recent messages to keep per NPC
CACHE_PATH: str = "history.json"

# ---------------------------------------------------------------------------

def _load_cache() -> Dict[str, List[Message]]:
    """Read history.json → in-memory dict[{npc_id}: List[Message]]."""
    if not os.path.exists(CACHE_PATH):
        return {}

    try:
        with open(CACHE_PATH, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return {npc: [Message.from_dict(m) for m in msgs] for npc, msgs in raw.items()}
    except Exception:
        # Corrupted or incompatible file—start fresh
        return {}


def _flush() -> None:
    """Serialize `short_term_memory` back to history.json (pretty-printed)."""
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {npc: [m.to_dict() for m in msgs] for npc, msgs in short_term_memory.items()},
            f,
            ensure_ascii=False,
            indent=2,
        )


# In-RAM store, pre-populated from disk
short_term_memory: Dict[str, List[Message]] = _load_cache()

# ---------------------------------------------------------------------------

def add_to_short_term(npc_id: str, message: Message) -> None:
    """Append a message and persist."""
    short_term_memory.setdefault(npc_id, []).append(message)
    # Trim to MAX_TURNS
    short_term_memory[npc_id] = short_term_memory[npc_id][-MAX_TURNS:]
    _flush()


def get_short_term(npc_id: str, limit: int = MAX_TURNS) -> List[Message]:
    """Return the most recent `limit` messages for `npc_id`."""
    return short_term_memory.get(npc_id, [])[-limit:]


def clear_short_term(npc_id: str) -> None:
    """Delete an NPC’s short-term memory."""
    if npc_id in short_term_memory:
        del short_term_memory[npc_id]
        _flush()


def get_all_short_term() -> Dict[str, List[Message]]:
    """Expose the full in-RAM cache (read-only use!)."""
    return short_term_memory
