
## Memory Sub‑System & Integration 

### 1 — What’s in here?
This package adds a **medical‑aware long‑term / short‑term memory layer** to the
GDGoC multi‑NPC chatbot.  
It now powers **two entry‑points**:

| Entry‑point | Purpose |
|-------------|---------|
| **`app.py` (root)** | Flask backend (voice + text) **with memory injection** — used by the React UI. |
| **`main_memory_integrated.py` (root)** | One‑command CLI wrapper for quick demos. |
| **`memory/main.py`** | Full CLI with manual commands (`/show memories`, `/remember`, etc.). |

The same memory engine drives them all.


### 2 — Key features

| Feature | Description |
|---------|-------------|
| **Short‑term memory** | Last *N* turns per NPC; cached in‑RAM & persisted to `history.json`. |
| **Long‑term memory (FAISS)** | Stores only **important** user facts & meaningful NPC replies in `faiss_memory_index/`. |
| **Medical importance filter** | `importance.py` checks health keywords + context to decide what to keep. |
| **Auto‑recall** | On every reply the NPC receives *≤3* memories with cosine sim ≥ 0.60 relating to the user’s latest line. |
| **Voice + Text** | `/voice_chat` and `/chat` both write & read the same memories. |
| **Helper CLIs** | `Helpers/clean_faiss_index.py`, `clear_long_term.py`, `list_long_term.py`. |


### 3 — Folder layout

```

memory/
│
├─ main.py                  ← full CLI demo
├─ long_term.py             ← FAISS wrapper
├─ short_term.py            ← short‑term cache
├─ utils_memory.py          ← Message dataclass & helpers
├─ importance.py            ← medical‑aware importance rules
├─ list_long_term.py        ← list all memories
│
├─ Helpers/
│   ├─ clean_faiss_index.py ← deduplicate / prune index
│   └─ clear_long_term.py   ← wipe index (makes fresh placeholder)
│
├─ faiss_memory_index/      ← on‑disk vectors (auto‑created)
├─ history.json             ← short‑term persistence
└─ requirements.txt

```

Additional root‑level helpers added by this branch:

```

main_memory_integrated.py   ← CLI launcher

````


### 4 — Quick start

```bash
# 1 – create venv
python -m venv venv
source venv/Scripts/activate        # Windows
# . venv/bin/activate               # macOS/Linux

# 2 – install ALL deps
pip install -r memory/requirements.txt
pip install flask flask-cors google-generativeai python-dotenv sentence-transformers

# 3 – add your key
echo GOOGLE_API_KEY=your‑key‑here > .env
````

#### Option A — Run full backend + React

```bash
python app.py              # Flask + memory
python -m http.server      # static React
open http://localhost:8000/
```

#### Option B — Quick CLI demo

```bash
python main_memory_integrated.py
```


### 5 — Helper scripts

```bash
python memory/list_long_term.py               # list stored memories
python memory/Helpers/clean_faiss_index.py    # deduplicate & shrink index
python memory/Helpers/clear_long_term.py      # wipe index
```

### 6 — Future UI improvements for memory features   

Add a collapsible right‑hand panel labelled “Memories” that, on click, fetches and lists the current NPC’s long‑term memories (GET /memories/<npc>).

Place a  “Remember this” button beside the text box that posts the message with a force_save flag to store it permanently.





