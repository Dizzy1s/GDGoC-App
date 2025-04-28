## README – Memory-Enhanced Multi-NPC Medical Chatbot

### Overview
This branch adds a **long-term / short-term memory subsystem** (FAISS + LangChain) to the GDGoC medical-support chatbot.  
Key features :

| Feature | Description |
|---------|-------------|
| **Short-term memory** | Keeps the last *N* user + NPC turns in RAM and persists them to `history.json`. |
| **Long-term memory**  | Stores only *important* facts (health issues, symptoms, plans, goals, emotions, etc.) in a FAISS vector index on disk (`faiss_memory_index/`). |
| **Smart importance filter** | `importance.py` decides what gets saved, with medical-specific keywords + optional Gemini fallback. |
| **Relevant-memory injection** | Before every response, only memories *semantically similar* to the current user input (cos ≥ 0.60) are injected into the prompt. |
| **Manual commands** | `/show memories`, `/remember this …`, `/clear`, `/exit`, etc. |
| **Helper CLIs** | `Helpers/list_long_term.py`, `clean_faiss_index.py`, `clear_long_term.py`. |


### Directory layout
```
memory/
│
├─ main.py                     ← multi-NPC CLI (entry-point)
├─ long_term.py                ← FAISS wrapper
├─ short_term.py               ← short-term cache
├─ utils_memory.py             ← Message dataclass & helpers
├─ importance.py               ← smart importance classifier (medical-aware)
├─ list_long_term.py           ← list all memories

├─ Helpers/
│   ├─ clean_faiss_index.py    ← deduplicate / prune index
│   └─ clear_long_term.py      ← wipe index (placeholder left)
│
├─ faiss_memory_index/         ← on-disk vectors (auto-created)
├─ history.json                ← short-term persistence
├─ .env                        ← YOUR Gemini key (never commit)
└─ requirements.txt
```


### Quick-start

```bash
# 1 – create & activate venv
python -m venv venv
source venv/Scripts/activate         # Windows
# . venv/bin/activate                # macOS/Linux

# 2 – install deps
pip install -r requirements.txt

# 3 – set your key
cp .env.example .env
#   then edit .env and place GOOGLE_API_KEY=...

# 4 – run chat
python main.py
```


### Helper scripts

```bash
python list_long_term.py     # show stored memories
python Helpers/clean_faiss_index.py  # deduplicate / keep only important lines
python Helpers/clear_long_term.py    # wipe index (placeholder recreated)
```



### Sample test prompts

| Goal | Example prompt |
|------|----------------|
| Save a personal fact | `Hi Leo, remember this: I restarted my gym routine three weeks ago.` |
| Recall memory | `Hi Leo, what do I love?` |
| Show all memories | `Leo, show memories` |
| Medical context | `Dr. Hale, I’ve been having migraines twice a week and taking Ibuprofen.` |
| Decision / plan | `Raya, remember this: I decided to book a therapist appointment on Friday.` |
| Retrieve health note | `Dr. Hale, what did I say about my migraines?` |

You should see only meaningful lines entering long-term memory, and NPCs will reference them contextually on future turns.


### Contributing / next steps
* Integrate helpers into the Flask backend (`app.py`) as a service layer.
* Add unit tests under `tests/`.
* Hook up memory listing & clearing to the React UI.

