

# GDGoC Medical Chatbot – Backend + Memory System


## Terminal Instructions

### Install dependencies

```bash
pip install flask flask-cors google-generativeai python-dotenv sentence-transformers
```

> *(If using the memory-enhanced CLI, also install FAISS and LangChain modules separately.)*


### Run Flask backend server

```bash
python app.py
```

This starts the API backend for the chatbot.



### Run local React server (static)

```bash
python -m http.server
```

Then open:

```
http://localhost:8000/
```

This loads the static React frontend.



# 🧠 Memory Integration CLI (Experimental)

A **multi-NPC CLI chatbot** with smart **short-term** and **long-term** memory has been added under `memory/`.

* Built with **LangChain**, **FAISS** vector store, and **medical-aware importance filtering**.
* Not yet connected to Flask — **but can be run locally for demonstrations**.

### To run memory-based chatbot locally:

```bash
pip install -r memory/requirements.txt
python main_memory_integrated.py
```

* This launches a terminal-based chatbot where conversations are saved in memory.
* Memories (e.g., health symptoms, appointments, feelings) are automatically filtered and stored.
* You can manually ask it to "remember" things or "show memories."



# 📁 Project Structure (Important parts)

```
app.py                 ← Flask backend server
index.html             ← Static React frontend
memory/                ← Memory-enhanced multi-NPC chatbot subsystem
main_memory_integrated.py  ← Simple launcher for memory CLI demo
requirements.txt       ← Full dependencies
```



# ⚙ Notes

* `.env` file must be created manually inside memory folder or you can paste your own inside the memory folder. It should contain:

  ```text
  GOOGLE_API_KEY=your-api-key-here
  ```

* `faiss_memory_index/` and `history.json` are automatically created during chatbot memory runs.



# Possible Future Integrations 

* Integrate `memory/` features into Flask backend (`app.py`) to allow online long-term memory in user chats.
* Memory management via admin dashboard (React) — view, clear, or edit stored memories.



