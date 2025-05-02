app.py is the main Flask backend

Terminal:

dependencies installation:

pip install flask flask-cors google-generativeai python-dotenv sentence-transformers

run backend:

python app.py

run local react server:

python -m http.server

Access:

http://localhost:8000/

Notes (memory-integration branch):

- Adds long-term and short-term memory with FAISS vector search.
- Important facts and health notes are saved and recalled in conversations.
- Same backend works with both voice and text routes.