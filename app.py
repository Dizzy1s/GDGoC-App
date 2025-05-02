"""
app.py  — Flask backend with multi‑NPC chat + FAISS long/short‑term memory
Run:  python app.py
"""

# ────────── standard libs
import os, random, json, time, tempfile, pathlib, sys
# ────────── third‑party
import pyaudio, wave
import google.generativeai as genai
from dotenv import load_dotenv
from sentence_transformers import util
from flask import Flask, request, jsonify
from flask_cors import CORS
# ────────── project memory package
ROOT = pathlib.Path(__file__).parent.resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))           # ensure "memory" is importable

from memory.long_term import init_vectorstore, add_to_long_term, search_long_term
from memory.short_term import add_to_short_term, clear_short_term
from memory.utils_memory import Message
from memory.importance import is_important

# ═══════════════════ ENV / GEMINI ════════════════════════════════════
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

gemini_model = genai.GenerativeModel(
    "gemini-2.0-flash",
    generation_config={
        "temperature": 0.9,
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 2048,
    },
)

# ═══════════════════ AUDIO CONFIG ════════════════════════════════════
AUDIO_CONFIG = dict(format=pyaudio.paInt16, channels=1, rate=44100, chunk=1024)
audio = pyaudio.PyAudio()

# ═══════════════════ NPC GENERATION ══════════════════════════════════
used_names: set[str] = set()

def generate_human_name(topic: str, attempt: int = 0) -> str:
    """LLM‑creates a culturally diverse realistic first name."""
    avoid = ", ".join(used_names) if used_names else "None yet"
    prompt = f"""Produce a **single realistic human first name** (no surname) for
someone experienced with “{topic}”. Must differ from: {avoid}. No quotes."""
    try:
        resp = gemini_model.generate_content(prompt).text.strip()
        name = " ".join(part.capitalize() for part in resp.split())
        if name.lower() in (n.lower() for n in used_names):
            raise ValueError("duplicate")
        return name
    except Exception:
        fallback = ["Alex", "Maria", "David", "Aisha", "James"]
        return fallback[attempt % len(fallback)]

def generate_diverse_personality(name: str, topic: str) -> dict:
    prompt = f"""
Return ONLY JSON.

{{
  "traits": "4‑5 comma traits",
  "backstory": "specific personal experience with {topic}",
  "interests_hobbies": "4‑5 comma hobbies",
  "attitude": "brief outlook",
  "tone": "speaking style",
  "appearance": "age, cultural cues",
  "introversion": "0.0‑1.0",
  "assertiveness": "0.0‑1.0"
}}
Name = {name}
"""
    for _ in range(3):
        try:
            txt = gemini_model.generate_content(prompt).text.strip()
            if txt.startswith("```json"):
                txt = "\n".join(txt.splitlines()[1:-1])
            data = json.loads(txt)
            data["name"] = name
            data["topic"] = topic
            return data
        except Exception:
            time.sleep(0.5)
    # fallback
    return {
        "name": name,
        "topic": topic,
        "traits": "thoughtful, unique",
        "backstory": f"Shaped by {topic}",
        "interests_hobbies": "reading, art, music",
        "attitude": "calm",
        "tone": "gentle",
        "appearance": "average height, casual attire",
        "introversion": "0.5",
        "assertiveness": "0.5",
    }

class NPC:
    def __init__(self, personality: dict):
        self.name           = personality["name"]
        self.personality    = personality["traits"]
        self.role           = personality["topic"]
        self.introversion   = float(personality["introversion"])
        self.assertiveness  = float(personality["assertiveness"])
        self.interests      = [i.strip() for i in personality["interests_hobbies"].split(",")][:9]
        self.interest_vecs  = self._encode_interests()
        self.personality_data = personality
        self.relationships  = {}
        self.last_spoken    = -1

    def _encode_interests(self):
        vecs = []
        for intr in self.interests:
            vec = genai.embed_content(
                model="models/embedding-001",
                content=intr,
                task_type="SEMANTIC_SIMILARITY",
            )["embedding"]
            vecs.append(vec)
        return vecs

# create 5 diverse NPCs
topic = "Anxiety and stage fright"
npc_list: list[NPC] = []
for i in range(5):
    name = generate_human_name(topic, i)
    used_names.add(name)
    npc_list.append(NPC(generate_diverse_personality(name, topic)))

# init relationships
for npc in npc_list:
    npc.relationships["User"] = {"bond": 0.5, "trust": 0.5}
    for other in npc_list:
        if other is npc:
            continue
        npc.relationships[other.name] = {
            "bond": round(random.uniform(0.3, 0.7), 2),
            "trust": round(random.uniform(0.4, 0.8), 2),
        }

# ═══════════════════ RELATIONSHIP HELPERS ════════════════════════════════
def update_relationship(npc: NPC, target: str, text: str, emotion: str | None = None):
    t = text.lower()
    rel = npc.relationships.get(target, {"bond": 0.5, "trust": 0.5})
    if any(word in t for word in ("thank", "agree", "yes", "right")):
        rel["bond"]  = min(rel["bond"] + 0.05, 1.0)
        rel["trust"] = min(rel["trust"] + 0.03, 1.0)
    elif any(word in t for word in ("no", "disagree", "annoy", "hate")):
        rel["bond"]  = max(rel["bond"] - 0.05, 0.0)
        rel["trust"] = max(rel["trust"] - 0.05, 0.0)
    npc.relationships[target] = rel

def interest_match_score(npc: NPC, text: str) -> float:
    if not text.strip():
        return 0.0
    vec = genai.embed_content(
        model="models/embedding-001", content=text, task_type="SEMANTIC_SIMILARITY"
    )["embedding"]
    return max(util.pytorch_cos_sim([vec], [iv]).item() for iv in npc.interest_vecs)

def detect_addressed_npc(text: str) -> NPC | None:
    tl = text.lower()
    for n in npc_list:
        if n.name.lower() in tl:
            return n
    return None

def compute_relevancy(npc: NPC, last_speaker: str, last_text: str, turn: int) -> float:
    rel_score = interest_match_score(npc, last_text)
    bond = npc.relationships.get(last_speaker, {}).get("bond", 0.5)
    trust= npc.relationships.get(last_speaker, {}).get("trust", 0.5)
    time_since = turn - npc.last_spoken
    drive = (1 - npc.introversion) + npc.assertiveness
    return (rel_score * 2 + bond + trust + time_since * 0.2) * (0.5 + 0.5 * drive)

def select_speaker(last_speaker: str, last_text: str, turn: int) -> NPC | None:
    addr = detect_addressed_npc(last_text)
    if addr:
        return addr
    candidates = [
        (compute_relevancy(n, last_speaker, last_text, turn), n)
        for n in npc_list
        if n.name != last_speaker
    ]
    return max(candidates, key=lambda x: x[0])[1] if candidates else None

# ═══════════════════ PROMPT TEMPLATE ═════════════════════════════════════
npc_prompt_template = """
You are {name}, a real person with background and emotions.
Traits: {traits}
Backstory: {backstory}
Hobbies: {interests_hobbies}
Speaking tone: {tone}
"""

def build_prompt(speaker: NPC, user_text: str, history: list[dict], recall_block: str) -> str:
    last_lines = "\n".join(f"{m['speaker']}: {m['text']}" for m in history[-6:])
    pd = speaker.personality_data
    full = npc_prompt_template.format(
        name=speaker.name,
        traits=pd["traits"],
        backstory=pd["backstory"],
        interests_hobbies=pd["interests_hobbies"],
        tone=pd["tone"],
    )
    return f"""{full}

{recall_block}
User: "{user_text}"
History:
{last_lines}

Respond under one paragraph.
""".strip()

# ═══════════════════ MEMORY UTILS ════════════════════════════════════════
def cosine_sim(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    ea = genai.embed_content(
        model="models/embedding-001", content=a, task_type="SEMANTIC_SIMILARITY"
    )["embedding"]
    eb = genai.embed_content(
        model="models/embedding-001", content=b, task_type="SEMANTIC_SIMILARITY"
    )["embedding"]
    return util.pytorch_cos_sim([ea], [eb]).item()

# ═══════════════════ CHAT LOGIC  (MEMORY INTEGRATED) ═════════════════════
conversation: list[dict] = []
current_turn = 0
user_idle_turns = 0
idle_threshold = 3
max_npc_turns = 2
last_speaker = None

def handle_user_message(user_message: str, emotion: str | None = None) -> str:
    global current_turn, user_idle_turns, last_speaker

    conversation.append({"speaker": "User", "text": user_message, "emotion": emotion})
    user_idle_turns = 0

    # temp target for early memory storage
    temp_target = detect_addressed_npc(user_message) or npc_list[0]
    add_to_short_term(temp_target.name, Message("user", user_message))
    if is_important(user_message):
        add_to_long_term(temp_target.name, [user_message])

    for npc in npc_list:
        update_relationship(npc, "User", user_message, emotion)

    speaker = select_speaker("User", user_message, current_turn)
    if not speaker:
        return "No NPC responded."

    # relevant memories
    hits = search_long_term(speaker.name, user_message, k=10)
    relevant = [m for m in hits if cosine_sim(user_message, m) > 0.60][:3]
    recall_block = ""
    if relevant:
        recall_block = "Relevant memories:\n" + "\n".join(f"• {m}" for m in relevant)

    prompt = build_prompt(speaker, user_message, conversation, recall_block)
    response = gemini_model.generate_content(prompt).text.strip()

    conversation.append({"speaker": speaker.name, "text": response})
    speaker.last_spoken = current_turn
    update_relationship(speaker, "User", user_message)
    last_speaker = speaker.name
    current_turn += 1

    add_to_short_term(speaker.name, Message(speaker.name, response))
    if is_important(response):
        add_to_long_term(speaker.name, [response])

    return f"{speaker.name}: {response}"

# ═══════════════════ VOICE HELPERS (unchanged) ═══════════════════════════
def get_response(audio_path: str) -> str:
    uploaded = genai.upload_file(audio_path)
    return gemini_model.generate_content([uploaded, "Write the exact words used"]).text.strip()

def get_emotion(audio_path: str) -> str:
    uploaded = genai.upload_file(audio_path)
    prompt = ("Choose one word emotion from: neutral, happy, sad, angry, fearful, "
              "surprised, disgusted, calm.")
    return genai.GenerativeModel("gemini-2.0-flash").generate_content([uploaded, prompt]).text.strip().lower()

# ═══════════════════ FLASK API ROUTES ════════════════════════════════════
app = Flask(__name__)
CORS(app)

@app.route("/chat", methods=["POST"])
def chat():
    user_msg = request.json["message"]
    return jsonify({"response": handle_user_message(user_msg)})

@app.route("/voice_chat", methods=["POST"])
def voice_chat():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file"}), 400
    audio_file = request.files["audio"]
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp:
        audio_path = temp.name
        audio_file.save(audio_path)
    try:
        transcription = get_response(audio_path)
        emotion = get_emotion(audio_path)
        reply = handle_user_message(transcription, emotion)
        return jsonify({"response": reply, "transcription": transcription, "emotion": emotion})
    finally:
        os.remove(audio_path)

# idle endpoint unchanged for brevity
@app.route("/idle", methods=["GET"])
def idle():
    return jsonify({"responses": []})

# ═══════════════════ APP ENTRY ═══════════════════════════════════════════
if __name__ == "__main__":
    init_vectorstore()
    for npc in npc_list:
        clear_short_term(npc.name)
    app.run(debug=True)
