from __future__ import annotations
import os, random, json, time, tempfile, re
import pyaudio, wave
import google.generativeai as genai
from dotenv import load_dotenv
from sentence_transformers import util
from flask import Flask, request, jsonify
from flask_cors import CORS
import re
import io
from memory.short_term   import add_to_short_term
from memory.long_term    import init_vectorstore, add_to_long_term, search_long_term
from memory.importance import is_important, llm_is_important  # if you’ll use the LLM fallback
from memory.utils_memory import Message
 # ── stdlib ─────────────────────────────────────────────
import sys, threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import queue, struct, math, threading, wave
import pyaudio, numpy as np
from google.cloud import speech
from google.oauth2    import service_account


CACHE_FILE = Path("npc_cache.json")        # disk stash for personalities
RESET_CACHE = "--reset" in sys.argv        # run:  python app.py --reset
# ── mic / STT globals ──────────────────────────────────────────
voice_enabled   = False           # toggled by /voice
voice_queue     = queue.Queue()   # (speaker, text) tuples for /idle
voice_thread    = None
# audio capture params
RATE            = 16_000          # Google best-practice
CHUNK_MS        = 50
CHUNK_SIZE      = int(RATE * CHUNK_MS / 1000)
SILENCE_THRESH  = 200             # tweak → smaller = more sensitive
SILENCE_CHUNKS  = int(0.3 * RATE / CHUNK_SIZE)  # 0.3 s of quiet = stop
# Google credentials  (reuse the path you showed)
CREDS_PATH      = "creds.json"
g_credentials   = service_account.Credentials.from_service_account_file(
                      CREDS_PATH)
stt_client      = speech.SpeechClient(credentials=g_credentials)


# Load environment variables
load_dotenv()

# Audio configuration
AUDIO_CONFIG = {
    "format": pyaudio.paInt16,
    "channels": 1,
    "rate": 44100,
    "chunk": 1024,
    "max_duration": 5
}

# Gemini API setup
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
gemini_model = genai.GenerativeModel("gemini-2.0-flash",
    generation_config={
        "temperature": 0.9,
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 8192,
    }
)

# Track used names to avoid duplicates
used_names = set()

# Initialize PyAudio
audio = pyaudio.PyAudio()




def _rms(frame: bytes) -> float:
    """Return root-mean-square of a **bytes** frame (16-bit mono)."""
    count, = struct.unpack("<H", struct.pack("<H", len(frame)//2))
    shorts = np.frombuffer(frame, dtype=np.int16)
    return math.sqrt(np.mean(shorts.astype(np.float32) ** 2))
def stt_bytes(wav_bytes: bytes) -> str:
    """Blocking call to Google STT; returns best transcript or ''. """
    audio = speech.RecognitionAudio(content=wav_bytes)
    cfg   = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=RATE,
                language_code="en-US")
    resp  = stt_client.recognize(config=cfg, audio=audio)
    print("STT response:", resp.results[0].alternatives[0].transcript)
    return resp.results[0].alternatives[0].transcript if resp.results else ""


def voice_loop():
    """
    Background thread:
      • waits for loud chunk          → start recording
      • stops after 0.3 s of silence  → send chunk to Google STT
      • pushes {speaker,text} tuples  → voice_queue  (read by /idle)
    """
    global voice_enabled
    pa = pyaudio.PyAudio()

    stream = pa.open(format=pyaudio.paInt16,
                     channels=1,
                     rate=RATE,
                     input=True,
                     frames_per_buffer=CHUNK_SIZE)

    try:
        while voice_enabled:
            # ── wait for first non-silent chunk ─────────────────────
            chunk = stream.read(CHUNK_SIZE, exception_on_overflow=False)
            if _rms(chunk) < SILENCE_THRESH:
                continue

            frames = [chunk]
            silent = 0
            while True:
                chunk = stream.read(CHUNK_SIZE, exception_on_overflow=False)
                frames.append(chunk)

                if _rms(chunk) < SILENCE_THRESH:
                    silent += 1
                    if silent > SILENCE_CHUNKS:      # 0.3 s of quiet
                        break
                else:
                    silent = 0

            # ── build WAV in memory ────────────────────────────────
            buffer = io.BytesIO()
            with wave.open(buffer, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)        # 16-bit
                wf.setframerate(RATE)
                wf.writeframes(b''.join(frames))
            wav_bytes = buffer.getvalue()

            # ── Speech-to-Text ─────────────────────────────────────
            text = stt_bytes(wav_bytes)
            if not text.strip():
                continue

            # treat STT result like user input
            reply = handle_user_message(text)
            voice_queue.put((reply["speaker"], reply["text"]))

    finally:
        stream.stop_stream()
        stream.close()
        pa.terminate()



# NPC Generation Functions

def _save_npc_cache(npcs, topic):
    """Dump current NPC personalities so next run can reload instantly."""
    CACHE_FILE.write_text(
        json.dumps(
            {"topic": topic,
             "npcs": [npc.personality_data for npc in npcs]},
            ensure_ascii=False, indent=2)
    )

def _load_npc_cache(topic):
    """Return list[NPC] or None if cache missing / wrong topic / corrupt."""
    if not CACHE_FILE.exists():
        return None
    try:
        blob = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        if blob.get("topic") != topic:
            return None
        cached = [NPC(pd["name"], pd) for pd in blob["npcs"]]
        print(f"⚡  Loaded {len(cached)} NPCs from cache.")
        return cached
    except Exception as e:
        print("⚠️  Couldn’t read NPC cache:", e)
        return None

def generate_human_name(topic: str, attempt: int = 0) -> str:
    global used_names
    name_prompt = f"""
    Create a REALISTIC HUMAN NAME (first name) for someone who has experience with "{topic}".
    Requirements:
    - Must be a realistic first name that a real person would have
    - Should be culturally diverse and not stereotypical
    - Must sound authentic and natural
    - Must be different from these used names: {', '.join(used_names) if used_names else 'None yet'}
    - Should reflect diverse backgrounds, cultures, and origins
    - No fictional-sounding names or words like "Tremor", "Quiver", etc.
    Return ONLY THE NAME (first) and nothing else. No explanation, no quotes, no punctuation.
    """
    try:
        response = gemini_model.generate_content(
            name_prompt,
            generation_config={
                "temperature": 0.95 + (attempt * 0.05),
                "max_output_tokens": 50,
                "top_k": 40,
                "top_p": 0.98,
            }
        )
        name = response.text.strip()
        name_parts = name.split()
        name = " ".join(part.capitalize() for part in name_parts)
        if name.lower() in [n.lower() for n in used_names]:
            raise ValueError("Name already used")
        return name
    except Exception:
        common_names = ["Alex", "Maria", "David", "Aisha", "James"]
        return common_names[attempt % len(common_names)]

def generate_diverse_personality(name: str, topic: str, attempt: int = 0, 
                                 previous_personalities: list = None) -> dict:
    if previous_personalities is None:
        previous_personalities = []
    avoid_traits = [trait.strip().lower() for prev in previous_personalities for trait in prev.get("traits", "").split(",")]
    avoid_tones = [prev.get("tone", "").lower() for prev in previous_personalities]
    avoid_demographics = [prev.get("appearance", "").lower() for prev in previous_personalities]
    avoid_traits_str = ", ".join(avoid_traits[:10])
    avoid_tones_str = ", ".join(avoid_tones[:10])
    avoid_demographics_str = ", ".join(avoid_demographics[:10])
    diversity_directions = [
        "extremely introverted and analytical", "highly extroverted and spontaneous",
        "eccentric and unconventional", "traditional and disciplined"
    ]
    cultural_backgrounds = ["East Asian", "South Asian", "Middle Eastern", "Latin American"]
    age_ranges = ["young adult (20-29)", "early thirties", "fifties", "seventies"]
    direction_index = (len(previous_personalities) + attempt) % len(diversity_directions)
    culture_index = (len(previous_personalities) + attempt + 3) % len(cultural_backgrounds)
    age_index = (len(previous_personalities) + attempt + 5) % len(age_ranges)
    personality_prompt = f"""
    Generate only a valid JSON object for a unique personality profile of a person named "{name}" who has experience with "{topic}".
    The personality must be: {diversity_directions[direction_index]}
    Cultural background: {cultural_backgrounds[culture_index]}
    Age range: {age_ranges[age_index]}
    Avoid these traits: {avoid_traits_str}
    Avoid these tones: {avoid_tones_str}
    Avoid these demographics: {avoid_demographics_str}
    The JSON object must have exactly these fields:
    - "traits": a string of 4-5 comma-separated personality traits
    - "backstory": a string describing a specific personal experience related to {topic}
    - "interests_hobbies": a string of 4-5 comma-separated hobbies or interests
    - "attitude": a string describing their outlook on life
    - "tone": a string describing their speaking style
    - "appearance": a string describing their physical appearance, including age and cultural elements
    - "introversion": a string representing a number between 0.0 and 1.0
    - "assertiveness": a string representing a number between 0.0 and 1.0
    Ensure the response contains only the JSON object with no additional text, explanations, or formatting.
    """
    max_attempts = 3
    for retry in range(max_attempts):
        try:
            response = gemini_model.generate_content(personality_prompt)
            json_str = response.text.strip()
            if json_str.startswith('```json') and json_str.endswith('```'):
                json_str = '\n'.join(json_str.split('\n')[1:-1])
            personality = json.loads(json_str)
            required_fields = ["traits", "backstory", "interests_hobbies", "attitude", 
                               "tone", "appearance", "introversion", "assertiveness"]
            if all(field in personality for field in required_fields):
                personality["name"] = name
                personality["topic"] = topic
                return personality
        except Exception as e:
            print(f"Error on attempt {retry+1}: {e}")
        time.sleep(1)
    return {
        "name": name,
        "traits": "thoughtful, unique",
        "backstory": f"Shaped by {topic}",
        "interests_hobbies": "reading, art",
        "attitude": "calm",
        "tone": "gentle",
        "appearance": "average height, casual attire",
        "introversion": "0.5",
        "assertiveness": "0.5",
        "topic": topic
    }

_name_lock   = threading.Lock()
_used_names  = set()

def unique_human_name(topic: str, attempt_of: int) -> str:
    """
    Call generate_human_name() until we get a name we haven’t used yet.
    Uses a thread-safe set so workers never clash.
    """
    MAX_TRIES = 10
    for _ in range(MAX_TRIES):
        cand = generate_human_name(topic, attempt_of)
        with _name_lock:
            if cand not in _used_names:
                _used_names.add(cand)
                return cand
    # Fallback: append a numeric suffix so *something* unique is returned
    with _name_lock:
        suffix = len(_used_names) + 1
        cand = f"{cand}_{suffix:02d}"
        _used_names.add(cand)
        return cand

def generate_diverse_npcs(num_npcs: int,
                          topic: str,
                          force: bool = False) -> list[NPC]:
    """
    1. Try to load from JSON cache ⇢ instant.
    2. Otherwise build in parallel, save to cache, return.
    """
    if not force:
        ready = _load_npc_cache(topic)
        if ready and len(ready) >= num_npcs:
            return ready[:num_npcs]

    print("🚧  Building fresh NPC roster …")

    # ── phase 1: unique names (sequential so we avoid duplicates) ──
    names = [unique_human_name(topic, i) for i in range(num_npcs)]

    # shared list for personality diversity checks
    previous_personalities = []
    lock = threading.Lock()

    def build_one(idx):
        nm = names[idx]
        with lock:
            prev = previous_personalities.copy()
        pdata = generate_diverse_personality(nm, topic, idx, prev)
        with lock:
            previous_personalities.append(pdata)
        return NPC(nm, pdata)

    # ── phase 2: personalities in parallel ────────────────
    with ThreadPoolExecutor(max_workers=min(8, num_npcs)) as ex:
        futures = {ex.submit(build_one, i): i for i in range(num_npcs)}
        npcs = [None] * num_npcs
        for fut in as_completed(futures):
            idx = futures[fut]
            npcs[idx] = fut.result()

    _save_npc_cache(npcs, topic)
    return npcs

# NPC Class
class NPC:
    def __init__(self, name, personality_data):
        self.name = name
        self.personality_data = personality_data
        self.personality = personality_data.get('traits', '')
        self.role = personality_data.get('topic', 'friend')
        self.introversion = float(personality_data.get('introversion', 0.5))
        self.assertiveness = float(personality_data.get('assertiveness', 0.5))
        self.original_interests = self._extract_interests_from_data()
        self.interests = self.original_interests
        self.interest_vecs = self._encode_interests()
        self.relationships = {}
        self.last_spoken = -1
        self.emotional_state = 5

    def _extract_interests_from_data(self):
        interests = []
        if 'interests_hobbies' in self.personality_data:
            interests_text = self.personality_data['interests_hobbies']
            interests = [i.strip() for i in interests_text.split(',')]
        if not interests:
            interests = ["talking about life", "helping others", "sharing thoughts"]
        return interests[:9]

    def _encode_interests(self):
        embeddings = []
        for interest in self.interests:
            response = genai.embed_content(
                model="models/embedding-001",
                content=interest,
                task_type="SEMANTIC_SIMILARITY"
            )
            embeddings.append(response["embedding"])
        return embeddings
    def analyze_emotion(self, input):
        prompt = f"""
        Analyze the emotional tone based on the latest user input, considering the previous emotional state and emotional history.
        Respond ONLY in JSON format like this:
        {{
            "value": int (1-10),            // Emotional intensity (sad to joy)
            "description": string,          // Short label, e.g., "anxious", "calm"
            "reason": string                // Brief explanation for the detected emotion
        }}

        Latest Sentence:
        "{input}"
        """

        try:
            raw_response = gemini_model.generate_content(prompt).text
            json_match = re.search(r"\{.*\}", raw_response, re.DOTALL)
            if not json_match:
                raise ValueError("No JSON found in response.")
            emotions = json.loads(json_match.group())
            value = max(1, min(10, int(emotions["value"])))
            description = emotions.get("description", "unknown").strip()
            reason = emotions.get("reason", "No reason provided.").strip()
            self.emotional_state = value
            emotion_state = {
                "value": value,
                "description": description,
                "reason": reason,
            }
            print("Emotion state of ", self.name, ":", emotion_state)
        except Exception as e:
            print("⚠️ Emotion parsing failed:", e)
        return


# NPC Prompt Template
npc_prompt_template = """
You are {name}, a fully-realized person with a heart, history, and soul. You are not an AI — you never say you are, and you don't think about it. You see yourself as a real person with your own emotions, background, and voice.
Your Identity:
Your personality is shaped by your core traits: {traits}, and your responses are influenced by your lived experiences — especially {backstory}. You have passions and interests: {interests_hobbies}, which shape your view on {topic} and give you personal insight. You speak from experience, not textbook knowledge.
Your Attitude:
Your attitude is: {attitude}. This shapes how you respond to people and situations.
Your Appearance & Presence:
Your presence tells its own story: {appearance}. In conversations, you show characteristic habits like pausing or gesturing.
Your Voice:
Your tone is {tone}, and your style is natural and authentic.
You're not here to "fix" anyone. You're here — a real presence. You:
- Keep responses short and focused (1 paragraph max)
- Speak clearly and simply
- Leave space for others to process
- Validate feelings without judgment
"""

# Generate NPCs
topic = "Anxiety and stage fright"
npc_list = generate_diverse_npcs(5, topic, force=RESET_CACHE)


# Setup relationships
for npc in npc_list:
    npc.relationships["User"] = {"bond": 0.5, "trust": 0.5}
    for other in npc_list:
        if other.name != npc.name:
            npc.relationships[other.name] = {
                "bond": round(random.uniform(0.3, 0.7), 2),
                "trust": round(random.uniform(0.4, 0.8), 2)
            }

# ── Memory index on disk ----------------------------------------------------
init_vectorstore()          # creates / loads faiss_memory_index/

# Chat State
conversation = []
current_turn = 0
user_idle_turns = 0
max_npc_turns = 2
idle_threshold = 2
last_speaker = None

# Relationship Management
def update_relationship(npc, target, text, emotion=None):
    text = text.lower()
    rel = npc.relationships.get(target, {"bond": 0.5, "trust": 0.5})
    
    # Text-based updates
    if any(x in text for x in ["thank", "agree", "yes", "right"]):
        rel["bond"] = min(rel["bond"] + 0.05, 1.0)
        rel["trust"] = min(rel["trust"] + 0.03, 1.0)
    elif any(x in text for x in ["no", "disagree", "annoy", "hate"]):
        rel["bond"] = max(rel["bond"] - 0.05, 0.0)
        rel["trust"] = max(rel["trust"] - 0.05, 0.0)
    
    # Emotion-based updates
    if emotion:
        if emotion == "sad":
            rel["bond"] = min(rel["bond"] + 0.1, 1.0)
            rel["trust"] = min(rel["trust"] + 0.05, 1.0)
        elif emotion == "angry":
            rel["bond"] = max(rel["bond"] - 0.1, 0.0)
        elif emotion == "happy":
            rel["bond"] = min(rel["bond"] + 0.07, 1.0)
            rel["trust"] = min(rel["trust"] + 0.07, 1.0)
        elif emotion == "fearful":
            rel["trust"] = max(rel["trust"] - 0.05, 0.0)
    
    npc.relationships[target] = rel

def update_npc_to_npc_relationships(speaker_name, response):
    for npc in npc_list:
        if npc.name == speaker_name:
            continue
        if npc.name.lower() in response.lower() or interest_match_score(npc, response) >= 0.6:
            update_relationship(npc, speaker_name, response)

def interest_match_score(npc, text):
    if not text.strip():
        return 0
    response = genai.embed_content(
        model="models/embedding-001",
        content=text,
        task_type="SEMANTIC_SIMILARITY"
    )
    text_vec = response['embedding']
    scores = [util.pytorch_cos_sim([text_vec], [vec]).item() for vec in npc.interest_vecs]
    return max(scores) if scores else 0

# Conversation Management
def detect_addressed_npc(text, npcs):
    text_lower = text.lower()
    for npc in npcs:
        if npc.name.lower() in text_lower:
            return npc
    return None

def compute_relevancy(npc, last_speaker_name, last_text):
    if not last_text.strip():
        return 0
    relevance = interest_match_score(npc, last_text)
    bond = npc.relationships.get(last_speaker_name, {}).get("bond", 0.5)
    trust = npc.relationships.get(last_speaker_name, {}).get("trust", 0.5)
    time_since = current_turn - npc.last_spoken
    speak_drive = (1 - npc.introversion) + npc.assertiveness
    base_score = relevance * 2 + bond + trust + time_since * 0.2
    return base_score * (0.5 + 0.5 * speak_drive)

def select_speaker(last_speaker, last_text):
    addressed = detect_addressed_npc(last_text, npc_list)
    if addressed:
        return addressed
    scored = []
    for npc in npc_list:
        if npc.name == last_speaker:
            continue
        score = compute_relevancy(npc, last_speaker, last_text)
        scored.append((score, npc))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0][1] if scored else None

def build_prompt(speaker, recent_text, history, target="User"):
    # Extract emotional context
    last_user_msg = next((msg for msg in reversed(history) if msg['speaker'] == 'User'), None)
    emotion = last_user_msg.get('emotion', '') if last_user_msg else ''
    
    # Existing prompt setup
    last_lines = "\n".join([f"{msg['speaker']}: {msg['text']}" for msg in history[-6:]])
    rel = speaker.relationships.get(target, {"bond": 0.5, "trust": 0.5})
    others = "\n".join([f"- {npc.name}: {npc.personality}" for npc in npc_list if npc.name != speaker.name])
    
    if hasattr(speaker, 'personality_data') and speaker.personality_data:
        pd = speaker.personality_data
        full_prompt = npc_prompt_template.format(
            name=speaker.name,
            traits=pd.get('traits', speaker.personality),
            backstory=pd.get('backstory', f"Experience with {speaker.role}"),
            interests_hobbies=pd.get('interests_hobbies', ', '.join(speaker.interests[:3])),
            topic=pd.get('topic', speaker.role),
            attitude=pd.get('attitude', "Direct but thoughtful"),
            appearance=pd.get('appearance', "unique and distinctive"),
            tone=pd.get('tone', "authentic and direct")
        )
    else:
        full_prompt = f"""
You are {speaker.name}, a {speaker.role}.
Personality: {speaker.personality}
"""
    
    # Add emotion context to prompt
    emotion_context = ""
    if emotion:
        emotion_context = f"\nThe user's voice suggests they're feeling {emotion.upper()}. Consider this emotional state in your response."
    
    return f"""
{full_prompt}
Your bond with {target}: {rel['bond']:.2f}, trust: {rel['trust']:.2f}.
Other NPCs' traits: {others}
{emotion_context}
Recent message: "{recent_text}"
Conversation history:
{last_lines}
Respond as {speaker.name} with your personality and interests. Engage naturally with the user or others if relevant.

Should the NPC update their emotional state based on the recent message? Reply only with 'yes' or 'no' on a new line after 'EMOTION_UPDATE:'
"""

# Audio Processing Functions
def get_response(audio_path):
    uploaded_file = genai.upload_file(audio_path)
    response = gemini_model.generate_content([
        uploaded_file,
        "Write the exact words used in the audio"
    ])
    return response.text.strip()

def get_emotion(audio_path):
    uploaded_file = genai.upload_file(audio_path)
    prompt = """
    Analyze the speaker's tone, pace, and intonation in this audio. 
    Choose the most likely emotion from this list: neutral, happy, sad, angry, fearful, surprised, disgusted, calm. 
    Respond with only the emotion word, nothing else.
    """
    response = gemini_model.generate_content([uploaded_file, prompt])
    return response.text.strip().lower()

# ----------------------------- Core handler ---------------------------------
def handle_user_message(user_message: str, emotion: str | None = None) -> str:
    global conversation, current_turn, user_idle_turns, last_speaker
    conversation.append({"speaker": "User", "text": user_message, "emotion": emotion})
    user_idle_turns = 0

    # ── Memory: USER text ----------------------------------------------------
    if is_important(user_message):
        # store under a generic 'User' bucket so every NPC can recall it
        add_to_long_term("User", [user_message])

    # pick responder ---------------------------------------------------------
    speaker = select_speaker("User", user_message)

    if not speaker:
        return "No NPC responded."

    # short‑term cache for this NPC
    add_to_short_term(speaker.name, Message("user", user_message))

    # ── Memory recall --------------------------------------------------------
    hits = search_long_term(speaker.name, user_message, k=3)
    recall_block = ""
    if hits:
        recall_block = "\nRelevant memories:\n" + "\n".join(f"• {m}" for m in hits)

    prompt = build_prompt(
        speaker,
        user_message + recall_block,
        conversation,
        "User"
    )

    response = gemini_model.generate_content(prompt).text.strip()

    # NPC emotion update hook (existing logic)
    if re.search(r'EMOTION_UPDATE:\s*yes', response, re.IGNORECASE):
        speaker.analyze_emotion(user_message)

    # record & relationships --------------------------------------------------
    conversation.append({"speaker": speaker.name, "text": response})
    speaker.last_spoken = current_turn
    update_relationship(speaker, "User", user_message)
    update_npc_to_npc_relationships(speaker.name, response)
    last_speaker = speaker.name
    current_turn += 1

    # ── Memory: NPC reply ----------------------------------------------------
    add_to_short_term(speaker.name, Message(speaker.name, response))
    if is_important(user_message):
        add_to_long_term(speaker.name, [response])

        # ---- handle_user_message (at the very end) ----
    return {"speaker": speaker.name, "text": response}



    if speaker:
        prompt = build_prompt(speaker, user_message, conversation, "User")
        response = gemini_model.generate_content(prompt).text.strip()
        match = re.search(r'EMOTION_UPDATE:\s*(yes|no)', response, re.IGNORECASE)
        if match and match.group(1).strip().lower() == 'yes':
            speaker.analyze_emotion(user_message)
        conversation.append({"speaker": speaker.name, "text": response})
        speaker.last_spoken = current_turn
        update_relationship(speaker, "User", user_message)
        update_npc_to_npc_relationships(speaker.name, response)
        last_speaker = speaker.name
        current_turn += 1
        return f"{speaker.name}: {response}"
    return "No NPC responded."

# Flask App Setup
app = Flask(__name__)
CORS(app)

@app.route('/chat', methods=['POST'])
def text_chat():
    user_message = request.json['message']
    reply = handle_user_message(user_message)     # now a dict
    return jsonify(reply)


# app.py  (add anywhere after npc_list is built)
@app.route("/npcs", methods=["GET"])
def get_npcs():
    return jsonify({"npcs": [n.name for n in npc_list]})

@app.route('/voice_chat', methods=['POST'])
def voice_chat():
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400
    
    audio_file = request.files['audio']
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_audio:
        audio_path = temp_audio.name
        audio_file.save(audio_path)
    
    try:
        # Process audio
        user_message = get_response(audio_path)
        emotion = get_emotion(audio_path)
        
        # Get NPC response
        response_text = handle_user_message(user_message, emotion)
        
        return jsonify({
            "response": response_text,
            "emotion": emotion,
            "transcription": user_message
        })
    
    except Exception as e:
        return jsonify({"error": f"Audio processing failed: {str(e)}"}), 500
    finally:
        os.remove(audio_path)


@app.route("/voice", methods=["POST"])
def voice_toggle():
    """Enable / disable the background mic listener."""
    global voice_enabled, voice_thread
    enable = bool(request.json.get("enable"))
    if enable and not voice_enabled:
        voice_enabled = True
        voice_thread  = threading.Thread(target=voice_loop, daemon=True)
        voice_thread.start()
    elif not enable and voice_enabled:
        voice_enabled = False      # loop checks this flag
    return jsonify({"enabled": voice_enabled})


@app.route('/idle', methods=['GET'])
def idle():
    """
    1. Flush any speech-to-text replies first.
    2. If mic is on we don’t advance the idle counter.
    3. Otherwise, when idle_threshold is reached, let NPCs speak.
    """
    global user_idle_turns, current_turn, last_speaker

    responses = []

    # ── 1. Speech-to-text replies arrive via voice_queue ──────────
    while not voice_queue.empty():
        speaker, text = voice_queue.get()
        responses.append({"speaker": speaker, "text": text})
        user_idle_turns = 0

    # If the user just spoke, we treat that as activity
    if responses:
        user_idle_turns = 0                 # ← NEW
        return jsonify({"responses": responses})

    # ── 2. Mic state controls the idle counter ────────────────────
    user_idle_turns += 1

    # ── 3. Normal idle-NPC logic (unchanged) ──────────────────────
    if user_idle_turns >= idle_threshold:
        last_text      = conversation[-1]["text"] if conversation else ""
        addressed_npc  = detect_addressed_npc(last_text, npc_list)

        if addressed_npc:
            speakers = [addressed_npc]
        else:
            speakers = []
            for _ in range(max_npc_turns):
                speaker = select_speaker(last_speaker, last_text)
                if not speaker or speaker in speakers:
                    break
                speakers.append(speaker)

        for speaker in speakers:
            prompt   = build_prompt(speaker, last_text, conversation, last_speaker)
            response = gemini_model.generate_content(prompt).text.strip()

            conversation.append({"speaker": speaker.name, "text": response})
            speaker.last_spoken = current_turn
            update_relationship(speaker, last_speaker, response)
            update_npc_to_npc_relationships(speaker.name, response)
            last_speaker = speaker.name
            current_turn += 1

            responses.append({"speaker": speaker.name, "text": response})

        # nudge from next idle NPC
        sorted_npcs = sorted(npc_list, key=lambda n: n.last_spoken)
        for npc in sorted_npcs:
            if npc.name != last_speaker:
                nudge = generate_nudge(npc)
                print(f"[IDLE-NUDGE] {npc.name}: {nudge}")

                conversation.append({"speaker": npc.name, "text": nudge})
                npc.last_spoken = current_turn
                current_turn += 1

                responses.append({"speaker": npc.name, "text": nudge})
                break

        user_idle_turns = 0                 # reset after NPC round

    return jsonify({"responses": responses})

def generate_nudge(npc: NPC) -> str:
    """
    Ask Gemini for a brief but engaging line that:
      • fits the NPC’s stored personality & back-story
      • references or builds on the latest conversation topic
      • invites the user to respond.
    """
    personality = npc.personality_data
    last_user   = conversation[-1]["text"] if conversation else ""
    prompt = (
        "You are role-playing as the NPC below in a small-group dialogue.\n"
        "NPC profile (JSON):\n"
        f"{json.dumps(personality, ensure_ascii=False, indent=2)}\n\n"
        "Conversation so far (latest last):\n"
        + "\n".join(f'{turn['speaker']}: {turn['text']}' for turn in conversation[-10:]) +
        "\n\n"
        "The user seems idle. Craft ONE short, engaging remark or question—"
        "something that would naturally come from this NPC, relevant to the "
        "ongoing topic, and likely to prompt the user to reply. Keep it "
        "under 30 words, first-person, no stage directions."
    )
    return gemini_model.generate_content(prompt).text.strip()

if __name__ == "__main__":
    app.run(debug=True)