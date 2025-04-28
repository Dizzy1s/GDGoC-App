import os
import random
import google.generativeai as genai
from dotenv import load_dotenv
from sentence_transformers import util
import threading
import re
import json

load_dotenv()

# === Gemini setup ===
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
gemini_model = genai.GenerativeModel("gemini-2.0-flash")
embedding_model = genai.GenerativeModel("gemini-embedding-1.0")

initial_state = {
    "input": "",
    "conversation_history": [],
    "emotional_state": {
        "value": 5,
        "description": "neutral",
        "reason": "default initial state",
    },
    "emotional_history": [
        {"value": 5, "description": "neutral", "reason": "default initial state"}
    ],
    "response": "",
}


# === NPC Class ===
class NPC:
    def __init__(self, name, personality, role, introversion, assertiveness, interests):
        self.name = name
        self.personality = personality
        self.role = role
        self.introversion = introversion  # 0 = extrovert, 1 = introvert
        self.assertiveness = assertiveness  # 0 to 1
        self.original_interests = interests
        self.interests = interests
        self.interest_vecs = self._encode_interests()
        self.relationships = {}
        self.last_spoken = -1
        self.state = initial_state.copy()

    def _encode_interests(self):
        embeddings = []
        for interest in self.interests:
            response = genai.embed_content(
                model="models/embedding-001",
                content=interest,
                task_type="SEMANTIC_SIMILARITY",
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
        
        Previous Emotional State:
        {json.dumps(self.state['emotional_state'], indent=2)}

        Emotional History:
        {json.dumps(self.state['emotional_history'], indent=2)}

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

            emotion_state = {
                "value": value,
                "description": description,
                "reason": reason,
            }
            print("Emotion state of ", self.name, ":", emotion_state)
            self.state["emotional_state"] = emotion_state
            self.state["emotional_history"].append(emotion_state)
        except Exception as e:
            print("⚠️ Emotion parsing failed:", e)
        return


# === Initialize NPCs with Interests ===
npc_list = [
    NPC(
        "Mina",
        "funny and sarcastic, loves books and memes",
        "bookworm friend",
        introversion=0.7,
        assertiveness=0.3,
        interests=[
            "reading fantasy novels",
            "sharing dark humor memes",
            "spending time in quiet libraries",
            "discussing plot twists",
            "making sarcastic jokes",
            "analyzing fictional characters",
            "writing fan fiction",
            "introvert humor",
            "late-night online forums",
        ],
    ),
    NPC(
        "Dr. Hale",
        "calm, analytical, thoughtful",
        "mental health expert",
        introversion=0.6,
        assertiveness=0.6,
        interests=[
            "mental health awareness",
            "talking about therapy techniques",
            "understanding cognitive behavior",
            "practicing mindfulness",
            "writing reflection journals",
            "emotional regulation strategies",
            "discussing neuroscience",
            "CBT frameworks",
            "building mental resilience",
        ],
    ),
    NPC(
        "Leo",
        "energetic, impulsive, supportive",
        "chatty extrovert",
        introversion=0.1,
        assertiveness=0.9,
        interests=[
            "motivating friends",
            "leading group games",
            "planning spontaneous parties",
            "giving pep talks",
            "sharing funny stories",
            "breaking awkward silence",
            "hyping people up",
            "team bonding events",
            "talking to new people",
        ],
    ),
    NPC(
        "Raya",
        "empathetic and introspective, passionate about self-expression and emotional healing through art",
        "art therapist",
        introversion=0.8,
        assertiveness=0.4,
        interests=[
            "emotional expression through painting",
            "healing with creative writing",
            "talking about trauma recovery",
            "exploring emotional intelligence",
            "art therapy sessions",
            "creating safe spaces",
            "guided self-reflection",
            "journaling emotions",
            "helping others process feelings",
        ],
    ),
    NPC(
        "Kai",
        "curious and outspoken, loves analyzing media and pop culture with a critical lens",
        "media critic",
        introversion=0.2,
        assertiveness=0.85,
        interests=[
            "debating media representation",
            "reviewing pop culture trends",
            "discussing satire in film",
            "critiquing celebrity behavior",
            "analyzing YouTube essays",
            "talking about TV tropes",
            "breaking down social commentary",
            "exploring internet culture",
            "calling out clichés in shows",
        ],
    ),
]


# === Setup Relationships ===
for npc in npc_list:
    npc.relationships["User"] = {"bond": 0.5, "trust": 0.5}
    for other in npc_list:
        if other.name != npc.name:
            npc.relationships[other.name] = {
                "bond": round(random.uniform(0.3, 0.7), 2),
                "trust": round(random.uniform(0.4, 0.8), 2),
            }

# === Chat State ===
conversation = []
current_turn = 0
user_idle_turns = 0
max_npc_turns = 2
idle_threshold = 3


# === Relationship Updater ===
def update_relationship(npc, target, text):
    text = text.lower()
    rel = npc.relationships.get(target, {"bond": 0.5, "trust": 0.5})
    if any(x in text for x in ["thank", "agree", "yes", "right"]):
        rel["bond"] = min(rel["bond"] + 0.05, 1.0)
        rel["trust"] = min(rel["trust"] + 0.03, 1.0)
    elif any(x in text for x in ["no", "disagree", "annoy", "hate"]):
        rel["bond"] = max(rel["bond"] - 0.05, 0.0)
        rel["trust"] = max(rel["trust"] - 0.05, 0.0)
    npc.relationships[target] = rel


# === Smarter NPC-to-NPC Relationship Updates ===
def update_npc_to_npc_relationships(speaker_name, response):
    for npc in npc_list:
        if npc.name == speaker_name:
            continue
        if (
            npc.name.lower() in response.lower()
            or interest_match_score(npc, response) >= 0.6
        ):
            update_relationship(npc, speaker_name, response)


# === Embedding-based Interest Scorer ===
def interest_match_score(npc, text):
    if not text.strip():
        return 0
    response = genai.embed_content(
        model="models/embedding-001", content=text, task_type="SEMANTIC_SIMILARITY"
    )
    text_vec = response["embedding"]
    scores = [util.pytorch_cos_sim(text_vec, vec).item() for vec in npc.interest_vecs]
    return max(scores) if scores else 0


# === Direct Address Detection ===
def detect_addressed_npc(text, npcs):
    text_lower = text.lower()
    for npc in npcs:
        if npc.name.lower() in text_lower:
            return npc
    return None


# === Compute Relevancy Score ===
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


# === Select Speaker ===
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


# === Prompt Builder ===
def build_prompt(speaker, recent_text, history, target="User"):
    last_lines = "\n".join([f"{msg['speaker']}: {msg['text']}" for msg in history[-6:]])
    emotional_state = speaker.state["emotional_state"]
    emotional_history = speaker.state["emotional_history"][
        -5:
    ]  # Last 5 emotional states

    # Relationship details used implicitly in NPC behavior, not mentioned in the response
    rel = speaker.relationships.get(target, {"bond": 0.5, "trust": 0.5})
    others = "\n".join(
        [
            f"- {npc.name}: {npc.personality} (Bond: {speaker.relationships.get(npc.name, {}).get('bond', 0.5):.2f}, Trust: {speaker.relationships.get(npc.name, {}).get('trust', 0.5):.2f})"
            for npc in npc_list
            if npc.name != speaker.name
        ]
    )

    # Bond and trust guide used internally to shape NPC behavior
    bond_trust_guide = """
    - If bond and trust are low: NPCs will be more distant, brief in responses, or disengaged.
    - If bond and trust are moderate: NPCs will be neutral, somewhat open but not deeply personal.
    - If bond and trust are high: NPCs will engage warmly, with empathy and openness.
    """

    # Emotional guide (new)
    emotional_guide = f"""
    Current Emotional State:
    - Value: {emotional_state['value']} / 10
    - Description: {emotional_state['description']}
    - Reason: {emotional_state['reason']}

    Recent Emotional History (last 5 entries):
    {json.dumps(emotional_history, indent=2)}

    Guidelines:
    - If emotions are positive (e.g., happy, excited), the NPC should be more enthusiastic and supportive.
    - If emotions are negative (e.g., sad, anxious), the NPC should be more gentle, empathetic, and patient.
    - Take into account how recent emotional trends have evolved when crafting your response tone.
    """

    return f"""
You are {speaker.name}, a {speaker.role}.
Personality: {speaker.personality}
Introversion: {speaker.introversion}, Assertiveness: {speaker.assertiveness}

Your bond with {target}: {rel['bond']:.2f}, trust: {rel['trust']:.2f}.
Other NPCs' traits (these will guide your responses, but do not need to be referenced directly):
{others}

Here is a guide for how you should interact based on bond and trust with others:
{bond_trust_guide}

Here is your emotional context based on your current and recent emotional states:
{emotional_guide}

Recent message: \"{recent_text}\" (Consider tone and sentiment for your response)

Conversation history:
{last_lines}

Respond as {speaker.name} with your personality, emotions, and interests. Engage naturally with the user, adapting your tone based on the emotional and relational context. 
Do NOT explicitly mention bond, trust, or emotional value numbers; simply let them shape your tone and style.
"""


# === Nudge User ===
def generate_nudge(npc):
    options = [
        "What do you think about that?",
        "We'd love to hear your perspective.",
        "You've been quiet — everything okay?",
        "Any thoughts on this?",
        "Jump in when you're ready!",
    ]
    return f"{npc.name}: {random.choice(options)}"


def update_npc(npc, user_input):
    update_relationship(npc, "User", user_input)
    npc.analyze_emotion(user_input)


# === Main Loop ===
def chat_loop():
    global current_turn, user_idle_turns

    print("Chat started. Type 'exit' to stop.\n")
    last_speaker = None

    while True:
        if user_idle_turns < idle_threshold:
            user_input = input("You: ")
            if user_input.strip().lower() == "exit":
                break
            user_idle_turns = 0
            conversation.append({"speaker": "User", "text": user_input})
            last_speaker = "User"

            threads = []
            for npc in npc_list:
                thread = threading.Thread(target=update_npc, args=(npc, user_input))
                thread.start()
                threads.append(thread)

            # Wait for all threads to finish
            for thread in threads:
                thread.join()

            speaker = select_speaker("User", user_input)
            if speaker:
                prompt = build_prompt(speaker, user_input, conversation, "User")
                response = gemini_model.generate_content(prompt).text.strip()
                print(f"{speaker.name}: {response}")
                conversation.append({"speaker": speaker.name, "text": response})
                speaker.last_spoken = current_turn
                update_relationship(speaker, "User", user_input)
                update_npc_to_npc_relationships(speaker.name, response)
                last_speaker = speaker.name
        else:
            last_text = conversation[-1]["text"]
            addressed_npc = detect_addressed_npc(last_text, npc_list)
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
                prompt = build_prompt(speaker, last_text, conversation, last_speaker)
                response = gemini_model.generate_content(prompt).text.strip()
                print(f"{speaker.name}: {response}")
                conversation.append({"speaker": speaker.name, "text": response})
                speaker.last_spoken = current_turn
                update_relationship(speaker, last_speaker, response)
                update_npc_to_npc_relationships(speaker.name, response)
                last_speaker = speaker.name
                current_turn += 1

            if user_idle_turns >= idle_threshold:
                sorted_npcs = sorted(npc_list, key=lambda n: n.last_spoken)
                for npc in sorted_npcs:
                    if npc.name != last_speaker:
                        nudge = generate_nudge(npc)
                        print(nudge)
                        conversation.append({"speaker": npc.name, "text": nudge})
                        npc.last_spoken = current_turn
                        current_turn += 1
                        break

        current_turn += 1
        user_idle_turns += 1


# === Start Chat ===
if __name__ == "__main__":
    chat_loop()
