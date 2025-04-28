# wasay.py  – Memory-augmented multi-NPC chat
import os, random, google.generativeai as genai
from dotenv import load_dotenv
from sentence_transformers import util

# ----- Memory imports ------------------------------------------------------
from short_term   import add_to_short_term, clear_short_term
from long_term    import init_vectorstore, add_to_long_term, search_long_term
from importance import is_important, llm_is_important
from utils_memory import Message
# ---------------------------------------------------------------------------

load_dotenv()

# === Gemini setup ===
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
gemini_model    = genai.GenerativeModel("gemini-2.0-flash")
embedding_model = genai.GenerativeModel("gemini-embedding-1.0")

# === NPC Class =============================================================
class NPC:
    def __init__(self, name, personality, role, introversion, assertiveness, interests):
        self.name = name
        self.personality  = personality
        self.role         = role
        self.introversion = introversion   # 0 extro- 1 intro-
        self.assertiveness= assertiveness  # 0–1
        self.interests    = interests
        self.interest_vecs= self._encode_interests()
        self.relationships= {}
        self.last_spoken  = -1

    def _encode_interests(self):
        vecs=[]
        for interest in self.interests:
            vecs.append(
                genai.embed_content(model="models/embedding-001",
                                    content=interest,
                                    task_type="SEMANTIC_SIMILARITY")["embedding"]
            )
        return vecs
# ---------------------------------------------------------------------------

npc_list = [
    NPC("Mina","funny and sarcastic, loves books and memes","bookworm friend",0.7,0.3,
        ["reading fantasy novels","sharing dark humor memes","quiet libraries","plot twists","making sarcastic jokes",
         "analyzing fictional characters","writing fan fiction","introvert humor","late-night forums"]),
    NPC("Dr. Hale","calm, analytical, thoughtful","mental health expert",0.6,0.6,
        ["mental health awareness","therapy techniques","cognitive behavior","mindfulness","reflection journals",
         "emotional regulation","neuroscience","CBT frameworks","building resilience"]),
    NPC("Leo","energetic, impulsive, supportive","chatty extrovert",0.1,0.9,
        ["motivating friends","leading group games","spontaneous parties","pep talks","funny stories",
         "breaking awkward silence","hyping people up","team bonding","meeting new people"]),
    NPC("Raya","empathetic, introspective art therapist","art therapist",0.8,0.4,
        ["emotional expression through painting","healing with writing","trauma recovery","emotional intelligence",
         "art therapy","safe spaces","self-reflection","journaling emotions","helping others process feelings"]),
    NPC("Kai","curious media critic with sharp tongue","media critic",0.2,0.85,
        ["media representation","pop-culture trends","satire in film","celebrity behavior","YouTube essays",
         "TV tropes","social commentary","internet culture","calling out clichés"]),
]

# --- init relationships ----------------------------------------------------
for npc in npc_list:
    npc.relationships["User"] = {"bond":0.5,"trust":0.5}
    for other in npc_list:
        if other.name!=npc.name:
            npc.relationships[other.name] = {
                "bond": round(random.uniform(0.3,0.7),2),
                "trust":round(random.uniform(0.4,0.8),2)
            }

# === Global chat state =====================================================
conversation, current_turn, user_idle_turns = [],0,0
max_npc_turns, idle_threshold = 2,3

# === Relationship helpers --------------------------------------------------
def update_relationship(npc, target, text):
    tl=text.lower()
    rel=npc.relationships.get(target,{"bond":0.5,"trust":0.5})
    if any(x in tl for x in ["thank","agree","yes","right"]):
        rel["bond"]=min(rel["bond"]+0.05,1); rel["trust"]=min(rel["trust"]+0.03,1)
    elif any(x in tl for x in ["no","disagree","annoy","hate"]):
        rel["bond"]=max(rel["bond"]-0.05,0); rel["trust"]=max(rel["trust"]-0.05,0)
    npc.relationships[target]=rel

def update_npc_to_npc_relationships(speaker_name,response):
    for npc in npc_list:
        if npc.name==speaker_name: continue
        if npc.name.lower() in response.lower() or interest_match_score(npc,response)>=0.6:
            update_relationship(npc,speaker_name,response)

# === Similarity helpers ----------------------------------------------------
def interest_match_score(npc,text):
    if not text.strip(): return 0
    vec=genai.embed_content(model="models/embedding-001",content=text,task_type="SEMANTIC_SIMILARITY")["embedding"]
    return max(util.pytorch_cos_sim(vec,v).item() for v in npc.interest_vecs)

def detect_addressed_npc(text,npcs):
    tl=text.lower()
    for n in npcs:
        if n.name.lower() in tl: return n
    return None

# === Speaker selection -----------------------------------------------------
def compute_relevancy(npc,last_spk,last_text):
    if not last_text.strip(): return 0
    relevance=interest_match_score(npc,last_text)
    rel=npc.relationships.get(last_spk,{"bond":0.5,"trust":0.5})
    time_since=current_turn-npc.last_spoken
    drive=(1-npc.introversion)+npc.assertiveness
    return (relevance*2+rel["bond"]+rel["trust"]+time_since*0.2)*(0.5+0.5*drive)

def select_speaker(last_spk,last_txt):
    addressed=detect_addressed_npc(last_txt,npc_list)
    if addressed: return addressed
    candidates=[(compute_relevancy(n,last_spk,last_txt),n) for n in npc_list if n.name!=last_spk]
    return max(candidates,key=lambda x:x[0])[1] if candidates else None

# === Memory importance rules ----------------------------------------------

def _should_save_npc_reply(text:str)->bool:
    tl=text.lower()
    keywords=["decided","plan","goal","remember","important","proud","worried","feel","progress","family","gym","diet"]
    return any(k in tl for k in keywords) and len(text)>50


# === Prompt builder (Wasay style, unchanged) -------------------------------
def build_prompt(speaker,recent_text,history,target="User"):
    last_lines="\n".join(f"{m['speaker']}: {m['text']}" for m in history[-6:])
    rel=speaker.relationships.get(target,{"bond":0.5,"trust":0.5})
    others="\n".join(
        f"- {n.name}: {speaker.relationships.get(n.name,{}).get('bond',0.5):.2f} bond, "
        f"{speaker.relationships.get(n.name,{}).get('trust',0.5):.2f} trust"
        for n in npc_list if n.name!=speaker.name
    )
    return f"""
You are {speaker.name}, a {speaker.role}.
Personality: {speaker.personality}
Introversion: {speaker.introversion}, Assertiveness: {speaker.assertiveness}

Bond with {target}: {rel['bond']:.2f}, Trust: {rel['trust']:.2f}
Other NPCs:
{others}

User said: "{recent_text}"
Recent conversation:
{last_lines}

Respond naturally as {speaker.name}. Do not reveal bond/trust numbers.
""".strip()

# === Nudge ---------------------------------------------------------------
def generate_nudge(npc):
    return f"{npc.name}: {random.choice(['What do you think?','Any thoughts?','Everything okay?','Jump in when ready!'])}"

# === Chat Loop =============================================================
def chat_loop():
    global current_turn, user_idle_turns

    print("Chat started. Type 'exit' to stop.\n")
    last_speaker = None

    show_memory_triggers = [
        "show memory", "show memories", "what do you remember", "recall",
        "tell me memories", "remind me", "memory lane", "long-term memories",
        "what did we talk about"
    ]
    save_memory_triggers = [
        "remember this", "save this", "this is important",
        "please remember", "note this", "keep this", "remember that"
    ]

    # ── helper for similarity threshold ────────────────────────────────────
    def _similar(a: str, b: str) -> float:
        va = genai.embed_content(model="models/embedding-001",
                                 content=a,
                                 task_type="SEMANTIC_SIMILARITY")["embedding"]
        vb = genai.embed_content(model="models/embedding-001",
                                 content=b,
                                 task_type="SEMANTIC_SIMILARITY")["embedding"]
        return util.pytorch_cos_sim(va, vb).item()

    while True:
        # ─────────────────────────────── USER ACTIVE ───────────────────────
        if user_idle_turns < idle_threshold:
            user_input = input("You: ").strip()
            if user_input.lower() == "exit":
                break

            user_idle_turns = 0
            conversation.append({"speaker": "User", "text": user_input})
            last_speaker = "User"
            lowered = user_input.lower()

            # ----- choose responding NPC (name-mention > scorer) ------------
            target_npc = detect_addressed_npc(user_input, npc_list)
            speaker = target_npc or select_speaker(last_speaker, user_input)
            if not speaker:
                continue

            # ----- manual save ----------------------------------------------
            if any(t in lowered for t in save_memory_triggers):
                clean = user_input   # could strip prefix if desired
                add_to_long_term(speaker.name, [clean])
                print(f"[System] Saved that to {speaker.name}'s long-term memory.")
                continue

            # ----- manual show ---------------------------------------------
            if any(t in lowered for t in show_memory_triggers):
                mems = search_long_term(speaker.name, "*", k=20)
                if mems:
                    print(f"[{speaker.name}] Recent memories:")
                    for i, m in enumerate(mems, 1):
                        print(f" {i}. {m}")
                else:
                    print(f"[{speaker.name}] No long-term memories yet.")
                continue

            # ----- save USER turn if important -----------------------------
            add_to_short_term(speaker.name, Message("user", user_input))
            if is_important(user_input) or llm_is_important(user_input):
                add_to_long_term(speaker.name, [user_input])

            # ----- retrieve relevant memories ------------------------------
            hits = search_long_term(speaker.name, user_input, k=10)
            relevant = [m for m in hits if _similar(user_input, m) > 0.60][:3]
            recall_block = ""
            if relevant:
                recall_block = "\nRelevant memories:\n" + \
                               "\n".join(f"• {m}" for m in relevant) + "\n"

            # ----- build prompt & generate ---------------------------------
            prompt = recall_block + build_prompt(speaker, user_input, conversation, "User")
            response = gemini_model.generate_content(prompt).text.strip()
            print(f"{speaker.name}: {response}")

            # ----- save NPC turn if important ------------------------------
            conversation.append({"speaker": speaker.name, "text": response})
            speaker.last_spoken = current_turn
            update_relationship(speaker, "User", user_input)
            update_npc_to_npc_relationships(speaker.name, response)

            add_to_short_term(speaker.name, Message(speaker.name, response))
            if _should_save_npc_reply(response) and is_important(response):
                add_to_long_term(speaker.name, [response])

            last_speaker = speaker.name

        # ─────────────────────────────── USER IDLE ─────────────────────────
        else:
            last_text = conversation[-1]["text"]
            addressed = detect_addressed_npc(last_text, npc_list)
            speakers = [addressed] if addressed else []

            if not addressed:
                for _ in range(max_npc_turns):
                    nxt = select_speaker(last_speaker, last_text)
                    if not nxt or nxt in speakers:
                        break
                    speakers.append(nxt)

            for sp in speakers:
                prompt = build_prompt(sp, last_text, conversation, last_speaker)
                response = gemini_model.generate_content(prompt).text.strip()
                print(f"{sp.name}: {response}")

                conversation.append({"speaker": sp.name, "text": response})
                sp.last_spoken = current_turn
                update_relationship(sp, last_speaker, response)
                update_npc_to_npc_relationships(sp.name, response)

                add_to_short_term(sp.name, Message(sp.name, response))
                if _should_save_npc_reply(response) and is_important(response):
                    add_to_long_term(sp.name, [response])

                last_speaker = sp.name
                current_turn += 1

            # idle nudge
            if user_idle_turns >= idle_threshold:
                for npc in sorted(npc_list, key=lambda n: n.last_spoken):
                    if npc.name != last_speaker:
                        nudge = generate_nudge(npc)
                        print(nudge)
                        conversation.append({"speaker": npc.name, "text": nudge})
                        npc.last_spoken = current_turn

                        add_to_short_term(npc.name, Message(npc.name, nudge))
                        if is_important(nudge):
                            add_to_long_term(npc.name, [nudge])

                        current_turn += 1
                        break

        current_turn += 1
        user_idle_turns += 1

# === Start fresh session ====================================================
if __name__=="__main__":
    init_vectorstore()
    for npc in npc_list: clear_short_term(npc.name)
    chat_loop()
