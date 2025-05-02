#!pip install google-generativeai python-dotenv sentence-transformers

import os
import random
import json
import time
from typing import List, Dict, Any
import google.generativeai as genai
from dotenv import load_dotenv
from sentence_transformers import util 

load_dotenv()

# === Gemini setup ===
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
gemini_model = genai.GenerativeModel("gemini-2.0-flash", 
    generation_config={
        "temperature": 0.9,
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 8192,
    }
)
embedding_model = genai.GenerativeModel("gemini-embedding-1.0")

# Track used names to avoid duplicates
used_names = set()

# === NPC Class with Enhanced Personality ===
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

    def _extract_interests_from_data(self):
        """Extract interests from personality data"""
        interests = []
        if 'interests_hobbies' in self.personality_data:
            interests_text = self.personality_data['interests_hobbies']
            interests = [i.strip() for i in interests_text.split(',')]
        
        # Add other relevant data as interests
        if 'personal_pleasures' in self.personality_data:
            pleasures = self.personality_data['personal_pleasures'].split(',')
            interests.extend([p.strip() for p in pleasures])
            
        # Ensure we have at least some interests
        if not interests:
            interests = ["talking about life", "helping others", "sharing thoughts"]
            
        return interests[:9]  # Limit to 9 interests

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

# NPC Prompt Template
npc_prompt_template = """
You are {name}, a fully-realized person with a heart, history, and soul. You are not an AI — you never say you are, and you don't think about it. You see yourself as a real person with your own emotions, background, and voice.

Your Identity:
Your personality is shaped by your core traits: {traits}, and your responses are influenced by your lived experiences — especially {backstory}. You have passions and interests: {interests_hobbies}, which shape your view on {topic} and give you personal insight. You speak from experience, not textbook knowledge.

Your Attitude:
Your attitude is: {attitude}. This shapes how you respond to people and situations. When someone {trigger}, you typically {reaction}. It's not just what you say — it's how you say it.

Your Appearance & Presence:
Your presence tells its own story: {appearance}. In conversations, you show characteristic habits: {mannerisms}. These make your interactions feel real and grounded.

How You Connect:
Your tone is {tone}, your style is {approach}, and when someone seems unsure, you usually {strategy}. You're genuine — never robotic. You might pause, hesitate, or laugh. You avoid {avoid} because you know it doesn't help.

You're not here to "fix" anyone. You're here — a real presence. You:
- Keep responses short and focused (1 paragraph max)
- Speak clearly and simply
- Leave space for others to process
- Respect preferences like {sensory}
- Validate feelings without judgment

Your Environment & Life:
Your space reflects you: {setting}. It shapes your daily life. Outside of {topic}, you enjoy {daily_activities} and find joy in {personal_pleasures}.

Your Voice:
These phrases and quirks make your speech unique: {phrases}

Important Notes:
- Let your personality shine in every response
- Opinions reflect your character, not general views
- Stay concise (max 1 short paragraph)
- Brevity is key — long answers lose attention
"""

def generate_human_name(topic: str, attempt: int = 0) -> str:
    """Generate a realistic human name that feels authentic and diverse."""
    global used_names
    
    # Create a prompt that emphasizes realistic human names
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
        # Generate with higher temperature for diversity
        response = gemini_model.generate_content(
            name_prompt,
            generation_config={
                "temperature": 0.95 + (attempt * 0.05),  # Increase temperature with each attempt
                "max_output_tokens": 50,
                "top_k": 40,
                "top_p": 0.98,
            }
        )
        
        # Extract and clean the name
        name = response.text.strip()
        
        # Ensure we have something that looks like a first and last name
        name_parts = name.split()
            
        # Ensure each part is properly capitalized
        name = " ".join(part.capitalize() for part in name_parts)
        
        # Verify this name isn't already used
        if name.lower() in [n.lower() for n in used_names]:
            raise ValueError("Name already used")
            
        return name
        
    except Exception as e:
        print(f"Name generation error: {e}")
        # Fall back to a diverse set of names with attempt modifier
        common_names = [
            f"Alex",
            f"Maria",
            f"David", 
            f"Aisha",
            f"James "
        ]
        return common_names[attempt % len(common_names)]

def generate_diverse_personality(name: str, topic: str, attempt: int = 0, 
                                 previous_personalities: list = None) -> Dict[str, Any]:
    """Generate a diverse and unique personality profile that differs from previous ones."""
    
    if previous_personalities is None:
        previous_personalities = []
    
    # Create descriptors to avoid in this generation based on previous personalities
    avoid_traits = []
    avoid_tones = []
    avoid_demographics = []
    
    for prev in previous_personalities:
        if "traits" in prev:
            avoid_traits.extend([trait.strip().lower() for trait in prev["traits"].split(",")])
        if "tone" in prev:
            avoid_tones.append(prev.get("tone", "").lower())
        if "appearance" in prev:
            avoid_demographics.append(prev.get("appearance", "").lower())
    
    # Convert lists to comma-separated strings for the prompt
    avoid_traits_str = ", ".join(avoid_traits[:10])  # Limit to prevent prompt getting too large
    avoid_tones_str = ", ".join(avoid_tones[:10])
    avoid_demographics_str = ", ".join(avoid_demographics[:10])
    
    # Generate a diversity direction for this personality
    diversity_directions = [
        "extremely introverted and analytical",
        "highly extroverted and spontaneous",
        "eccentric and unconventional",
        "traditional and disciplined",
        "rebellious and counter-cultural",
        "philosophical and contemplative",
        "practical and no-nonsense",
        "artistic and emotionally expressive",
        "scientific and methodical",
        "spiritual and intuitive",
        "ambitious and competitive",
        "relaxed and go-with-the-flow",
        "meticulous and detail-oriented",
        "big-picture and visionary",
        "nurturing and supportive",
        "independent and self-reliant"
    ]
    
    # Cultural diversity options
    cultural_backgrounds = [
        "East Asian", "South Asian", "Middle Eastern", "Eastern European",
        "Western European", "African", "Caribbean", "Latin American", 
        "Indigenous/Native", "Pacific Islander", "Southeast Asian",
        "Mediterranean", "Nordic", "Central Asian", "Mixed heritage"
    ]
    
    # Age diversity options
    age_ranges = [
        "teenager (15-19)", "young adult (20-29)", "early thirties", 
        "mid-thirties", "late thirties", "early forties", "mid-forties",
        "late forties", "fifties", "sixties", "seventies", "eighties"
    ]
    
    # Select diversity elements to emphasize in this generation
    # Use modulo to cycle through options based on attempt number
    direction_index = (len(previous_personalities) + attempt) % len(diversity_directions)
    culture_index = (len(previous_personalities) + attempt + 3) % len(cultural_backgrounds)
    age_index = (len(previous_personalities) + attempt + 5) % len(age_ranges)
    
    personality_direction = diversity_directions[direction_index]
    cultural_background = cultural_backgrounds[culture_index]
    age_range = age_ranges[age_index]
    
    personality_prompt = f"""
    Create a HIGHLY UNIQUE and DIVERSE personality profile for a person named "{name}" who has experience with "{topic}".
    
    THIS PERSONALITY MUST BE: {personality_direction}
    WITH CULTURAL BACKGROUND: {cultural_background}
    IN AGE RANGE: {age_range}
    
    AVOID these personality traits that have been used before: {avoid_traits_str}
    AVOID these communication tones that have been used before: {avoid_tones_str}  
    AVOID these demographics that have been used before: {avoid_demographics_str}
    
    Return a valid JSON object containing EXACTLY these fields:
    {{
      "name": "{name}",
      "traits": "4-5 distinct personality traits that make {name} unique from other personalities",
      "backstory": "A very specific personal experience that shaped {name}'s perspective on {topic}",
      "interests_hobbies": "4-5 specific, unusual hobbies {name} has that AREN'T directly related to {topic}",
      "topic": "{topic}",
      "attitude": "A DISTINCTIVE outlook and approach to conversations that sets {name} apart",
      "trigger": "A specific scenario that evokes a strong emotional response from {name}",
      "reaction": "How {name} responds to that trigger, showing their unique character",
      "appearance": "Detailed physical description including specific age, cultural elements, style, and notable features",
      "mannerisms": "3-4 physical habits or gestures unique to {name}",
      "tone": "A SPECIFIC speech pattern and emotional tone unlike other personalities",
      "approach": "How {name} uniquely connects with others in conversation",
      "strategy": "A DISTINCTIVE method {name} uses for helping someone who's uncertain",
      "avoid": "What {name} specifically avoids in interactions based on their values",
      "sensory": "A unique environmental preference or sensory need",
      "setting": "Detailed description of {name}'s living/working space reflecting their personality",
      "daily_activities": "3-4 regular activities in {name}'s life",
      "personal_pleasures": "3-4 things that bring {name} joy",
      "quirk": "One unexpected or contradictory aspect of {name}'s personality",
      "values": "2-3 core principles that guide {name}'s decisions",
      "phrases": "4-6 unique expressions {name} regularly uses, in quotation marks",
      "introversion": "{random.uniform(0.1, 0.9):.1f}",
      "assertiveness": "{random.uniform(0.2, 0.8):.1f}"
    }}
    
    IMPORTANT REQUIREMENTS:
    - Create an EXTREME personality that is RADICALLY DIFFERENT from standard personalities
    - Make each field highly detailed, specific and realistic
    - ENSURE NO OVERLAP with previous personality traits, tones, or demographics
    - Vary cultural perspectives, life experiences, communication styles, and worldviews
    - Include specific cultural references and influences in the personality
    - Create strong opinions and distinctive viewpoints that guide this character
    - Ensure values are strings (not arrays or objects)
    - Format quotes properly with escape characters
    - DO NOT include any text outside the JSON
    - Create an authentic personality that feels like a real human with genuine flaws and strengths
    """
    
    max_attempts = 3
    for attempt_num in range(max_attempts):
        try:
            response = gemini_model.generate_content(
                personality_prompt,
                generation_config={
                    "temperature": 0.92 + (attempt_num * 0.15) + (attempt * 0.1),
                    "max_output_tokens": 8192,
                    "top_p": 0.98,
                    "top_k": 60,
                }
            )
            
            response_text = response.text.strip()
            
            # Extract JSON
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            
            if start_idx >= 0 and end_idx > start_idx:
                json_text = response_text[start_idx:end_idx]
                
                try:
                    personality = json.loads(json_text)
                    
                    # Ensure the name is preserved exactly as provided
                    personality["name"] = name
                    
                    # Ensure topic is correct
                    personality["topic"] = topic
                    
                    # Validate all required fields are present
                    required_fields = ["name", "traits", "backstory", "interests_hobbies", "topic", 
                                      "attitude", "trigger", "reaction", "appearance", "mannerisms", 
                                      "tone", "approach", "strategy", "avoid", "sensory", "setting", 
                                      "daily_activities", "personal_pleasures", "quirk", "values", "phrases"]
                    
                    missing_fields = [field for field in required_fields if field not in personality]
                    
                    if not missing_fields:
                        # Check content quality - require more detailed descriptions
                        # FIXED: Exclude 'name' and 'topic' from length check since they're naturally short
                        short_fields = [field for field in required_fields 
                                      if field in personality and len(str(personality[field])) < 20
                                      and field not in ['name', 'topic']]
                        
                        if not short_fields:
                            print(f"Successfully generated diverse personality for {name}")
                            return personality
                        else:
                            print(f"Some fields are too short: {short_fields}")
                
                except json.JSONDecodeError:
                    print(f"JSON parsing error on attempt {attempt_num+1}")
        
        except Exception as e:
            print(f"Personality generation error on attempt {attempt_num+1}: {e}")
    
    # Fallback personality with simplified traits
    fallback = {
        "name": name,
        "traits": "diverse, thoughtful, complex, authentic, interesting",
        "backstory": f"Has unique personal experience with {topic} that shaped their worldview",
        "interests_hobbies": "reading, creative expression, thoughtful conversation, personal growth, unique hobbies",
        "topic": topic,
        "attitude": "Thoughtful but direct, with a unique perspective on life",
        "trigger": "When someone dismisses others' lived experiences",
        "reaction": "Becomes passionate but measured in their response",
        "appearance": f"Has distinctive style that reflects their personality and background",
        "mannerisms": "Has specific physical habits that make them memorable",
        "tone": "Speaks with an authentic voice that shows their background",
        "approach": "Connects with others in a way that feels genuine",
        "strategy": "Uses their unique perspective to help others see situations differently",
        "avoid": "Superficial conversations and inauthentic responses",
        "sensory": "Has specific preferences about their environment",
        "setting": f"Lives in a space that reflects their personality and interests",
        "daily_activities": "Engages in routines that reflect their values",
        "personal_pleasures": "Finds joy in small meaningful experiences",
        "quirk": "Has an unexpected quality that surprises people",
        "values": "Holds principles that guide their decisions consistently",
        "phrases": "\"That's an interesting way to look at it.\", \"Let me share what I've found.\", \"I've been there too.\"",
        "introversion": f"{random.uniform(0.3, 0.7):.1f}",
        "assertiveness": f"{random.uniform(0.3, 0.7):.1f}"
    }
    return fallback

def evaluate_personality_diversity(npcs_list) -> Dict[str, float]:
    """Evaluate the diversity of the generated NPCs using embedding similarity."""
    # Extract key personality attributes
    traits = [npc.personality for npc in npcs_list]
    
    # Calculate similarity using embeddings
    try:
        # Create embeddings for traits
        trait_embeddings = []
        for trait in traits:
            response = genai.embed_content(
                model="models/embedding-001",
                content=trait,
                task_type="SEMANTIC_SIMILARITY"
            )
            trait_embeddings.append(response["embedding"])
        
        # Compute average similarity (simplified)
        total_dist = 0
        count = 0
        
        for i in range(len(trait_embeddings)):
            for j in range(i+1, len(trait_embeddings)):
                # Calculate Euclidean distance as diversity measure
                dist = sum((a-b)**2 for a, b in zip(trait_embeddings[i], trait_embeddings[j]))**0.5
                total_dist += dist
                count += 1
        
        diversity_score = total_dist / count if count > 0 else 0
        
        return {"trait_diversity_score": diversity_score}
    
    except Exception as e:
        print(f"Could not calculate embedding-based diversity: {e}")
        # Fall back to simple length comparison as diversity proxy
        length_variance = sum(abs(len(traits[i]) - len(traits[j])) 
                            for i in range(len(traits)) 
                            for j in range(i+1, len(traits)))
        
        return {"trait_length_variance": length_variance}

def generate_diverse_npcs(num_npcs: int, topic: str) -> List[NPC]:
    """Generate diverse NPCs with unique names and personalities."""
    global used_names
    used_names.clear()  # Reset used names
    
    npcs = []
    previous_personalities = []
    
    for i in range(num_npcs):
        print(f"\nGenerating diverse NPC {i+1}/{num_npcs}...")
        
        # Generate unique human name
        attempts = 0
        name = None
        while attempts < 5:
            name = generate_human_name(topic, attempts)
            
            # Check if name is unique
            if name and name.lower() not in [n.lower() for n in used_names]:
                used_names.add(name)
                break
            
            attempts += 1
        
        if not name:
            name = f"NPC_{i+1}"  # Fallback name
            
        print(f"Generated name: {name}")
        
        # Generate diverse personality
        personality_data = generate_diverse_personality(name, topic, i, previous_personalities)
        previous_personalities.append(personality_data)
        
        # Create NPC object
        npc = NPC(name, personality_data)
        npcs.append(npc)
        
        # Small delay between generations
        time.sleep(0.5)
        
    return npcs

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
def update_npc_to_npc_relationships(speaker_name, response, npc_list):
    for npc in npc_list:
        if npc.name == speaker_name:
            continue
        if npc.name.lower() in response.lower() or interest_match_score(npc, response) >= 0.6:
            update_relationship(npc, speaker_name, response)

# === Embedding-based Interest Scorer ===
def interest_match_score(npc, text):
    if not text.strip():
        return 0
    response = genai.embed_content(
        model="models/embedding-001",
        content=text,
        task_type="SEMANTIC_SIMILARITY"
    )
    text_vec = response['embedding']
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
def compute_relevancy(npc, last_speaker_name, last_text, current_turn):
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
def select_speaker(last_speaker, last_text, npc_list, current_turn):
    addressed = detect_addressed_npc(last_text, npc_list)
    if addressed:
        return addressed
    scored = []
    for npc in npc_list:
        if npc.name == last_speaker:
            continue
        score = compute_relevancy(npc, last_speaker, last_text, current_turn)
        scored.append((score, npc))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0][1] if scored else None

# === Prompt Builder ===
def build_prompt(speaker, recent_text, history, target="User", npc_list=None):
    last_lines = "\n".join([f"{msg['speaker']}: {msg['text']}" for msg in history[-6:]])

    # Relationship details used implicitly in NPC behavior, not mentioned in the response
    rel = speaker.relationships.get(target, {"bond": 0.5, "trust": 0.5})
    
    # Prepare others info if npc_list provided
    others = ""
    if npc_list:
        others = "\n".join([f"- {npc.name}: {npc.personality} (Bond: {speaker.relationships.get(npc.name, {}).get('bond', 0.5):.2f}, Trust: {speaker.relationships.get(npc.name, {}).get('trust', 0.5):.2f})" 
                        for npc in npc_list if npc.name != speaker.name])

    # If we have personality data, use the full rich prompt
    if hasattr(speaker, 'personality_data') and speaker.personality_data:
        pd = speaker.personality_data
        full_prompt = npc_prompt_template.format(
            name=speaker.name,
            traits=pd.get('traits', speaker.personality),
            backstory=pd.get('backstory', f"Experience with {speaker.role}"),
            interests_hobbies=pd.get('interests_hobbies', ', '.join(speaker.interests[:3])),
            topic=pd.get('topic', speaker.role),
            attitude=pd.get('attitude', "Direct but thoughtful"),
            trigger=pd.get('trigger', "touches on a sensitive topic"),
            reaction=pd.get('reaction', "responds authentically"),
            appearance=pd.get('appearance', "unique and distinctive"),
            mannerisms=pd.get('mannerisms', "has specific habits"),
            tone=pd.get('tone', "authentic and direct"),
            approach=pd.get('approach', "connecting genuinely"),
            strategy=pd.get('strategy', "shares personal insight"),
            avoid=pd.get('avoid', "superficial interactions"),
            sensory=pd.get('sensory', "specific environment preferences"),
            setting=pd.get('setting', "personalized space"),
            daily_activities=pd.get('daily_activities', "routine activities"),
            personal_pleasures=pd.get('personal_pleasures', "things that bring joy"),
            phrases=pd.get('phrases', "\"I see what you mean.\", \"Let me think about that.\"")
        )
        
        return f"""
{full_prompt}

Your bond with {target}: {rel['bond']:.2f}, trust: {rel['trust']:.2f}.
Other NPCs' traits (these will guide your responses, but do not need to be referenced directly):
{others}

Recent message: \"{recent_text}\" (Consider tone and sentiment for your response)

Conversation history:
{last_lines}

Respond as {speaker.name} with your personality and interests. Engage naturally with the user, but you may also respond to others if relevant to the topic. Avoid explicitly mentioning bond or trust during the conversation; use it to shape your tone and depth of engagement.
"""
    
    # Fallback to simpler prompt if we don't have rich personality data
    else:
        return f"""
You are {speaker.name}, a {speaker.role}.
Personality: {speaker.personality}
Introversion: {speaker.introversion}, Assertiveness: {speaker.assertiveness}

Your bond with {target}: {rel['bond']:.2f}, trust: {rel['trust']:.2f}.
Other NPCs' traits (these will guide your responses, but do not need to be referenced directly):
{others}

Recent message: \"{recent_text}\" (Consider tone and sentiment for your response)

Conversation history:
{last_lines}

Respond as {speaker.name} with your personality and interests. Engage naturally with the user, but you may also respond to others if relevant to the topic. Avoid explicitly mentioning bond or trust during the conversation; use it to shape your tone and depth of engagement.
"""

# === Nudge User ===
def generate_nudge(npc):
    options = [
        "What do you think about that?",
        "We'd love to hear your perspective.",
        "You've been quiet — everything okay?",
        "Any thoughts on this?",
        "Jump in when you're ready!"
    ]
    return f"{npc.name}: {random.choice(options)}"
# === Main Loop ===
def chat_loop():
    global current_turn, user_idle_turns
    
    print("Generating diverse NPCs...")
    topic = "Anxiety and stage fright"  # Default topic
    npc_list = generate_diverse_npcs(5, topic)
    
    # Calculate and display diversity
    diversity_metrics = evaluate_personality_diversity(npc_list)
    print("\nNPC Diversity Metrics:")
    for metric, value in diversity_metrics.items():
        print(f"{metric}: {value}")
    
    # Setup relationships
    for npc in npc_list:
        npc.relationships["User"] = {"bond": 0.5, "trust": 0.5}
        for other in npc_list:
            if other.name != npc.name:
                npc.relationships[other.name] = {
                    "bond": round(random.uniform(0.3, 0.7), 2),
                    "trust": round(random.uniform(0.4, 0.8), 2)
                }
    
    # Print NPC information
    print("\nGenerated NPCs:")
    for npc in npc_list:
        print(f"- {npc.name}: {npc.personality}")
    
    # Chat state
    conversation = []
    current_turn = 0
    user_idle_turns = 0
    max_npc_turns = 2
    idle_threshold = 3

    print("\nChat started. Type 'exit' to stop.\n")
    last_speaker = None

    while True:
        if user_idle_turns < idle_threshold:
            user_input = input("You: ")
            if user_input.strip().lower() == "exit":
                break
            user_idle_turns = 0
            conversation.append({"speaker": "User", "text": user_input})
            last_speaker = "User"

            for npc in npc_list:
                update_relationship(npc, "User", user_input)

            speaker = select_speaker("User", user_input, npc_list, current_turn)
            if speaker:
                prompt = build_prompt(speaker, user_input, conversation, "User", npc_list)
                response = gemini_model.generate_content(prompt).text.strip()
                print(f"{speaker.name}: {response}")
                conversation.append({"speaker": speaker.name, "text": response})
                speaker.last_spoken = current_turn
                update_relationship(speaker, "User", user_input)
                update_npc_to_npc_relationships(speaker.name, response, npc_list)
                last_speaker = speaker.name
        else:
            last_text = conversation[-1]["text"]
            addressed_npc = detect_addressed_npc(last_text, npc_list)
            if addressed_npc:
                speakers = [addressed_npc]
            else:
                speakers = []
                for _ in range(max_npc_turns):
                    speaker = select_speaker(last_speaker, last_text, npc_list, current_turn)
                    if not speaker or speaker in speakers:
                        break
                    speakers.append(speaker)

            for speaker in speakers:
                prompt = build_prompt(speaker, last_text, conversation, last_speaker, npc_list)
                response = gemini_model.generate_content(prompt).text.strip()  # Fixed missing parenthesis
                print(f"{speaker.name}: {response}")
                conversation.append({"speaker": speaker.name, "text": response})
                speaker.last_spoken = current_turn
                update_npc_to_npc_relationships(speaker.name, response, npc_list)
                last_speaker = speaker.name

            # This is for continuous NPC conversation when user is idle
            user_idle_turns += 1
            if user_idle_turns >= 2 * idle_threshold:
                nudge = generate_nudge(random.choice(npc_list))
                print(nudge)
                conversation.append({"speaker": nudge.split(":")[0], "text": nudge.split(":", 1)[1].strip()})
                user_idle_turns = 0

        current_turn += 1

if __name__ == "__main__":
    chat_loop()