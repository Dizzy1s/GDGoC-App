# importance.py  – smart memory-importance classifier (medical-aware)

import re, google.generativeai as genai

# ── BANAL LINES we never store ─────────────────────────────────────────────
_BANAL = re.compile(r"\b(hi|hello|hey|how are you|what's up|exit|bye)\b", re.I)

# ── GENERIC positive cues (unchanged) ──────────────────────────────────────
_EMOTION  = r"\b(feel|felt|happy|love|sad|angry|nervous|anxious|excited|depressed|lonely)\b"
_DECISION = r"\b(started|stopped|decided|joining|quitting|plan|goal|restart|restarted)\b"
_LIFESTYLE= r"\b(gym|exercise|diet|family|career|project|exam|relationship)\b"

# ── MEDICAL-specific cues ──────────────────────────────────────────────────
_SYMPTOM   = r"\b(pain|headache|fever|cough|fatigue|nausea|insomnia|anxiety|stress|panic)\b"
_CONDITION = r"\b(diabetes|hypertension|asthma|depression|adhd|anemia|covid|allergy|migraine)\b"
_MEDS      = r"\b(ibuprofen|paracetamol|insulin|prozac|sertraline|therapy|antibiotic|dose|medication)\b"
_CARE      = r"\b(doctor|therapist|appointment|clinic|prescribed|diagnosed|treatment|blood test)\b"

_MEDICAL   = f"({_SYMPTOM}|{_CONDITION}|{_MEDS}|{_CARE})"

_POSITIVE = re.compile(f"({_EMOTION}|{_DECISION}|{_LIFESTYLE}|{_MEDICAL})", re.I)

# ── Public helpers ─────────────────────────────────────────────────────────
def is_important(text: str) -> bool:
    """Fast heuristic: True if line deserves long-term storage."""
    if len(text) < 15:
        return False
    if _BANAL.search(text):
        return False
    if len(text) > 200:               # detailed paragraph
        return True
    return bool(_POSITIVE.search(text))

# Optional Gemini fallback if you want extra judgement on edge-cases.
# Call this only when `is_important` returns False and you still feel unsure.
def llm_is_important(text: str) -> bool:
    prompt = (
        "Answer Y or N only.\n"
        "Should the following sentence be stored as long-term memory for a "
        "medical-support chatbot? (Store if it reveals personal health, "
        "symptoms, treatment plans, emotions, or major life details.)\n\n"
        f"\"{text}\""
    )
    answer = genai.GenerativeModel("gemini-2.0-flash").generate_content(prompt).text
    return answer.strip().upper().startswith("Y")
