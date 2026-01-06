import os
import re
import json
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import random

# -------------------------------------------------------
# ENV
# -------------------------------------------------------
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# -------------------------------------------------------
# DEVICE
# -------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------------------------------
# PATHS
# -------------------------------------------------------
QGEN_MODEL_PATH = "./interview_models/qgen_flan_t5"
EVAL_MODEL_PATH = "./interview_models/interview_eval_model"

# Optional persistent memory
MEMORY_FILE = "asked_questions.json"

# -------------------------------------------------------
# LOAD MODELS
# -------------------------------------------------------
tokenizer_q = AutoTokenizer.from_pretrained(QGEN_MODEL_PATH)
model_q = AutoModelForSeq2SeqLM.from_pretrained(QGEN_MODEL_PATH).to(device)

tokenizer_e = AutoTokenizer.from_pretrained(EVAL_MODEL_PATH)
model_e = AutoModelForSeq2SeqLM.from_pretrained(EVAL_MODEL_PATH).to(device)

model_q.eval()
model_e.eval()

# -------------------------------------------------------
# HELPERS
# -------------------------------------------------------
def normalize_question(q):
    return re.sub(r"[^a-z0-9 ]", "", q.lower()).strip()

def is_bad_question(q):
    q = q.lower().strip()

    # self-comparison like X vs X
    if re.search(r"(\b\w+\b).+different from.+\1", q):
        return True

    # repeated word spam
    tokens = q.split()
    for t in set(tokens):
        if tokens.count(t) >= 3:
            return True

    if "same as" in q or "difference between x and x" in q:
        return True

    return False

# -------------------------------------------------------
# QUESTION MEMORY (SESSION-LEVEL)
# -------------------------------------------------------
asked_questions = set()

# Uncomment to persist across runs
def load_memory():
    global asked_questions
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r") as f:
            asked_questions = set(json.load(f))

def save_memory():
    with open(MEMORY_FILE, "w") as f:
        json.dump(list(asked_questions), f)


def safe_json_parse(text):
    text = text.strip()

    # Case 1: direct parse
    try:
        return json.loads(text)
    except:
        pass

    # Case 2: add braces if missing
    try:
        if not text.startswith("{"):
            text = "{" + text
        if not text.endswith("}"):
            text = text + "}"
        return json.loads(text)
    except:
        pass

    # Case 3: extract {...} using regex
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        try:
            return json.loads(match.group())
        except:
            pass

    # Case 4: parse scores manually (last resort)
    try:
        scores = {}
        feedback = ""
        # Find numeric scores
        for k in ["technical", "communication", "depth", "overall"]:
            m = re.search(rf'{k}"?\s*:\s*(\d+)', text)
            if m:
                scores[k] = int(m.group(1))
            else:
                if k != "overall":
                    scores[k] = 2
        # feedback
        m = re.search(r'feedback"\s*:\s*"(.*?)"', text)
        if m:
            feedback = m.group(1)
        return {
            "scores": {k: scores.get(k, 2) for k in ["technical","communication","depth"]},
            "overall": scores.get("overall",2),
            "feedback": feedback or "Evaluation parsing recovered."
        }
    except:
        pass

    # Fallback
    return {
        "scores": {"technical": 2, "communication": 2, "depth": 1},
        "overall": 2,
        "feedback": "Evaluation parsing failed."
    }
    
def is_valid_question(role, question):
    q = question.strip().lower()

    if role.lower() in ["fullstack developer", "frontend developer", "backend developer"]:
        # Accept LeetCode questions
        if q.startswith("leetcode") and "—" in q:
            return True
        # Accept conceptual questions (>=6 words, not repeated)
        if len(q.split()) > 6 and not is_bad_question(q) and random.random() < 0.7:
            return True
        return False

    elif role.lower() in ["product manager", "pm"]:
        return len(q.split()) > 5 and not is_bad_question(q)

    return True

# -------------------------------------------------------
# QUESTION GENERATION (STABLE + VARIED)
# -------------------------------------------------------
def generate_question(role, difficulty, max_tries=10):
    """
    Generates a single interview question for a given role and difficulty.
    For Fullstack/Backend/Frontend:
      - Mixes LeetCode-style and conceptual questions
    For PM/PM-related roles:
      - Generates only conceptual questions
    """

    prompt = (
        f"task: generate_question | role: {role} | difficulty: {difficulty}\n"
        "Generate a single interview question.\n"
        "Prefer conceptual and real-world questions.\n"
        "Avoid coding challenge or LeetCode-style questions unless explicitly requested.\n"
    )

    for _ in range(max_tries):
        inputs = tokenizer_q(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=256
        ).to(device)

        with torch.no_grad():
            output = model_q.generate(
                **inputs,
                max_new_tokens=50,
                num_beams=4,
                do_sample=True,
                temperature=0.8,  # more diversity
                top_p=0.95,       # allow lower-probability conceptual outputs
                repetition_penalty=1.2,
                no_repeat_ngram_size=2,
                early_stopping=True
            )

        question = tokenizer_q.decode(output[0], skip_special_tokens=True).strip()
        print("RAW QGEN OUTPUT:", question)

        norm = normalize_question(question)
        if norm in asked_questions or is_bad_question(question):
            continue

        q_words = len(question.split())

        # -----------------------------
        # Fullstack / Backend / Frontend
        # -----------------------------
        if role.lower() in ["fullstack developer", "backend developer", "frontend developer"]:
            is_leetcode = question.lower().startswith("leetcode")
            # Prefer conceptual
            if not is_leetcode and q_words >= 5:
                asked_questions.add(norm)
                return question
            # Allow LeetCode occasionally
            if is_leetcode and random.random() < 0.3:
                asked_questions.add(norm)
                return question
            continue

        # Product Manager / PM
        elif role.lower() in ["product manager", "pm"]:
            if q_words > 5:
                asked_questions.add(norm)
                return question
            continue

        # -----------------------------
        # Other roles (fallback)
        # -----------------------------
        else:
            if q_words > 3:
                asked_questions.add(norm)
                return question

    # -----------------------------
    # Fallback after max_tries
    # -----------------------------
    return "Can you explain a core concept related to your role and how you have applied it in practice?"

# -------------------------------------------------------
# ANSWER EVALUATION (STRICT JSON)
# -------------------------------------------------------
def evaluate_answer(role, question, answer):
    prompt = (
        "Evaluate the following interview answer.\n"
        "Return a valid JSON object using braces { } exactly as shown.\n"
        "Include keys: scores (technical, communication, depth), overall, feedback.\n"
        f"Role: {role}\n"
        f"Question: {question}\n"
        f"Candidate answer: {answer}\n"
    )

    inputs = tokenizer_e(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=768
    ).to(device)

    with torch.no_grad():
        output = model_e.generate(
            **inputs,
            max_new_tokens=180,
            num_beams=2,
            do_sample=False,
            repetition_penalty=1.1,
            early_stopping=True
        )

    text = tokenizer_e.decode(output[0], skip_special_tokens=True)
    print("RAW EVAL OUTPUT:", text)
    try:
        return safe_json_parse(text)
    except Exception:
        return {
            "scores": {"technical": 2, "communication": 2, "depth": 1},
            "overall": 2,
            "feedback": "Evaluation parsing failed."
        }
# -------------------------------------------------------
# FOLLOW-UP GENERATION (NON-REPEATING)
# -------------------------------------------------------
def generate_followup(role, question, answer):
    prompt = (
        f"task: generate_followup | role: {role}\n"
        f"Question: {question}\n"
        f"Candidate answer: {answer}\n"
        "Generate one relevant follow-up question:"
    )

    inputs = tokenizer_q(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).to(device)

    with torch.no_grad():
        output = model_q.generate(
            **inputs,
            max_new_tokens=40,   # slightly longer
            num_beams=4,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.2,
            early_stopping=True
        )

    followup = tokenizer_q.decode(output[0], skip_special_tokens=True).strip()

    # Keep only one question mark sentence
    if "?" in followup:
        followup = followup.split("?")[0] + "?"

    # Reject if very short (<3 words) or identical to main question
    if len(followup.split()) < 3 or normalize_question(followup) == normalize_question(question):
        return "Can you go deeper into your approach and discuss trade-offs or real-world constraints?"

    # Simple duplicate word heuristic
    tokens = followup.lower().split()
    for t in set(tokens):
        if tokens.count(t) > 3:  # repeated word spam
            return "Can you explain your approach in more detail and discuss alternatives?"

    return followup

# -------------------------------------------------------
# INTERVIEW LOOP
# -------------------------------------------------------
def run_interview(role, difficulty):

    print("\nINTERVIEW START\n")

    question = generate_question(role, difficulty)
    print("Question:")
    print(question)

    answer = input("\nYour Answer:\n")

    evaluation = evaluate_answer(role, question, answer)
    print("\nEvaluation Feedback:")
    print(json.dumps(evaluation, indent=2))

    followup = generate_followup(role, question, answer)
    print("\nFollow-up Question:")
    print(followup)

    print("\nINTERVIEW STEP COMPLETE\n")

# -------------------------------------------------------
# RUN
# -------------------------------------------------------
load_memory()   
run_interview(role="Fullstack Developer", difficulty="easy")
save_memory()  