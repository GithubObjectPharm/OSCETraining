from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from openai import OpenAI
import os, re, uuid
from PyPDF2 import PdfReader
from docx import Document

# -------------------- Flask & OpenAI --------------------

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

client = OpenAI(
    api_key="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"
)

# -------------------- Global Case / Conversation State --------------------

case_context = {
    "raw": "",
    "facts": {},
    "summary": "",
    "persona": "",
    "gender": ""  # used by TTS
}

patient_state = {
    "summary": "",
    "turns": []
}

MAX_TURNS = 8  # small context for speed

# -------------------- Helpers --------------------

def extract_text(file_path: str) -> str:
    """Reads file contents for PDF, DOCX, or TXT."""
    ext = file_path.lower()
    if ext.endswith(".pdf"):
        text = ""
        with open(file_path, "rb") as f:
            pdf = PdfReader(f)
            for p in pdf.pages:
                text += p.extract_text() or ""
        return text
    elif ext.endswith(".docx"):
        doc = Document(file_path)
        return "\n".join(p.text for p in doc.paragraphs)
    elif ext.endswith(".txt"):
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    return ""


def extract_case_info(text: str) -> dict:
    """Extracts key structured data (name, age, gender, etc.)."""
    info = {}
    patterns = {
        "name": r"(?:Name|Patient Name|Pt Name)[:\s]*([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)",
        "age": r"(?:Age)[:\s]*([0-9]{1,3})",
        "gender": r"(?:Sex|Gender)[:\s]*([A-Za-z]+)",
        "weight": r"(?:Weight)[:\s]*([\d\.]+)",
        "height": r"(?:Height)[:\s]*([\d\.]+)",
        "allergies": r"(?:Allerg(?:y|ies))[:\s]*([^\n;]+)",
        "medications": r"(?:Medication(?:s)?|Rx|Drug(?:s)?)[:\s]*([^\n;]+)",
        "diagnosis": r"(?:Diagnosis|Condition|Medical Condition)[:\s]*([^\n]+)",
        "complaint": r"(?:Chief Complaint|Reason for Visit)[:\s]*([^\n]+)"
    }
    for k, pat in patterns.items():
        m = re.search(pat, text, re.I)
        if m:
            info[k] = m.group(1).strip()

    if "medications" in info:
        info["medications"] = [m.strip() for m in re.split(r"[;,]", info["medications"]) if m.strip()]
    if "allergies" in info:
        info["allergies"] = [a.strip() for a in re.split(r"[;,]", info["allergies"]) if a.strip()]

    # light pronoun sniff if gender missing (he/him vs she/her in case text)
    if "gender" not in info or not info["gender"]:
        pron = re.search(r"\b(he|she)\b", text, re.I)
        if pron:
            info["gender"] = "male" if pron.group(1).lower() == "he" else "female"

    return info


def infer_gender_from_name(name: str) -> str:
    """Guess gender based on name if not explicitly listed."""
    if not name:
        return ""
    first = name.split()[0].lower()
    female_names = {"jessica", "emily", "sarah", "olivia", "emma", "sophia", "isabella", "ava", "mia", "ella", "jess"}
    male_names = {"mike", "michael", "john", "james", "robert", "william", "david", "daniel", "matthew", "joseph"}
    if first in female_names:
        return "female"
    if first in male_names:
        return "male"
    return ""


def clamp_turns():
    """Keep only recent conversation turns."""
    if len(patient_state["turns"]) > MAX_TURNS:
        patient_state["turns"] = patient_state["turns"][-MAX_TURNS:]


def chat_once(messages, **kwargs):
    """Wrapper for a single chat completion (kept lean for latency)."""
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        **kwargs
    )
    return resp.choices[0].message.content.strip()


def summarize_running(previous_summary: str, new_user: str, last_assistant: str = "") -> str:
    """Maintain a compact running summary."""
    msgs = [
        {"role": "system", "content": (
            "Summarize the pharmacist–patient dialogue in 3–4 short sentences, "
            "from the patient's perspective. Track symptoms, timing, meds, adherence, and clarifications."
        )},
        {"role": "user", "content": (
            f"Previous summary:\n{previous_summary}\n\n"
            f"Pharmacist asked:\n{new_user}\n\n"
            f"Patient replied:\n{last_assistant}\n\n"
            "Update the running summary."
        )}
    ]
    return chat_once(msgs, temperature=0.2)


def detect_intent(utterance: str) -> str:
    """Lightweight intent classifier to steer brevity."""
    msgs = [
        {"role": "system", "content":
         "Classify this pharmacist question into one of: greeting, identity, age, gender, medications, allergies, "
         "diagnosis, symptom, duration, severity, lifestyle, adherence, triggers, relief, history, family, social, "
         "followup, plan, closing, other. Return one label only."},
        {"role": "user", "content": utterance}
    ]
    label = chat_once(msgs, temperature=0.0)
    return (label or "other").lower().strip()

# -------------------- Routes --------------------

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_case():
    """Read uploaded case, extract facts, build persona + summary."""
    global case_context, patient_state

    file = request.files.get("file")
    if not file:
        return jsonify({"error": "No file uploaded"}), 400

    filename = secure_filename(file.filename)
    path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(path)

    text = extract_text(path)
    if not text:
        return jsonify({"error": "Could not read file"}), 400

    facts = extract_case_info(text)
    if "gender" not in facts or not facts["gender"]:
        inferred = infer_gender_from_name(facts.get("name", ""))
        if inferred:
            facts["gender"] = inferred

    summary = chat_once(
        [
            {"role": "system", "content":
             "Write a brief first-person patient background (1–2 sentences). No greeting, no advice."},
            {"role": "user", "content": text}
        ],
        temperature=0.3
    )

    persona = chat_once(
        [
            {"role": "system", "content":
             "Describe the patient's tone and communication style in <=2 short lines "
             "(e.g., 'anxious but cooperative; concise, matter-of-fact')."},
            {"role": "user", "content": text}
        ],
        temperature=0.5
    )

    case_context = {
        "raw": text,
        "facts": facts,
        "summary": summary,
        "persona": persona,
        "gender": (facts.get("gender") or "").lower()
    }
    patient_state = {"summary": "", "turns": []}

    return jsonify({
        "message": "Case uploaded successfully.",
        "extracted": facts,
        "summary": summary,
        "persona": persona
    })


@app.route("/ask", methods=["POST"])
def ask():
    """Generate the patient's conversational reply."""
    global case_context, patient_state

    user_q = request.json.get("question", "").strip()
    if not user_q:
        return jsonify({"error": "No question"}), 400

    intent = detect_intent(user_q)
    turns_preview = patient_state["turns"][-6:]

    system_prompt = f"""
You are the patient in a pharmacy OSCE. Speak ONLY as the patient in first person.

PATIENT PERSONA:
{case_context.get('persona','')}

FACTS FROM CASE (do not contradict):
{case_context.get('facts',{})}

BACKGROUND SUMMARY (reference only):
{case_context.get('summary','')}

RUNNING SUMMARY (what's been said so far):
{patient_state.get('summary','(none)')}

STYLE RULES:
- Answer directly and briefly. 1–2 short sentences max unless specific detail is requested.
- Do NOT greet or ask "How can I help you?" — just answer the question.
- If something isn't known in the case, say you're not sure.
- Avoid repeating the whole story; no long monologues.
- Stay natural and consistent with the persona.
- Pharmacist’s intent: {intent}.
"""

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(turns_preview)
    messages.append({"role": "user", "content": user_q})

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.4,         # tighter for consistency/length
        presence_penalty=0.7,    # discourages repetition/rambling
        frequency_penalty=0.7,
        max_tokens=50            # ~1 short sentence (2 if small)
    )

    answer = completion.choices[0].message.content.strip()

    # Store conversation & summarize
    patient_state["turns"].append({"role": "user", "content": user_q})
    patient_state["turns"].append({"role": "assistant", "content": answer})
    clamp_turns()

    try:
        patient_state["summary"] = summarize_running(
            patient_state["summary"], user_q, answer
        )
    except Exception:
        pass

    return jsonify({"answer": answer})


@app.route("/reset-case", methods=["POST"])
def reset_case():
    global case_context, patient_state
    case_context = {"raw": "", "facts": {}, "summary": "", "persona": "", "gender": ""}
    patient_state = {"summary": "", "turns": []}
    return jsonify({"status": "reset"})


@app.route("/tts", methods=["POST"])
def tts():
    """Generate quick voice output and wait until MP3 file ready (prevents UI flicker)."""
    text = request.json.get("text", "").strip()
    if not text:
        return jsonify({"error": "No text provided"}), 400

    clean = re.sub(r"[\n\r]+", " ", text).strip()[:420]

    gender = case_context.get("gender", "").lower()
    if "female" in gender:
        voice_choice = "alloy"
    elif "male" in gender:
        voice_choice = "verse"
    else:
        voice_choice = "verse"

    audio_filename = f"voice_{uuid.uuid4().hex}.mp3"
    audio_path = os.path.join(app.config["UPLOAD_FOLDER"], audio_filename)

    # wait for speech to be fully written before responding
    with client.audio.speech.with_streaming_response.create(
        model="gpt-4o-mini-tts",
        voice=voice_choice,
        input=clean
    ) as r:
        r.stream_to_file(audio_path)

    return jsonify({"audio": f"/uploads/{audio_filename}", "ready": True})


@app.route("/uploads/<path:filename>")
def serve_upload(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


# -------------------- Run --------------------

if __name__ == "__main__":
    app.run(debug=True)
