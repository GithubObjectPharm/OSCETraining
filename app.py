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

# ---------------- OPENAI CONFIG ----------------
OPENAI_API_KEY = "sk-PASTE-YOUR-KEY-HERE"

client = OpenAI(api_key=OPENAI_API_KEY)
# -------------------- Global Case / Conversation State --------------------

case_context = {
    "raw": "",
    "facts": {},
    "summary": "",
    "persona": "",
    "gender": ""
}

patient_state = {
    "summary": "",
    "turns": []
}

MAX_TURNS = 8


def extract_references(text: str) -> str:
    """
    Extract references from OSCE-style cases.
    Works even if PDF references are not text-extractable.
    """
    refs = []

    # 1. Explicit references section (best case)
    m = re.search(r"(References|REFERENCES)\s*[:\n]+(.+?)(\n\n|\Z)", text, re.I | re.S)
    if m:
        return m.group(2).strip().replace("\n", "; ")

    # 2. Keyword-based inference (OSCE standard)
    keywords = [
        "Health Canada",
        "CPS",
        "Compendium of Pharmaceuticals",
        "Product Monograph",
        "FDA",
        "UpToDate",
        "Lexicomp",
        "NAPRA",
        "ISMP",
        "RxTx"
    ]

    for k in keywords:
        if re.search(k, text, re.I):
            refs.append(k)

    if refs:
        return "; ".join(sorted(set(refs)))

    # 3. Last-resort fallback
    return "Standard clinical references (e.g., CPS, product monograph)"


def extract_text(file_path: str) -> str:
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

    # post processing
    if "medications" in info:
        info["medications"] = [m.strip() for m in re.split(r"[;,]", info["medications"]) if m.strip()]

    if "allergies" in info:
        info["allergies"] = [m.strip() for m in re.split(r"[;,]", info["allergies"]) if m.strip()]

    if "gender" not in info or not info["gender"]:
        pron = re.search(r"\b(he|she)\b", text, re.I)
        if pron:
            info["gender"] = "male" if pron.group(1).lower() == "he" else "female"

    return info


def infer_gender_from_name(name: str) -> str:
    if not name:
        return ""
    first = name.split()[0].lower()
    female = {"jessica", "emily", "sarah", "olivia", "emma", "sophia", "isabella", "ava", "mia", "ella", "jess"}
    male = {"mike", "michael", "john", "james", "robert", "william", "david", "daniel", "matthew", "joseph"}
    if first in female: return "female"
    if first in male: return "male"
    return ""


def clamp_turns():
    if len(patient_state["turns"]) > MAX_TURNS:
        patient_state["turns"] = patient_state["turns"][-MAX_TURNS:]


def chat_once(msgs, **kwargs):
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=msgs,
        **kwargs
    )
    return resp.choices[0].message.content.strip()


# -------------------- CORE ROUTES --------------------

@app.route("/")
def home():
    return render_template("index.html")


# -------------------- Start Session (PATIENT SPEAKS FIRST) --------------------

@app.route("/start-session", methods=["POST"])
def start_session():
    global case_context, patient_state

    system_prompt = f"""
You are the patient. Begin the OSCE conversation naturally.
Say 1–2 sentences like:
- "Hi, I’m not feeling well today."
- "Hello… I had some questions about my medication."
- "Hi… I’ve been having this issue."

PERSONA: {case_context['persona']}
FACTS: {case_context['facts']}
BACKGROUND: {case_context['summary']}
"""

    greeting = chat_once(
        [{"role": "system", "content": system_prompt}],
        temperature=0.5,
        max_tokens=60
    )

    patient_state["turns"] = [{"role": "assistant", "content": greeting}]

    return jsonify({"greeting": greeting})


# -------------------- Upload --------------------

@app.route("/upload", methods=["POST"])
def upload_case():
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
            {"role": "system", "content": "Write a brief first-person patient background (1–2 sentences)."},
            {"role": "user", "content": text}
        ],
        temperature=0.3
    )

    persona = chat_once(
        [
            {"role": "system", "content": "Describe the patient's tone in <=2 short lines."},
            {"role": "user", "content": text}
        ],
        temperature=0.5
    )

    summary_prompt = [
        {"role": "system", "content": "Extract a 1–2 sentence OSCE case summary."},
        {"role": "user", "content": text}
    ]

    case_summary = chat_once(summary_prompt, temperature=0.3)
    references = extract_references(text)


    case_context = {
        "raw": text,
        "facts": facts,
        "summary": summary,
        "persona": persona,
        "gender": (facts.get("gender") or "").lower()
    }
    patient_state = {
        "summary": "",
        "turns": []
    }

    # Push summary into chat state
    patient_state["turns"].append({
        "role": "assistant",
        "content": summary
    })

    # Push references into chat state (if present)
    if references:
        patient_state["turns"].append({
            "role": "assistant",
            "content": f"References: {references}"
        })

    # ✅ ALWAYS return a response
    return jsonify({
        "message": "Case uploaded successfully.",
        "case_summary": case_summary,
        "summary": summary,
        "persona": persona,
        "extracted": facts,
        "references": references or ""
    })




# -------------------- ASK --------------------

@app.route("/ask", methods=["POST"])
def ask():
    global case_context, patient_state

    data = request.get_json(silent=True) or {}
    user_q = (data.get("question") or "").strip()

    if not user_q:
        return jsonify({"error": "No question"}), 400

    turns_preview = patient_state["turns"][-6:]

    system_prompt = f"""
You are the patient in a pharmacy OSCE.
Answer briefly, naturally, 1–2 sentences max.
Stay in first-person only.

PERSONA: {case_context['persona']}
FACTS: {case_context['facts']}
BACKGROUND: {case_context['summary']}
"""

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(turns_preview)
    messages.append({"role": "user", "content": user_q})

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.4,
        max_tokens=60
    )
    answer = completion.choices[0].message.content.strip()

    patient_state["turns"].append({"role": "user", "content": user_q})
    patient_state["turns"].append({"role": "assistant", "content": answer})
    clamp_turns()

    return jsonify({"answer": answer})


# -------------------- TTS (WORKING NEW SDK VERSION) --------------------

@app.route("/tts", methods=["POST"])
def tts():
    text = request.json.get("text", "").strip()
    if not text:
        return jsonify({"error": "No text provided"}), 400

    gender = case_context.get("gender", "")
    voice_choice = "alloy" if gender == "female" else "verse"

    audio_filename = f"voice_{uuid.uuid4().hex}.mp3"
    audio_path = os.path.join(app.config["UPLOAD_FOLDER"], audio_filename)

    with client.audio.speech.with_streaming_response.create(
        model="gpt-4o-mini-tts",
        voice=voice_choice,
        input=text
    ) as r:
        r.stream_to_file(audio_path)

    return jsonify({"audio": f"/uploads/{audio_filename}", "ready": True})

# -------------------- LIST CHAPTERS --------------------

@app.route("/list-chapters")
def list_chapters():
    base_path = "Chapters"   # Make sure this folder is in the project root

    if not os.path.exists(base_path):
        return jsonify({"error": "Chapters folder not found"}), 404

    result = {}

    # Loop through items in the Chapters folder
    for chapter in os.listdir(base_path):
        chapter_path = os.path.join(base_path, chapter)

        # Only accept folders (Chapter 1, Chapter 2, etc.)
        if os.path.isdir(chapter_path):

            files = []
            for f in os.listdir(chapter_path):
                if f.lower().endswith((".txt", ".pdf", ".docx")):
                    files.append(f)

            # Add to dictionary
            result[chapter] = files

    return jsonify(result)
# -------------------- LOAD DEFAULT CASE --------------------

@app.route("/load-default-case", methods=["POST"])
def load_default_case():
    global case_context, patient_state

    data = request.get_json()
    chapter = data.get("chapter")
    filename = data.get("file")
    
    if not chapter or not filename:
        return jsonify({"error": "Invalid request"}), 400

    full_path = os.path.join("Chapters", chapter, filename)

    if not os.path.exists(full_path):
        return jsonify({"error": "Case file not found"}), 404

    # Read file content using your extract_text() function
    try:
        text = extract_text(full_path)
    except:
        return jsonify({"error": "Failed to read file"}), 500

    # Extract structured facts
    facts = extract_case_info(text)

    # Infer gender if not present
    if "gender" not in facts or not facts["gender"]:
        inferred = infer_gender_from_name(facts.get("name", ""))
        if inferred:
            facts["gender"] = inferred

    # Generate patient background
    summary = chat_once(
        [
            {"role": "system", "content": "Write a brief first-person patient background (1–2 sentences)."},
            {"role": "user", "content": text}
        ],
        temperature=0.3
    )

    # Tone/persona
    persona = chat_once(
        [
            {"role": "system", "content": "Describe the patient's tone in <=2 short lines."},
            {"role": "user", "content": text}
        ],
        temperature=0.5
    )

    # Case Summary
    case_summary = chat_once(
        [
            {"role": "system", "content": "Extract a 1–2 sentence OSCE case summary."},
            {"role": "user", "content": text}
        ],
        temperature=0.3
    )
    references = extract_references(text)


    # Save context
    case_context = {
        "raw": text,
        "facts": facts,
        "summary": summary,
        "persona": persona,
        "gender": (facts.get("gender") or "").lower()
    }
    patient_state = {"summary": "", "turns": []}

    return jsonify({
        "case_summary": case_summary,
        "summary": summary,
        "persona": persona,
        "extracted": facts,
        "references": references
    })

@app.route("/auto-greet", methods=["POST"])
def auto_greet():
    global case_context, patient_state

    system_prompt = f"""
You are the patient. Provide a simple greeting, 1 sentence.
Examples:
- "Hi, I'm here because I'm not feeling well today."
- "Hello, I had some concerns about my medication."
- "Hi, I’ve been having this issue and wanted to ask for advice."

PERSONA: {case_context['persona']}
FACTS: {case_context['facts']}
BACKGROUND: {case_context['summary']}
"""

    greeting = chat_once(
        [{"role": "system", "content": system_prompt}],
        temperature=0.5,
        max_tokens=40
    )

    patient_state["turns"] = [{"role": "assistant", "content": greeting}]

    return jsonify({"greeting": greeting})


@app.route("/uploads/<path:filename>")
def serve_upload(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


# -------------------- Run --------------------

if __name__ == "__main__":
    app.run(debug=True)
