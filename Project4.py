from flask import Flask, render_template, request, jsonify
import json
import torch
import faiss
import numpy as np
from huggingface_hub import login
from langchain.docstore.document import Document
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
import pandas as pd
from datetime import datetime, timedelta
from difflib import SequenceMatcher
import string
import re
import pandas as pd
import nltk
from nltk.corpus import words
import re
from rapidfuzz import fuzz
import sqlite3

# -------------------------------
# Gibberish Detection
# -------------------------------

nltk.download('words')
english_words = set(words.words())

indian_names = {
    "kaarthik", "narain", "yazhini", "ananya", "sivakumar", "anand", "rajesh",
    "albert", "arjun", "priya", "deepak", "krishna", "lakshmi", "manoj", "rohini",
    "ganesh", "vignesh", "naveen", "ramesh", "arun", "anitha", "gopal", "sundar",
    "sathish", "bharathi", "saravanan", "kavya", "devi", "hari", "varun", "dinesh",
    "nandhini", "subash", "meena", "revathi", "gowtham", "karthikeyan","anandhi",
    "vijay", "jagath", "ashwin", "guru", "surya","Belina","Anu","Jeeva","Sathya","Kavin",
    "Sakthi","Prakash","Vimal","Nithish","Kavin","Pradeep","Sanjay","Ravi","Ajith","Kumar","Senthil","Rajeshwari","Lakshmanan","Mohan","Suresh","Vijaya","Kalyani","Nandha","Sangeetha","Divya","Pooja","Rajini","Sowmya","Nithya","Harish","Vasanth","Ashok","Kishore","Raghav","Yash","Aditya","Aravind","Chitra","Bhavani","Janani","Keerthi","Latha","Madhavi","Nisha","Pallavi","Radha","Sahana","Tamilarasi","Usha","Vijayalakshmi"
}
indian_names = {name.lower() for name in indian_names}
def is_gibberish_word(word):
    word = word.lower()
    # Ignore very short words
    if len(word) <= 1:
        return False
    # Mostly symbols/numbers → gibberish
    letter_count = sum(c.isalpha() for c in word)
    if letter_count / max(len(word), 1) < 0.5:
        return True
    # If in dictionary → not gibberish
    if word in english_words:
        return False
    if word.lower() in indian_names:
        return False
    # Not in dictionary → possible gibberish
    return True

def is_gibberish(text, threshold=0.5):
    words_list = re.findall(r'\b\w+\b', text)
    if not words_list:
        return True
    gibberish_count = sum(is_gibberish_word(w) for w in words_list)
    ratio = gibberish_count / len(words_list)
    return ratio > threshold

# -------------------------------
# Timetable Data Placeholder
# -------------------------------

timetable = None  # Will be loaded dynamically after admin uploads a file

def answer_from_timetable(query):
    global timetable

    if timetable is None or timetable.empty:
        return "⚠️ Timetable not uploaded yet. Please ask admin to upload it."

    q = query.lower()

    # Identify department
    departments = timetable['department'].str.lower().unique()
    dept = next((d for d in departments if d in q), None)

    # Identify year
    year_match = re.search(r"\b(\d)(?:st|nd|rd|th)? year\b", q)
    year = int(year_match.group(1)) if year_match else None

    # Identify subject or code
    for _, row in timetable.iterrows():
        if row["subject"].lower() in q or row["code"].lower() in q:
            return (f"{row['subject']} exam for {row['department']} (Year {row['year']}) "
                    f"is on {row['date']} from {row['time']}.")

    # If department + year are mentioned → show all their exams
    if dept and year:
        df = timetable[
            (timetable["department"].str.lower() == dept) &
            (timetable["year"] == year)
        ]
        if not df.empty:
            result = "\n".join([
                f"📘 {r['subject']} ({r['code']}) — {r['date']} | {r['time']}"
                for _, r in df.iterrows()
            ])
            return f"Here’s the exam schedule for {dept.upper()} - Year {year}:\n\n{result}"

    return "I couldn’t find that in the timetable."

def check_timetable(user_message):
    q = user_message.lower()
    dept_match = re.search(r"(mca gen ai|cse|ece|eee|mech|civil|mba)", q)
    dept = dept_match.group(0) if dept_match else None
    year_match = re.search(r"\b(\d)(?:st|nd|rd|th)? year\b", q)
    year = int(year_match.group(1)) if year_match else None

    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()

    # Dept + year → all exams
    if dept and year:
        c.execute("SELECT subject, code, date, time FROM timetable WHERE LOWER(department)=? AND year=? ORDER BY date", (dept, year))
        rows = c.fetchall()
        conn.close()
        if rows:
            result = "\n".join([f"📘 {r[0]} ({r[1]}) — {r[2]} | {r[3]}" for r in rows])
            return f"Here’s the exam schedule for {dept.upper()} - Year {year}:\n\n{result}"
        else:
            return "No timetable found for the provided department and year."

    # Subject or code search (using LIKE for partial matches)
    for keyword in re.findall(r"\b\w+\b", q):
        c.execute("""
            SELECT department, year, date, time, subject FROM timetable
            WHERE LOWER(subject) LIKE ? OR LOWER(code) LIKE ?
        """, (f"%{keyword}%", f"%{keyword}%"))
        row = c.fetchone()
        if row:
            conn.close()
            from datetime import datetime
            exam_date = datetime.strptime(row[2], "%Y-%m-%d").strftime("%d %b %Y")  # convert to 13 Oct 2025
            return f"{row[4]} exam for {row[0]} (Year {row[1]}) is on {exam_date} from {row[3]}."


# -------------------------------
# Seat Data Placeholder
# -------------------------------
seat_df = None  # Will be loaded dynamically after admin uploads a file

def load_seat_data(excel_file):
    """
    Loads seat data dynamically when admin uploads a file.
    """
    try:
        df = pd.read_excel(excel_file)
        df['Date'] = pd.to_datetime(df['Date']).dt.date  # Convert Excel dates to datetime.date
        print(f"✅ Seat data loaded successfully with {len(df)} entries!")
        return df
    except Exception as e:
        print(f"⚠️ Error loading seat data: {e}")
        return pd.DataFrame(columns=["Register number", "Session", "Date", "Seat Number"])

# -------------------------------
# Seat Checker Function
# -------------------------------
def check_seat_from_message(user_message):
    msg = user_message.upper()

    # Extract details
    reg_match = re.search(r'RA\d{13}', msg)
    reg_no = reg_match.group(0) if reg_match else None

    session_match = re.search(r'\b(FN|AN)\b', msg)
    session = session_match.group(0) if session_match else None

    date_match = re.search(r'(\d{1,2}/\d{1,2}/\d{4}|\d{4}-\d{2}-\d{2})', msg)
    if "TOMORROW" in msg:
        date = (datetime.now() + timedelta(days=1)).date().strftime('%Y-%m-%d')
    elif date_match:
        try:
            date = pd.to_datetime(date_match.group(0), dayfirst=False, errors='coerce').strftime('%Y-%m-%d')
        except:
            return "⚠️ Invalid date format. Use MM/DD/YYYY or YYYY-MM-DD."
    else:
        date = None

    missing = []
    if not reg_no:
        missing.append("register number")
    if not session:
        missing.append("session (FN/AN)")
    if not date:
        missing.append("date (MM/DD/YYYY or 'tomorrow')")

    if missing:
        return "Please provide your " + ", ".join(missing) + " to check your seat."

    # Query DB
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("""
        SELECT seat_number FROM seats
        WHERE register_number = ? AND session = ? AND date = ?
    """, (reg_no, session, date))
    row = c.fetchone()
    conn.close()

    if row:
        return f"🎟️ Seat for {reg_no} on {date} ({session}): **{row[0]}**"
    else:
        return "No seat details found for the provided information."


# -------------------------------
# Authenticate Hugging Face
# -------------------------------
# login("HF_TOKEN") replace HF_TOKEN with your actual token or set as env variable

import os
from huggingface_hub import login

login(token=os.getenv("HF_TOKEN"))


# -------------------------------
# Load JSON dataset for RAG
# -------------------------------
with open("srm_data.json", "r") as f:
    data = json.load(f)

docs = [Document(page_content=entry["completion"], metadata={"question": entry["prompt"]}) for entry in data]
documents = [d.page_content for d in docs]

# -------------------------------
# Embeddings + FAISS
# -------------------------------
embedder = SentenceTransformer("all-MiniLM-L6-v2")
doc_embeddings = embedder.encode(documents, convert_to_numpy=True)
dimension = doc_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(doc_embeddings)

# -------------------------------
# Load Model
# -------------------------------
try:
    model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        dtype=torch.float16
    )
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device_map="auto", dtype=torch.float16)
except Exception as e:
    from transformers import AutoModelForSeq2SeqLM
    model_name = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    generator = pipeline("text2text-generation", model=model, tokenizer=tokenizer)

# -------------------------------
# Normalize text
# -------------------------------
def normalize(text):
    return text.lower().translate(str.maketrans("", "", string.punctuation)).strip()

# -------------------------------
# Temperature Fetcher
# -------------------------------

import requests

API_KEY = "7b3b0c3ccdf5f72ec472153dc330dc61" 
CITY = "Chennai,IN"

def get_potheri_temperature():
    url = f"http://api.openweathermap.org/data/2.5/weather?q={CITY}&units=metric&appid={API_KEY}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        temp = data['main']['temp']
        description = data['weather'][0]['description']
        return f"The current temperature in SRM Potheri, Chennai is {temp}°C with {description}."
    else:
        return "Sorry, I couldn't fetch the weather for Potheri, Chennai."

# -------------------------------
# RAG Query
# -------------------------------
def rag_query(question, top_k=5):

    q_norm = normalize(question)

    seat_match = re.search(r'RA\d{13}', question.upper())
    session_match = re.search(r'\b(FN|AN)\b', question.upper())
    date_match = re.search(r'(\d{1,2}/\d{1,2}/\d{4}|\d{4}-\d{2}-\d{2})', question)

    if seat_match and session_match and date_match:
        return check_seat_from_message(question)

    if is_gibberish(question):
            return "Hmm… I cannot understand that! Could you rephrase that?"

    def is_timetable_query(q):
        # Year pattern
        year_match = re.search(r"\b(\d)(?:st|nd|rd|th)? year\b", q.lower())
        # Keywords
        keywords = ["exam", "timetable", "schedule", "subject", "paper", "date", "when is"]
        keyword_match = any(k in q.lower() for k in keywords)
        # Department or subject code
        dept_match = any(d.lower() in q.lower() for d in timetable['department'].unique()) if timetable is not None else False
        code_match = any(c.lower() in q.lower() for c in timetable['code'].unique()) if timetable is not None else False
        return year_match or keyword_match or dept_match or code_match

    if is_timetable_query(question):
        return check_timetable(question)

    static_responses = {
        "hi": "Hi there 👋! I’m your SRM AI Assistant. How can I help you today?",
        "hello": "Hello! 😊 I’m your friendly SRM AI Assistant — ready to assist you!",
        "hey": "Hey! 👋 What would you like to know about SRM?",
        "who are you": "I am an AI Assistant 🤖 — built to help you explore SRMIST!",
        "what is your name": "I’m SRM AI Assistant, your smart campus guide ✨.",
        "About yourself": "I’m an AI Assistant for SRM, created by MCA Generative AI 26 students Kaarthik Narain, Vijay, Jagath Ashwin, and Guru Surya under Dr. Sivakumar S's guidance.",
        "help": "Sure! I can assist you with information about SRMIST, exam timetables, seat allocations, and more. Just ask your question!",
        "thanks": "You’re welcome! 😊 If you have more questions, feel free to ask",
        "bye": "Goodbye! 👋 Have a great day ahead!",
        "good morning": "Good morning! ☀️ How can I assist you today?",
        "good afternoon": "Good afternoon! 🌞 What would you like to know about SRM?",
        "good evening": "Good evening! 🌆 How can I help you?",
        "good night": "Good night! 🌙 If you have any questions, just ask!",
        "time now": f"The current time is {datetime.now().strftime('%H:%M')}.",
        "date today": f"Today's date is {datetime.now().strftime('%B %d, %Y')}.",
        "thank you": "You're welcome! 😊 If you have more questions, feel free to ask.",
        "thanks and bye": "You're welcome! 👋 Have a great day ahead!"

    }

    if "temperature" in question.lower() or "weather" in question.lower():
        return get_potheri_temperature()


    for key, value in static_responses.items():
        if re.search(rf'\b{re.escape(key)}\b', q_norm):
            return value

        # --- Step 0: Exact match check first ---
    for entry in data:
        if normalize(entry["prompt"]) == normalize(question):
            return entry["completion"]

    # --- Step 1: Fuzzy matching using RapidFuzz ---
    best_score, best_answer = 0, None
    for entry in data:
        score = fuzz.partial_ratio(q_norm, normalize(entry["prompt"]))  # 0-100
        if score > best_score:
            best_score, best_answer = score, entry["completion"]

    if best_score > 70:  # Threshold can be tuned
        return best_answer


    q_emb = embedder.encode([question], convert_to_numpy=True)
    D, I = index.search(q_emb, top_k)
    retrieved_docs = list(dict.fromkeys([documents[i] for i in I[0]]))
    if len(D[0]) == 0 or D[0][0] > 2.1:
        return "I'm sorry, I can only answer questions related to SRM."
    context = "\n\n".join(retrieved_docs[:3])[:1500]

    prompt_text="""You are an AI assistant for SRM students. Answer the user's question using ONLY the context below. 
Do NOT copy any sentences verbatim. 
Rewrite the information in your own words, concisely and clearly. 
Use bullet points if needed. 
If the question is unrelated to SRM, reply: "I'm sorry, I can only answer questions related to SRM.

Context:
{context}

User question: {question}
Answer: """

    output = generator(prompt_text, max_new_tokens=200, temperature=0.7, top_p=0.95, do_sample=True,
                       pad_token_id=tokenizer.eos_token_id or tokenizer.pad_token_id)[0]["generated_text"]
    answer = output[len(prompt_text):].strip() if prompt_text in output else output.strip()
    return answer if answer else "I'm sorry, I can only answer questions related to SRM."

# -------------------------------
# Flask App
# -------------------------------
app = Flask(__name__)

DB_FILE = "srm_data.db"  # SQLite database file for timetable and seat data

# Ensure database tables exist
conn = sqlite3.connect(DB_FILE)
c = conn.cursor()

# Create timetable table
c.execute("""
CREATE TABLE IF NOT EXISTS timetable (
    department TEXT,
    year INTEGER,
    date TEXT,
    subject TEXT,
    code TEXT,
    time TEXT
)
""")

# Create seats table
c.execute("""
CREATE TABLE IF NOT EXISTS seats (
    register_number TEXT,
    session TEXT,
    date TEXT,
    seat_number TEXT
)
""")

conn.commit()
conn.close()


@app.route("/")
def home():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    question = data.get("question")
    answer = rag_query(question)
    return jsonify({"answer": answer})

from werkzeug.utils import secure_filename
import os

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
ALLOWED_EXTENSIONS = {"xlsx"}

DB_FILE = "srm_data.db"  # SQLite database file for timetable and seat data

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/upload_timetable", methods=["POST"])
def upload_timetable():
    file = request.files.get("file")
    if not file:
        return jsonify({"status": "error", "message": "No file uploaded"}), 400

    if not allowed_file(file.filename):
        return jsonify({"status": "error", "message": "Only .xlsx files are allowed"}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    try:
        df = pd.read_excel(filepath)
        df.columns = df.columns.str.lower().str.strip()

        expected_cols = {"department", "year", "date", "subject", "code", "time"}
        if not expected_cols.issubset(df.columns):
            raise ValueError(f"Missing columns: {expected_cols - set(df.columns)}")

        # Convert dates to string
        df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.strftime('%Y-%m-%d')

        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute("DELETE FROM timetable")  # remove old timetable
        for _, row in df.iterrows():
            c.execute("""
                INSERT INTO timetable (department, year, date, subject, code, time)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (row['department'], row['year'], row['date'], row['subject'], row['code'], row['time']))
        conn.commit()
        conn.close()

        return jsonify({"status": "success", "message": "Timetable uploaded successfully!"})
    except Exception as e:
        print(f"⚠️ Error uploading timetable: {e}")
        return jsonify({"status": "error", "message": f"Failed to upload timetable: {e}"}), 500



@app.route("/upload_seats", methods=["POST"])
def upload_seats():
    file = request.files.get("file")
    if not file:
        return jsonify({"status": "error", "message": "No file uploaded"}), 400

    if not allowed_file(file.filename):
        return jsonify({"status": "error", "message": "Only .xlsx files are allowed"}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    try:
        df = pd.read_excel(filepath)
        df['Date'] = pd.to_datetime(df['Date']).dt.date

        # Store in SQLite
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute("DELETE FROM seats")  # remove old seat data
        for _, row in df.iterrows():
            c.execute("""
                INSERT INTO seats (register_number, session, date, seat_number)
                VALUES (?, ?, ?, ?)
            """, (row['Register number'], row['Session'], row['Date'], row['Seat Number']))
        conn.commit()
        conn.close()

        print(f"✅ Seat data uploaded to DB with {len(df)} rows!")
        return jsonify({"status": "success", "message": "Seat data updated successfully!"})
    except Exception as e:
        print(f"⚠️ Error uploading seat data: {e}")
        return jsonify({"status": "error", "message": f"Failed to upload seat data: {e}"}), 500


@app.route("/feedback", methods=["POST"])
def feedback():
    data = request.get_json()
    fb = data.get("feedback")
    message = data.get("message")
    with open("feedback_logs.csv", "a") as f:
        f.write(f"{datetime.now()},{fb},{message}\n")
    return jsonify({"status": "ok"})

import csv
from datetime import datetime

FEEDBACK_FILE = "feedback.csv"

try:
    with open(FEEDBACK_FILE, "x", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "assistant_message", "feedback"])
except FileExistsError:
    pass

@app.route("/feedback", methods=["POST"])
def handle_feedback():
    data = request.get_json()
    message = data.get("message")
    feedback = data.get("feedback")

    if not message or not feedback:
        return jsonify({"status": "error", "message": "Missing data"}), 400

    with open(FEEDBACK_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        writer.writerow([timestamp, message, feedback])

    return jsonify({"status": "success"})

@app.route("/clear_data", methods=["POST"])
def clear_data():
    data_type = request.form.get("type")  # "timetable" or "seats"

    if data_type not in ["timetable", "seats"]:
        return jsonify({"status": "error", "message": "Invalid data type"}), 400

    try:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute(f"DELETE FROM {data_type}")  # clears the table
        conn.commit()
        conn.close()
        return jsonify({"status": "success", "message": f"{data_type.capitalize()} data cleared successfully!"})
    except Exception as e:
        return jsonify({"status": "error", "message": f"Failed to clear data: {e}"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
