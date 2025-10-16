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
import dateparser

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
        if row["subject"].lower() in q.lower() or row["code"].lower() in q.lower() or q.lower() in row["subject"].lower():
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

# Load embedding model (only once)
embedder = SentenceTransformer('all-MiniLM-L6-v2')

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import sqlite3
from datetime import datetime
import numpy as np

# Load once
embedder = SentenceTransformer('all-MiniLM-L6-v2')

from rapidfuzz import fuzz
import re
from datetime import datetime
import sqlite3

# Add your acronym mapping here
SUBJECT_ACRONYMS = {
    "ooad": "object oriented analysis and design",
    "ai": "artificial intelligence",
    "cas": "cognitive analytical skills",
    "eh": "ethical hacking",
    "fsd": "full stack development"
}

from rapidfuzz import fuzz
import re
import sqlite3
from datetime import datetime

# Add your acronym mapping here
SUBJECT_ACRONYMS = {
    "ooad": "object oriented analysis and design",
    "ai": "artificial intelligence",
    "cas": "cognitive analytical skills",
    "eh": "ethical hacking",
    "fsd": "full stack development"
}

def check_timetable(user_message):
    """
    Returns timetable info based on user query.
    Handles:
    - Full subject match
    - Partial/fuzzy match
    - Acronym mapping
    - Department + year queries
    - Department-only queries
    """
    q = user_message.lower().strip()
    
    # Step 0: Replace acronyms dynamically
    for acr, full in SUBJECT_ACRONYMS.items():
        q = re.sub(rf"\b{acr}\b", full, q)
    
    # Step 1: Fetch all timetable rows
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT department, year, date, time, subject, code FROM timetable")
    rows = c.fetchall()
    conn.close()

    # Step 2: Exact subject match
    for r in rows:
        if q == r[4].lower():
            exam_date = datetime.strptime(r[2], "%Y-%m-%d").strftime("%d %b %Y")
            return f"{r[4]} exam for {r[0]} (Year {r[1]}) is on {exam_date} from {r[3]}."

    # Step 3: Partial/fuzzy match on subject or code
    best_score, best_row = 0, None
    for r in rows:
        score_subj = fuzz.partial_ratio(q, r[4].lower())
        score_code = fuzz.partial_ratio(q, r[5].lower())
        max_score = max(score_subj, score_code)
        if max_score > best_score:
            best_score = max_score
            best_row = r

    if best_score > 70:
        r = best_row
        exam_date = datetime.strptime(r[2], "%Y-%m-%d").strftime("%d %b %Y")
        return f"{r[4]} exam for {r[0]} (Year {r[1]}) is on {exam_date} from {r[3]}."

    # Step 4: Department + year query
    dept_match = None
    year_match = None
    departments = sorted(set([r[0].lower() for r in rows]), key=lambda x: -len(x))
    
    # Try exact word match first
    for d in departments:
        if f" {d} " in f" {q} ":
            dept_match = d
            break
    # Fallback: partial match
    if not dept_match:
        for d in departments:
            if d in q:
                dept_match = d
                break

    # Year extraction
    ym = re.search(r"\b(\d)(?:st|nd|rd|th)? year\b", q)
    if ym:
        year_match = int(ym.group(1))

    # Dept + year
    if dept_match and year_match:
        matching_rows = [r for r in rows if r[0].lower() == dept_match and r[1] == year_match]
        if matching_rows:
            # Try fuzzy match within filtered rows first
            best_score, best_row = 0, None
            for r in matching_rows:
                score_subj = fuzz.partial_ratio(q, r[4].lower())
                score_code = fuzz.partial_ratio(q, r[5].lower())
                max_score = max(score_subj, score_code)
                if max_score > best_score:
                    best_score = max_score
                    best_row = r
            if best_row and best_score >= 50:
                r = best_row
                exam_date = datetime.strptime(r[2], "%Y-%m-%d").strftime("%d %b %Y")
                return f"{r[4]} exam for {r[0]} (Year {r[1]}) is on {exam_date} from {r[3]}."
            
            # Fallback: list all exams for dept+year
            result = "\n".join([
                f"📘 {r[4]} ({r[5]}) — {datetime.strptime(r[2], '%Y-%m-%d').strftime('%d %b %Y')} | {r[3]}"
                for r in matching_rows
            ])
            return f"Here’s the exam schedule for {dept_match.upper()} - Year {year_match}:\n\n{result}"

    # Step 5: Department-only (no year)
    if dept_match:
        matching_rows = [r for r in rows if dept_match in r[0].lower()]
        if matching_rows:
            result = "\n".join([
                f"📘 {r[4]} ({r[5]}) — {datetime.strptime(r[2], '%Y-%m-%d').strftime('%d %b %Y')} | {r[3]}"
                for r in matching_rows
            ])
            return f"Here’s the exam schedule for {dept_match.upper()}:\n\n{result}"

    return "I couldn’t find that in the timetable. Please check the subject name, code, or department/year."



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

import re
import sqlite3
import dateparser
from datetime import datetime

# -------------------------------
# Flexible Session Mapping
# -------------------------------
SESSION_MAP = {
    "FN": ["FN", "forenoon", "morning"],
    "AN": ["AN", "afternoon", "evening"]
}

def parse_session(msg):
    """Return standard session code 'FN' or 'AN' based on flexible terms."""
    msg = msg.lower()
    for standard, variants in SESSION_MAP.items():
        for v in variants:
            if v.lower() in msg:
                return standard
    return None

def parse_date(msg):
    """Handles natural language dates using dateparser and outputs YYYY-MM-DD."""
    dt = dateparser.parse(msg, settings={'PREFER_DATES_FROM': 'future'})
    if dt:
        return dt.date().strftime('%Y-%m-%d')
    
    # Fallback for MM/DD/YY or MM/DD/YYYY
    date_match = re.search(r'(\d{1,2}/\d{1,2}/\d{2,4}|\d{4}-\d{2}-\d{2})', msg)
    if date_match:
        try:
            return datetime.strptime(date_match.group(0), "%m/%d/%y").strftime('%Y-%m-%d')
        except:
            try:
                return datetime.strptime(date_match.group(0), "%m/%d/%Y").strftime('%Y-%m-%d')
            except:
                try:
                    return datetime.strptime(date_match.group(0), "%Y-%m-%d").strftime('%Y-%m-%d')
                except:
                    return None
    return None

# -------------------------------
# Updated Seat Checker
# -------------------------------
DB_FILE = "srm_data.db"  # Make sure your SQLite DB file

def check_seat_from_message(user_message):
    """
    Check seat for a student from a user message.
    Handles:
    - Flexible session names (FN, AN, morning, afternoon)
    - Natural language dates
    - Standard register numbers
    """
    msg = user_message.strip()

    # 1️⃣ Extract register number
    reg_match = re.search(r'RA\d{13}', msg.upper())
    reg_no = reg_match.group(0) if reg_match else None

    # 2️⃣ Extract session
    session = parse_session(msg)

    # 3️⃣ Extract date
    date = parse_date(msg)

    # 4️⃣ Check missing info
    missing = []
    if not reg_no:
        missing.append("register number")
    if not session:
        missing.append("session (FN/AN or morning/afternoon)")
    if not date:
        missing.append("date (MM/DD/YYYY, YYYY-MM-DD, or natural language like 'tomorrow', 'next Friday')")

    if missing:
        return "Please provide your " + ", ".join(missing) + " to check your seat."

    # 5️⃣ Query SQLite DB
    try:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute("""
            SELECT seat_number FROM seats
            WHERE register_number = ? AND session = ? AND date = ?
        """, (reg_no, session, date))
        row = c.fetchone()
        conn.close()
    except Exception as e:
        return f"⚠️ Database error: {e}"

    # 6️⃣ Return seat info
    if row:
        return f"🎟️ Seat for {reg_no} on {date} ({session}): **{row[0]}**"
    else:
        return "No seat details found for the provided information."

# -------------------------------
# Authenticate Hugging Face
# -------------------------------
login("YOUR_HUGGINGFACE_TOKEN")

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

    # 1️⃣ Extract seat info
    reg_match = re.search(r'RA\d{13}', question.upper())
    reg_no = reg_match.group(0) if reg_match else None
    session = parse_session(question)
    date = parse_date(question)

    # If all present → check seat
    if reg_no and session and date:
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

@app.route('/team')
def team():
    return render_template('team.html')

# Create announcements table
c.execute("""
CREATE TABLE IF NOT EXISTS announcements (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    message TEXT NOT NULL,
    date TEXT NOT NULL,
    urgent INTEGER DEFAULT 0
)
""")

conn.commit()
conn.close()

# Fetch all announcements
@app.route("/get_announcements")
def get_announcements():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT id, title, message, date, urgent FROM announcements ORDER BY id DESC")
    rows = c.fetchall()
    conn.close()

    announcements = [
        {"id": r[0], "title": r[1], "message": r[2], "date": r[3], "urgent": bool(r[4])} 
        for r in rows  # use the fetched rows, not c.fetchall() again
    ]
    return jsonify(announcements)

# Admin adds a new announcement
@app.route("/admin/add_news", methods=["POST"])
def add_news():
    data = request.get_json()
    title = data.get("title")
    message = data.get("message")
    date = data.get("date")
    urgent = 1 if data.get("urgent") else 0

    if not title or not message or not date:
        return jsonify({"status":"error", "message":"All fields are required!"}), 400

    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("INSERT INTO announcements (title, message, date, urgent) VALUES (?, ?, ?, ?)",
              (title, message, date, urgent))
    conn.commit()
    conn.close()
    return jsonify({"status": "success", "message": "Announcement added!"})

# Delete a news item
@app.route("/admin/delete_news/<int:news_id>", methods=["DELETE"])
def delete_news(news_id):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("DELETE FROM announcements WHERE id = ?", (news_id,))
    conn.commit()
    conn.close()
    return jsonify({"status": "success", "message": "Announcement deleted!"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
