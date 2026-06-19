import os
import re
import json
import base64
import sqlite3
import tempfile
from datetime import date, timedelta
from urllib.parse import quote

import joblib
import pandas as pd
import qrcode
import streamlit as st
import yagmail
from gtts import gTTS
from dotenv import load_dotenv
from groq import Groq
from streamlit_mic_recorder import mic_recorder

try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None


st.set_page_config(page_title="AI Hotel Agent", page_icon="🏨", layout="wide")
load_dotenv()

GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY"))
GROQ_MODEL = st.secrets.get("GROQ_MODEL", os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"))
GROQ_VISION_MODEL = st.secrets.get("GROQ_VISION_MODEL", os.getenv("GROQ_VISION_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct"))
GROQ_STT_MODEL = st.secrets.get("GROQ_STT_MODEL", os.getenv("GROQ_STT_MODEL", "whisper-large-v3-turbo"))

EMAIL_USER = st.secrets.get("EMAIL_USER", os.getenv("EMAIL_USER"))
EMAIL_APP_PASSWORD = st.secrets.get("EMAIL_APP_PASSWORD", os.getenv("EMAIL_APP_PASSWORD"))
UPI_ID = st.secrets.get("UPI_ID", os.getenv("UPI_ID", "madhu@@ybl"))

DB_NAME = "hotel_streamlit.db"


def get_connection():
    return sqlite3.connect(DB_NAME, check_same_thread=False)


def create_tables():
    conn = get_connection()
    c = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS rooms (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            room_type TEXT NOT NULL,
            price REAL NOT NULL
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS bookings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            guest_name TEXT,
            email TEXT,
            phone TEXT,
            room_type TEXT,
            check_in TEXT,
            check_out TEXT,
            guests INTEGER,
            total_price REAL,
            payment_status TEXT DEFAULT 'Pending',
            transaction_id TEXT
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS reviews (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            guest_name TEXT,
            review_text TEXT,
            sentiment TEXT
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS food_orders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            guest_name TEXT,
            room_number TEXT,
            food_item TEXT,
            quantity INTEGER,
            status TEXT DEFAULT 'Pending'
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS service_requests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            guest_name TEXT,
            room_number TEXT,
            service_type TEXT,
            message TEXT,
            status TEXT DEFAULT 'Pending'
        )
    """)

    c.execute("""
        CREATE INDEX IF NOT EXISTS idx_booking_room_dates
        ON bookings(room_type, check_in, check_out)
    """)

    conn.commit()
    conn.close()


def insert_default_rooms():
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM rooms")

    if c.fetchone()[0] == 0:
        rooms = [
            ("Standard Single", 1800),
            ("Standard Double", 2500),
            ("Deluxe Single", 3500),
            ("Deluxe Double", 4500),
            ("Executive Room", 5500),
            ("Business Room", 6000),
            ("Family Room", 6500),
            ("Garden View Room", 7000),
            ("Suite Room", 8000),
            ("Ocean View Room", 9000),
            ("Luxury Suite", 10000),
            ("Honeymoon Suite", 12000),
            ("Presidential Suite", 15000),
            ("Penthouse Suite", 20000)
        ]

        c.executemany(
            "INSERT INTO rooms (room_type, price) VALUES (?, ?)",
            rooms
        )

    conn.commit()
    conn.close()


create_tables()
insert_default_rooms()


@st.cache_resource
def load_models():
    room_model = None
    sentiment_model = None

    if os.path.exists("room_recommendation_model.pkl"):
        room_model = joblib.load("room_recommendation_model.pkl")

    if os.path.exists("sentiment_model.pkl"):
        sentiment_model = joblib.load("sentiment_model.pkl")

    chatbot_model = None
    if os.path.exists("chatbot_intent_model.pkl"):
        chatbot_model = joblib.load("chatbot_intent_model.pkl")

    return room_model, sentiment_model, chatbot_model


room_model, sentiment_model, chatbot_model = load_models()


def safe_image(path, caption=None):
    if os.path.exists(path):
        st.image(path, caption=caption, width="stretch")
    else:
        st.warning(f"Image missing: {path}")

def get_room_image(room_type):
    room_type = room_type.lower()

    if "standard single" in room_type:
        return "assets/standard_single.jpg"

    elif "standard double" in room_type:
        return "assets/standard.jpg"

    elif "deluxe single" in room_type:
        return "assets/deluxe_single.jpg"

    elif "deluxe double" in room_type:
        return "assets/deluxe.jpg"

    elif "suite single" in room_type:
        return "assets/suite_single.jpg"

    elif "suite double" in room_type:
        return "assets/suite.jpg"

    elif "executive" in room_type:
        return "assets/deluxe.jpg"

    elif "business" in room_type:
        return "assets/deluxe.jpg"

    elif "family" in room_type:
        return "assets/suite.jpg"

    elif "garden" in room_type:
        return "assets/suite.jpg"

    elif "ocean" in room_type:
        return "assets/suite.jpg"

    elif "honeymoon" in room_type:
        return "assets/suite.jpg"

    elif "presidential" in room_type:
        return "assets/suite.jpg"

    elif "penthouse" in room_type:
        return "assets/suite.jpg"

    elif "suite" in room_type:
        return "assets/suite.jpg"

    else:
        return "assets/standard.jpg"
def show_rooms():
    conn = get_connection()
    df = pd.read_sql_query("SELECT room_type, price FROM rooms", conn)
    conn.close()
    return df


def closest_room_type(text):
    rooms = show_rooms()["room_type"].tolist()
    text = str(text).lower()

    for room in rooms:
        if room.lower() == text:
            return room

    if "deluxe " in text:
        return "Deluxe Double"
    if "standard" in text:
        return "Standard Double"
    if "suite" in text:
        return "Suite Room"
    if "family" in text:
        return "Family Room"
    if "business" in text:
        return "Business Room"

    return None


def recommend_room_from_chat(budget, guests, stay_days):

    if room_model is None:
        return "Room recommendation model not found."

    input_df = pd.DataFrame(
        [[budget, guests, stay_days]],
        columns=["budget", "guests", "stay_days"]
    )

    return room_model.predict(input_df)[0]

def analyze_review_from_chat(review_text):

    if sentiment_model is None:
        return "Sentiment model not found."

    result = sentiment_model.predict([review_text])[0]



    conn = get_connection()
    conn.execute(
        "INSERT INTO reviews (guest_name, review_text, sentiment) VALUES (?, ?, ?)",
        ("Chat User", review_text, result)
    )
    conn.commit()
    conn.close()

    return result


def send_booking_email(to_email, guest_name, room_type, check_in, check_out, total_price):
    if not EMAIL_USER or not EMAIL_APP_PASSWORD:
        return False

    body = f"""
Dear {guest_name},

Your room booking is confirmed.

Room Type: {room_type}
Check-in Date: {check_in}
Check-out Date: {check_out}
Total Amount: ₹{total_price}

Thank you for choosing your hotel.

Regards,
Hotel Team
"""

    try:
        yag = yagmail.SMTP(EMAIL_USER, EMAIL_APP_PASSWORD)
        yag.send(
            to=to_email,
            subject="Hotel Room Booking Confirmation",
            contents=body
        )
        return True
    except Exception as e:
        st.warning(f"Email not sent: {e}")
        return False


def book_room_from_chat(guest_name, email, phone, room_type, check_in, check_out, guests):
    if not guest_name or not email or not phone:
        return "Missing guest name, email, or phone."

    if "@" not in email:
        return "Invalid email address."

    if not phone.isdigit() or len(phone) < 10:
        return "Invalid phone number."

    room_type = closest_room_type(room_type)

    if not room_type:
        return "Room type not found. Type `show rooms` to see room names."

    conn = get_connection()
    c = conn.cursor()

    room = c.execute(
        "SELECT price FROM rooms WHERE room_type = ?",
        (room_type,)
    ).fetchone()

    if room is None:
        conn.close()
        return "Room type not found."

    try:
        days = (pd.to_datetime(check_out) - pd.to_datetime(check_in)).days
    except Exception:
        conn.close()
        return "Invalid date format. Use YYYY-MM-DD."

    if days <= 0:
        conn.close()
        return "Check-out date must be after check-in date."

    existing_booking = c.execute("""
        SELECT COUNT(*)
        FROM bookings
        WHERE room_type = ?
        AND NOT (
            check_out <= ?
            OR check_in >= ?
        )
    """, (room_type, check_in, check_out)).fetchone()[0]

    if existing_booking > 0:
        conn.close()
        return "This room is already booked for selected dates."

    total_price = room[0] * days

    c.execute("""
        INSERT INTO bookings
        (guest_name, email, phone, room_type, check_in, check_out, guests, total_price)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        guest_name, email, phone, room_type,
        check_in, check_out, guests, total_price
    ))

    conn.commit()
    conn.close()

    email_sent = send_booking_email(
        email, guest_name, room_type, check_in, check_out, total_price
    )

    status = "Confirmation email sent." if email_sent else "Email not sent."
    return f"Room booked successfully. Room: {room_type}, Total Amount: ₹{total_price}. {status}"


def place_food_order(guest_name, room_number, food_item, quantity):
    conn = get_connection()
    c = conn.cursor()

    c.execute("""
        INSERT INTO food_orders
        (guest_name, room_number, food_item, quantity)
        VALUES (?, ?, ?, ?)
    """, (guest_name, room_number, food_item, quantity))

    conn.commit()
    conn.close()

    return f"Food order placed: {quantity} x {food_item} for room {room_number}."


def create_service_request(guest_name, room_number, service_type, message):
    conn = get_connection()
    c = conn.cursor()

    c.execute("""
        INSERT INTO service_requests
        (guest_name, room_number, service_type, message)
        VALUES (?, ?, ?, ?)
    """, (guest_name, room_number, service_type, message))

    conn.commit()
    conn.close()

    return f"Room service request created for room {room_number}: {service_type}."


def create_upi_qr(amount):
    UPI_ID = os.getenv("UPI_ID", "yourmadhu@ybl")
    name = "NM Hotels"

    upi_link = (
        f"upi://pay?pa={UPI_ID}"
        f"&pn={quote(name)}"
        f"&am={amount}"
        f"&cu=INR"
    )

    qr_img = qrcode.make(upi_link)
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    qr_img.save(temp_file.name)
    return temp_file.name


def analyze_image_with_groq(uploaded_file, user_question):
    if not GROQ_API_KEY:
        return "Groq API Key not found."

    image_bytes = uploaded_file.getvalue()
    image_base64 = base64.b64encode(image_bytes).decode("utf-8")

    client = Groq(api_key=GROQ_API_KEY)

    response = client.chat.completions.create(
        model=GROQ_VISION_MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_question},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{uploaded_file.type};base64,{image_base64}"
                        }
                    }
                ]
            }
        ],
        max_tokens=500
    )

    return response.choices[0].message.content


def extract_pdf_text(uploaded_file):
    if PdfReader is None:
        return "PDF library not installed. Run: py -m pip install pypdf"

    try:
        reader = PdfReader(uploaded_file)
        text = ""

        for page in reader.pages:
            text += page.extract_text() or ""

        return text[:5000]

    except Exception as e:
        return f"PDF extraction error: {e}"


def convert_voice_to_text(audio):
    try:
        audio_bytes = audio["bytes"]

        temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".webm")
        temp_audio.write(audio_bytes)
        temp_audio.close()

        client = Groq(api_key=GROQ_API_KEY)

        with open(temp_audio.name, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                file=audio_file,
                model=GROQ_STT_MODEL
            )

        return transcription.text

    except Exception as e:
        return f"Voice-to-text error: {e}"


def speak_response(text):
    try:
        tts = gTTS(text)
        temp_mp3 = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tts.save(temp_mp3.name)
        return temp_mp3.name
    except Exception:
        return None


def detect_action(user_text):
    text = user_text.lower()

    if (
        any(word in text for word in [
            "book", "booking", "reserve", "reservation",
            "need a room", "want a room", "deluxe room",
            "standard room", "suite room", "room for"
        ])
    ):
        return "BOOK_ROOM"

    if "food" in text or "biryani" in text or "order" in text:
        return "FOOD_ORDER"

    if "service" in text or "cleaning" in text or "towel" in text:
        return "ROOM_SERVICE"

    if "recommend" in text:
        return "ROOM_RECOMMEND"

    if "review" in text:
        return "REVIEW"

    if chatbot_model:
        try:
            return chatbot_model.predict([user_text])[0]
        except Exception:
            pass

    return "CHAT"


def stream_and_speak(response_text):
    st.write_stream(word + " " for word in response_text.split())

    audio_file = speak_response(response_text)
    if audio_file:
        st.audio(audio_file)


def extract_json_from_text(text):
    try:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group())
        return json.loads(text)
    except Exception:
        return {}


def extract_details_with_groq(user_text, task, current_data):
    if not GROQ_API_KEY:
        return {}

    today = date.today()
    tomorrow = today + timedelta(days=1)

    client = Groq(api_key=GROQ_API_KEY)

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {
                "role": "system",
                "content": f"""
You extract hotel {task} details.

Today is {today}.
Tomorrow is {tomorrow}.

Return ONLY valid JSON.
Do not add explanation.

Current data:
{json.dumps(current_data)}

For booking JSON keys:
guest_name, email, phone, room_type, check_in, check_out, guests

For food JSON keys:
guest_name, room_number, food_item, quantity

For service JSON keys:
guest_name, room_number, service_type, message

Use YYYY-MM-DD dates.
If value is missing, use null.
"""
            },
            {"role": "user", "content": user_text}
        ],
        temperature=0
    )

    return extract_json_from_text(response.choices[0].message.content)
def smart_booking_parser(user_text, current_data):
    if not GROQ_API_KEY:
        return {}

    client = Groq(api_key=GROQ_API_KEY)

    today = date.today()
    tomorrow = today + timedelta(days=1)

    prompt = f"""
You are a smart NLP parser for hotel booking.

User may type with spelling mistakes, grammar mistakes, mixed words, or incomplete text.

Today: {today}
Tomorrow: {tomorrow}

Extract booking details and return ONLY valid JSON.

Current data:
{json.dumps(current_data)}

Required keys:
guest_name, email, phone, room_type, check_in, check_out, guests

Rules:
- Convert dates to YYYY-MM-DD.
- Understand words like today, tomorrow, day after tomorrow.
- Understand spelling mistakes like singel=single, delux=deluxe.
- If user gives email, save email.
- If user gives 10 digit number, save phone.
- If user gives room type, match closest room type.
- If missing, use null.
- Do not write explanation.
"""

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_text}
        ],
        temperature=0
    )

    return extract_json_from_text(response.choices[0].message.content)


def update_dict(old, new):
    for key, value in new.items():
        if value not in [None, "", "null", "None"]:
            old[key] = value
    return old


def missing_fields(data):
    return [k for k, v in data.items() if v in [None, "", "null", "None"]]


# ---------------- CSS ----------------
st.markdown("""
<style>
.stApp {
    background-color: #f8fafc;
}
.main-title {
    font-size: 48px;
    font-weight: 800;
    color: #0f172a;
    text-align: center;
}
.sub-title {
    font-size: 18px;
    color: #64748b;
    text-align: center;
    margin-bottom: 30px;
}
.metric-card {
    background: linear-gradient(135deg, #2563eb, #38bdf8);
    padding: 25px;
    color: white;
    border-radius: 20px;
    text-align: center;
    box-shadow: 0px 6px 20px rgba(37,99,235,0.3);
    min-height: 140px;
}
.room-card {
    background: white;
    padding: 18px;
    border-radius: 18px;
    box-shadow: 0px 4px 18px rgba(0,0,0,0.08);
    margin-bottom: 20px;
}
.stButton button {
    width: 100%;
    border-radius: 12px;
    font-weight: bold;
    height: 48px;
}
section[data-testid="stSidebar"] {
    background-color: #0f172a;
}
section[data-testid="stSidebar"] * {
    color: white;
}
</style>
""", unsafe_allow_html=True)


st.markdown('<div class="main-title">🤖 AI Hotel Agent</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Book Rooms • Order Food • Request Services • Analyze Reviews • Voice AI Support</div>',
    unsafe_allow_html=True
)

st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["🏠 Home", "💬 AI Hotel Assistant", "🖼 Hotel Gallery"]
)


if "messages" not in st.session_state:
    st.session_state.messages = []

if "booking_data" not in st.session_state:
    st.session_state.booking_data = {
        "guest_name": None,
        "email": None,
        "phone": None,
        "room_type": None,
        "check_in": None,
        "check_out": None,
        "guests": None
    }

if "food_data" not in st.session_state:
    st.session_state.food_data = {
        "guest_name": None,
        "room_number": None,
        "food_item": None,
        "quantity": None
    }

if "service_data" not in st.session_state:
    st.session_state.service_data = {
        "guest_name": None,
        "room_number": None,
        "service_type": None,
        "message": None
    }

if "voice_question" not in st.session_state:
    st.session_state.voice_question = ""


# ---------------- HOME ----------------
if page == "🏠 Home":
    safe_image("assets/hotel_lobby.jpg", "Luxury Hotel Lobby")

    st.markdown("""
    ## ⭐ Luxury Hotel Experience

    This is an AI-powered hotel system where customers can book rooms,
    order food, request room service, analyze reviews, upload files/images,
    and get support through one AI Hotel Agent.
    """)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<div class="metric-card"><h2>🤖 AI Agent</h2></div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="metric-card"><h2>📊 ML Models</h2><p>Room recommendation and sentiment analysis</p></div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="metric-card"><h2>🎙 Voice AI</h2><p>Voice-to-text and spoken response</p></div>', unsafe_allow_html=True)

    st.divider()
    st.subheader("🏨 Featured Rooms")

    rooms_df = show_rooms().head(6)
    cols = st.columns(3)

    for index, room in rooms_df.iterrows():
        with cols[index % 3]:
            safe_image(get_room_image(room["room_type"]))
            st.markdown(f"""
            <div class="room-card">
                <h3>{room['room_type']}</h3>
                <h2>₹{room['price']} / night</h2>
                <p>📶 Free WiFi</p>
                <p>🍽 Breakfast Included</p>
                <p>🛏 Comfortable Stay</p>
            </div>
            """, unsafe_allow_html=True)

    st.divider()
    st.subheader("🖼 Hotel Gallery Preview")

    g1, g2, g3 = st.columns(3)

    with g1:
        safe_image("assets/restaurant.jpg", "Restaurant")

    with g2:
        safe_image("assets/swimming.jpg", "Swimming Pool")

    with g3:
        safe_image("assets/gym.jpg", "Fitness Center")


# ---------------- AI ASSISTANT ----------------
elif page == "💬 AI Hotel Assistant":
    st.header("💬 AI Hotel Agent")
    if st.button("🔄 Clear Chat / Start New Chat"):
        st.session_state.messages = []
        st.session_state.booking_data = {
            "guest_name": None,
            "email": None,
            "phone": None,
            "room_type": None,
            "check_in": None,
            "check_out": None,
            "guests": None
        }
        st.session_state.food_data = {
            "guest_name": None,
            "room_number": None,
            "food_item": None,
            "quantity": None
        }
        st.session_state.service_data = {
            "guest_name": None,
            "room_number": None,
            "service_type": None,
            "message": None
        }
        st.rerun()

    st.info("Use Enter to send. You can speak using the microphone. I can book rooms, order food, request services, recommend rooms and answer questions.")

    c1, c2, c3 = st.columns(3)

    with c1:
        if st.button("🏨 Book Room"):
            st.session_state.chat_prompt = "I need a deluxe room for 2 guests tomorrow"

    with c2:
        if st.button("🍽 Order Food"):
            st.session_state.chat_prompt = "Send 2 chicken biryanis to room 101"

    with c3:
        if st.button("🛎 Room Service"):
            st.session_state.chat_prompt = "Need room cleaning in room 101"

    with st.expander("📌 Example Commands"):
        st.write("""
show rooms

recommend, 5000, 2, 3

book room, Madhu, email@gmail.com, 9999999999, Deluxe Double, 2026-06-20, 2026-06-22, 2

place food, Madhu, 101, Chicken Biryani, 2

service, Madhu, 101, Cleaning, Need room cleaning

analyze review, excellent room and good service
        """)

    audio = mic_recorder(
        start_prompt="🎙 Start Recording",
        stop_prompt="⏹ Stop Recording",
        just_once=True
    )

    if audio:
        voice_text = convert_voice_to_text(audio)

        if voice_text.startswith("Voice-to-text error"):
            st.error(voice_text)
        else:
            st.session_state.voice_question = voice_text
            st.success(f"Recognized text: {voice_text}")

    uploaded_file = st.file_uploader(
        "Upload file or image for AI analysis",
        type=["txt", "pdf", "jpg", "jpeg", "png"]
    )

    file_content = ""

    if uploaded_file is not None:
        if uploaded_file.type == "text/plain":
            file_content = uploaded_file.read().decode("utf-8")
            st.success("Text file uploaded successfully.")

        elif uploaded_file.type == "application/pdf":
            file_content = extract_pdf_text(uploaded_file)

            if file_content.startswith("PDF library not installed"):
                st.error(file_content)

            elif file_content.startswith("PDF extraction error"):
                st.error(file_content)

            elif len(file_content.strip()) == 0:
                st.error("PDF text is empty. Try another PDF.")

            else:
                st.success("PDF text extracted successfully.")
                st.write(file_content[:500])

        elif uploaded_file.type in ["image/jpeg", "image/png"]:
            st.image(uploaded_file, caption="Uploaded Image", width="stretch")
            file_content = "IMAGE_UPLOADED"

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    if st.session_state.get("chat_prompt"):
        st.info(f"Suggested prompt: {st.session_state.chat_prompt}")

    if st.session_state.voice_question:
        st.info(f"Voice prompt detected: {st.session_state.voice_question}")
        if st.button("Send Voice Prompt"):
            st.session_state.pending_voice_prompt = st.session_state.voice_question

    user_question = st.chat_input("Ask your AI Hotel Agent...")

    if "pending_voice_prompt" in st.session_state:
        user_question = st.session_state.pending_voice_prompt
        del st.session_state.pending_voice_prompt

    if user_question:
        st.session_state.messages.append({"role": "user", "content": user_question})

        with st.chat_message("user"):
            st.write(user_question)

        with st.chat_message("assistant"):
            response_text = ""
            command = user_question.lower()

            if any(v is not None for v in st.session_state.booking_data.values()):
                intent = "BOOK_ROOM"

            elif any(v is not None for v in st.session_state.food_data.values()):
                intent = "FOOD_ORDER"

            elif any(v is not None for v in st.session_state.service_data.values()):
                intent = "ROOM_SERVICE"

            else:
                intent = detect_action(user_question)

            if uploaded_file is not None and file_content == "IMAGE_UPLOADED":
                response_text = analyze_image_with_groq(uploaded_file, user_question)
                stream_and_speak(response_text)

            elif "show" in command and "room" in command:
                rooms_df = show_rooms()
                st.dataframe(rooms_df, width="stretch")
                response_text = "Here are the available rooms."
                stream_and_speak(response_text)

            elif command.startswith("recommend,"):
                parts = user_question.split(",")

                if len(parts) == 4:
                    budget = int(parts[1].strip())
                    guests = int(parts[2].strip())
                    stay_days = int(parts[3].strip())
                    result = recommend_room_from_chat(budget, guests, stay_days)
                    response_text = f"Recommended Room: {result}"
                else:
                    response_text = "Use format: recommend, 5000, 2, 3"

                stream_and_speak(response_text)

            elif command.startswith("book room,"):
                parts = user_question.split(",")

                if len(parts) == 8:
                    response_text = book_room_from_chat(
                        parts[1].strip(),
                        parts[2].strip(),
                        parts[3].strip(),
                        parts[4].strip(),
                        parts[5].strip(),
                        parts[6].strip(),
                        int(parts[7].strip())
                    )
                else:
                    response_text = "Use format: book room, Name, email, phone, Room Type, check_in, check_out, guests"

                stream_and_speak(response_text)

            elif command.startswith("place food,"):
                parts = user_question.split(",")

                if len(parts) == 5:
                    response_text = place_food_order(
                        parts[1].strip(),
                        parts[2].strip(),
                        parts[3].strip(),
                        int(parts[4].strip())
                    )
                else:
                    response_text = "Use format: place food, Name, Room Number, Food Item, Quantity"

                stream_and_speak(response_text)

            elif command.startswith("service,"):
                parts = user_question.split(",")

                if len(parts) == 5:
                    response_text = create_service_request(
                        parts[1].strip(),
                        parts[2].strip(),
                        parts[3].strip(),
                        parts[4].strip()
                    )
                else:
                    response_text = "Use format: service, Name, Room Number, Service Type, Message"

                stream_and_speak(response_text)

            elif command.startswith("analyze review,"):
                review_text = user_question.split(",", 1)[1].strip()
                result = analyze_review_from_chat(review_text)
                response_text = f"Review Sentiment: {result}"
                stream_and_speak(response_text)

            elif intent == "BOOK_ROOM":

                booking = st.session_state.booking_data

                missing = [
                    k for k, v in booking.items()
                    if v in [None, "", "null", "None"]
                ]

                if missing and "," not in user_question:

                    if "@" in user_question:
                        booking["email"] = user_question

                    elif user_question.isdigit():
                        booking["phone"] = user_question

                    elif re.match(r"^\d{4}-\d{2}-\d{2}$", user_question):
                        booking["check_out"] = user_question
                    else:
                        booking[missing[0]] = user_question

                    st.session_state.booking_data = booking
                if "," in user_question:
                    parts = [x.strip() for x in user_question.split(",")]

                    if len(parts) >= 7:
                        st.session_state.booking_data = {
                            "guest_name": parts[0],
                            "email": parts[1],
                            "phone": parts[2],
                            "room_type": parts[3],
                            "check_in": parts[4],
                            "check_out": parts[5],
                            "guests": parts[6]
                        }

                extracted = smart_booking_parser(
                    user_question,
                    st.session_state.booking_data
                )
                st.session_state.booking_data = update_dict(
                    st.session_state.booking_data,
                    extracted
                )
                booking = st.session_state.booking_data

                bad_name_phrases = [
                    "i need",
                    "book",
                    "booking",
                    "room",
                    "deluxe",
                    "suite",
                    "standard",
                    "guest",
                    "tomorrow"
                ]

                if booking.get("guest_name"):

                    name = str(booking["guest_name"]).lower()

                    if any(word in name for word in bad_name_phrases):
                        booking["guest_name"] = None

                st.session_state.booking_data = booking

                missing = missing_fields(st.session_state.booking_data)

                if missing:

                    response_text = f"""
                    Current Booking Data:

                    Name: {st.session_state.booking_data['guest_name']}
                    Email: {st.session_state.booking_data['email']}
                    Phone: {st.session_state.booking_data['phone']}
                    Check In: {st.session_state.booking_data['check_in']}
                    Check Out: {st.session_state.booking_data['check_out']}
                    Guests: {st.session_state.booking_data['guests']}

                    Please provide: {', '.join(missing)}
                    """
                else:

                    data = st.session_state.booking_data

                    response_text = book_room_from_chat(
                        data["guest_name"],
                        data["email"],
                        data["phone"],
                        data["room_type"],
                        data["check_in"],
                        data["check_out"],
                        int(data["guests"])
                    )

                    st.session_state.booking_data = {
                        "guest_name": None,
                        "email": None,
                        "phone": None,
                        "room_type": None,
                        "check_in": None,
                        "check_out": None,
                        "guests": None
                    }

                stream_and_speak(response_text)


            elif intent == "FOOD_ORDER":
                extracted = extract_details_with_groq(
                    user_question,
                    "food",
                    st.session_state.food_data
                )

                st.session_state.food_data = update_dict(
                    st.session_state.food_data,
                    extracted
                )

                missing = missing_fields(st.session_state.food_data)

                if missing:
                    response_text = "Please provide: " + ", ".join(missing)
                else:
                    data = st.session_state.food_data

                    response_text = place_food_order(
                        data["guest_name"],
                        data["room_number"],
                        data["food_item"],
                        int(data["quantity"])
                    )

                    st.session_state.food_data = {
                        "guest_name": None,
                        "room_number": None,
                        "food_item": None,
                        "quantity": None
                    }

                stream_and_speak(response_text)

            elif intent == "ROOM_SERVICE":
                extracted = extract_details_with_groq(
                    user_question,
                    "service",
                    st.session_state.service_data
                )

                st.session_state.service_data = update_dict(
                    st.session_state.service_data,
                    extracted
                )

                missing = missing_fields(st.session_state.service_data)

                if missing:
                    response_text = "Please provide: " + ", ".join(missing)
                else:
                    data = st.session_state.service_data

                    response_text = create_service_request(
                        data["guest_name"],
                        data["room_number"],
                        data["service_type"],
                        data["message"]
                    )

                    st.session_state.service_data = {
                        "guest_name": None,
                        "room_number": None,
                        "service_type": None,
                        "message": None
                    }

                stream_and_speak(response_text)

            elif file_content:
                client = Groq(api_key=GROQ_API_KEY)

                response = client.chat.completions.create(
                    model=GROQ_MODEL,
                    messages=[
                        {
                            "role": "system",
                            "content": "Analyze the uploaded file content and answer the user question."
                        },
                        {
                            "role": "user",
                            "content": f"{user_question}\n\nFile content:\n{file_content}"
                        }
                    ]
                )

                response_text = response.choices[0].message.content
                stream_and_speak(response_text)

            else:
                client = Groq(api_key=GROQ_API_KEY)

                response = client.chat.completions.create(
                    model=GROQ_MODEL,
                    messages=[
                        {
                            "role": "system",
                            "content": """
                            You are an AI hotel agent.
                            Help with room booking, food order, room service,
                            payments, reviews, file analysis and support.
                            """
                        },
                        {"role": "user", "content": user_question}
                    ]
                )

                response_text = response.choices[0].message.content
                stream_and_speak(response_text)

        st.session_state.messages.append(
            {"role": "assistant", "content": response_text}
        )


elif page == "🖼 Hotel Gallery":

    st.header("🏨 Hotel Gallery")

    st.subheader("🏢 Hotel Facilities")

    col1, col2 = st.columns(2)

    with col1:
        safe_image("assets/hotel_lobby.jpg", "Luxury Hotel Lobby")
        safe_image("assets/restaurant.jpg", "Restaurant")
        safe_image("assets/reception.jpg", "Reception")

    with col2:
        safe_image("assets/swimming.jpg", "Swimming Pool")
        safe_image("assets/gym.jpg", "Fitness Center")

    st.divider()

    st.subheader("🛏 All Room Collection")

    rooms_df = show_rooms()

    cols = st.columns(3)

    for index, room in rooms_df.iterrows():

        with cols[index % 3]:

            safe_image(
                get_room_image(room["room_type"]),
                room["room_type"]
            )

            st.markdown(f"""
            <div class="room-card">
                <h3>{room['room_type']}</h3>
                <h2>₹{room['price']} / night</h2>
                <p>📶 Free WiFi</p>
                <p>🍽 Breakfast Included</p>
                <p>🛏 Comfortable Stay</p>
            </div>
            """, unsafe_allow_html=True)