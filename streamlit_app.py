import os
import re
import json
import base64
import sqlite3
import tempfile
import random
from datetime import date, timedelta
from urllib.parse import quote

import joblib
import pandas as pd
import qrcode
import streamlit as st
import yagmail
from dotenv import load_dotenv
from groq import Groq
from streamlit_mic_recorder import mic_recorder

try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None

APP_VERSION = "premium_ui_staff_access_v1"
DB_NAME = "hotel_streamlit.db"

st.set_page_config(page_title="AI Hotel Agent", page_icon="🏨", layout="wide")
load_dotenv()


def get_secret(key, default=None):
    try:
        return st.secrets[key]
    except Exception:
        return os.getenv(key, default)


GROQ_API_KEY = get_secret("GROQ_API_KEY")
GROQ_MODEL = get_secret("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_VISION_MODEL = get_secret("GROQ_VISION_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")
GROQ_STT_MODEL = get_secret("GROQ_STT_MODEL", "whisper-large-v3-turbo")
EMAIL_USER = get_secret("EMAIL_USER")
EMAIL_APP_PASSWORD = get_secret("EMAIL_APP_PASSWORD")
UPI_ID = get_secret("UPI_ID", "yourupi@ybl")
STAFF_PASSWORD = get_secret("STAFF_PASSWORD", "1234")


# ---------------- DATABASE ----------------
def get_connection():
    return sqlite3.connect(DB_NAME, check_same_thread=False)


def ensure_column(table_name, column_name, column_type):
    conn = get_connection()
    c = conn.cursor()
    c.execute(f"PRAGMA table_info({table_name})")
    columns = [row[1] for row in c.fetchall()]
    if column_name not in columns:
        c.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}")
    conn.commit()
    conn.close()


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
            customer_id TEXT,
            booking_id TEXT,
            room_number TEXT,
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

    ensure_column("bookings", "customer_id", "TEXT")
    ensure_column("bookings", "booking_id", "TEXT")
    ensure_column("bookings", "room_number", "TEXT")
    ensure_column("bookings", "payment_status", "TEXT DEFAULT 'Pending'")
    ensure_column("bookings", "transaction_id", "TEXT")


def room_price_list():
    return [
        ("Standard Single", 2000),
        ("Standard Double", 3000),
        ("Deluxe Single", 4000),
        ("Deluxe Double", 5000),
        ("Executive Room", 6500),
        ("Business Room", 7000),
        ("Family Room", 8000),
        ("Garden View Room", 8500),
        ("Suite Room", 10000),
        ("Ocean View Room", 11000),
        ("Luxury Suite", 13000),
        ("Honeymoon Suite", 15000),
        ("Presidential Suite", 20000),
        ("Penthouse Suite", 25000),
    ]


def insert_default_rooms():
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM rooms")
    if c.fetchone()[0] == 0:
        c.executemany("INSERT INTO rooms (room_type, price) VALUES (?, ?)", room_price_list())
    conn.commit()
    conn.close()


def update_room_prices():
    conn = get_connection()
    c = conn.cursor()
    for room_type, price in room_price_list():
        c.execute("UPDATE rooms SET price = ? WHERE room_type = ?", (price, room_type))
    conn.commit()
    conn.close()


create_tables()
insert_default_rooms()
update_room_prices()


# ---------------- HELPERS ----------------
def show_rooms():
    conn = get_connection()
    df = pd.read_sql_query("SELECT room_type, price FROM rooms", conn)
    conn.close()
    return df


def generate_customer_id(phone=None):
    phone = re.sub(r"\D", "", str(phone or ""))
    if len(phone) >= 4:
        return "CUST" + phone[-4:]
    return f"CUST{random.randint(1000, 9999)}"


def generate_booking_id():
    return f"BK{random.randint(100000, 999999)}"


def allocate_room_number(room_type):
    room_numbers = {
        "Standard Single": "101",
        "Standard Double": "102",
        "Deluxe Single": "201",
        "Deluxe Double": "202",
        "Executive Room": "301",
        "Business Room": "302",
        "Family Room": "401",
        "Garden View Room": "402",
        "Suite Room": "501",
        "Ocean View Room": "502",
        "Luxury Suite": "601",
        "Honeymoon Suite": "602",
        "Presidential Suite": "701",
        "Penthouse Suite": "801",
    }
    return room_numbers.get(room_type, "999")


def closest_room_type(text):
    rooms = show_rooms()["room_type"].tolist()
    text = str(text).lower().strip()
    for room in rooms:
        if room.lower() == text:
            return room
    if "standard single" in text or "single standard" in text:
        return "Standard Single"
    if "standard double" in text:
        return "Standard Double"
    if "deluxe single" in text or "delux single" in text:
        return "Deluxe Single"
    if "deluxe double" in text or "delux double" in text:
        return "Deluxe Double"
    if "deluxe" in text or "delux" in text:
        return "Deluxe Double"
    if "standard" in text:
        return "Standard Single"
    if "suite" in text:
        return "Suite Room"
    if "family" in text:
        return "Family Room"
    if "business" in text:
        return "Business Room"
    if "executive" in text:
        return "Executive Room"
    return None


def normalize_date(value):
    if value in [None, "", "null", "None"]:
        return None
    text = str(value).strip().lower()
    today = date.today()
    if text == "today":
        return today.strftime("%Y-%m-%d")
    if text in ["tomorrow", "tommorow", "tomorror"]:
        return (today + timedelta(days=1)).strftime("%Y-%m-%d")
    if text in ["day after tomorrow", "dayafter tomorrow", "dayafter tomorror"]:
        return (today + timedelta(days=2)).strftime("%Y-%m-%d")
    try:
        parsed = pd.to_datetime(text, dayfirst=True, errors="coerce")
        if pd.isna(parsed):
            return value
        return parsed.strftime("%Y-%m-%d")
    except Exception:
        return value


def safe_image(path, caption=None):
    if os.path.exists(path):
        st.image(path, caption=caption, width="stretch")
    else:
        st.warning(f"Image missing: {path}")


def get_room_image(room_type):
    room_type = str(room_type).lower()
    if "standard single" in room_type:
        return "assets/standard_single.jpg"
    if "standard double" in room_type:
        return "assets/standard.jpg"
    if "deluxe single" in room_type:
        return "assets/deluxe_single.jpg"
    if "deluxe double" in room_type:
        return "assets/deluxe.jpg"
    if "executive" in room_type or "business" in room_type:
        return "assets/deluxe.jpg"
    if any(x in room_type for x in ["suite", "family", "garden", "ocean", "honeymoon", "presidential", "penthouse"]):
        return "assets/suite.jpg"
    return "assets/standard.jpg"


# ---------------- MODELS ----------------
@st.cache_resource
def load_models():
    room_model = None
    sentiment_model = None
    chatbot_model = None
    if os.path.exists("room_recommendation_model.pkl"):
        room_model = joblib.load("room_recommendation_model.pkl")
    if os.path.exists("sentiment_model.pkl"):
        sentiment_model = joblib.load("sentiment_model.pkl")
    if os.path.exists("chatbot_intent_model.pkl"):
        chatbot_model = joblib.load("chatbot_intent_model.pkl")
    return room_model, sentiment_model, chatbot_model


room_model, sentiment_model, chatbot_model = load_models()


def recommend_room_from_chat(budget, guests, stay_days):
    if room_model is None:
        return "Room recommendation model not found."
    input_df = pd.DataFrame([[budget, guests, stay_days]], columns=["budget", "guests", "stay_days"])
    return room_model.predict(input_df)[0]


def analyze_review_from_chat(review_text):
    if sentiment_model is None:
        return "Sentiment model not found."
    result = sentiment_model.predict([review_text])[0]
    conn = get_connection()
    conn.execute(
        "INSERT INTO reviews (guest_name, review_text, sentiment) VALUES (?, ?, ?)",
        ("Chat User", review_text, result),
    )
    conn.commit()
    conn.close()
    return result


# ---------------- PAYMENT ----------------
def create_upi_qr(amount):
    name = "NM Hotels"
    upi_id = str(UPI_ID).strip()

    if not upi_id or "@" not in upi_id or "@@" in upi_id or "UPI_ID=" in upi_id:
        st.error("Invalid UPI ID. In .env use only: UPI_ID=yourrealupi@ybl")
        return None, None

    try:
        amount = f"{float(amount):.2f}"
    except Exception:
        st.error("Invalid payment amount.")
        return None, None

    upi_link = (
        f"upi://pay?"
        f"pa={quote(upi_id)}"
        f"&pn={quote(name)}"
        f"&am={amount}"
        f"&cu=INR"
        f"&tn={quote('Hotel Room Booking')}"
    )
    qr_img = qrcode.make(upi_link)
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    qr_img.save(temp_file.name)
    return temp_file.name, upi_link


def calculate_payment_breakdown(room_type, check_in, check_out):
    room_type = closest_room_type(room_type)
    if not room_type:
        return None

    conn = get_connection()
    c = conn.cursor()
    room = c.execute("SELECT price FROM rooms WHERE room_type = ?", (room_type,)).fetchone()
    conn.close()

    if room is None:
        return None

    try:
        nights = (pd.to_datetime(check_out) - pd.to_datetime(check_in)).days
    except Exception:
        return None

    if nights <= 0:
        return None

    room_price = float(room[0])
    subtotal = room_price * nights
    gst = subtotal * 0.12
    service_charge = subtotal * 0.05
    total = subtotal + gst + service_charge

    return {
        "room_type": room_type,
        "room_price": room_price,
        "nights": nights,
        "subtotal": round(subtotal, 2),
        "gst": round(gst, 2),
        "service_charge": round(service_charge, 2),
        "total": round(total, 2),
    }


def show_payment_breakdown_and_qr(booking):
    breakdown = calculate_payment_breakdown(booking["room_type"], booking["check_in"], booking["check_out"])
    if not breakdown:
        st.error("Payment breakdown could not be calculated.")
        return

    st.subheader("💳 Payment Breakdown")
    st.write(f"Booking ID: {booking.get('booking_id')}")
    st.write(f"Customer ID: {booking.get('customer_id')}")
    st.write(f"Room Number: {booking.get('room_number')}")
    st.write(f"Room Type: {breakdown['room_type']}")
    st.write(f"Room Price: ₹{breakdown['room_price']} per night")
    st.write(f"Nights: {breakdown['nights']}")
    st.write(f"Subtotal: ₹{breakdown['subtotal']}")
    st.write(f"GST 12%: ₹{breakdown['gst']}")
    st.write(f"Service Charge 5%: ₹{breakdown['service_charge']}")
    st.success(f"Total Amount: ₹{breakdown['total']}")

    qr_path, upi_link = create_upi_qr(breakdown["total"])
    if qr_path:
        st.image(qr_path, caption="Scan to Pay", width=250)
        st.markdown(
            f"""
            <a href="{upi_link}">
                <button style="background-color:#16a34a;color:white;padding:12px 20px;border:none;border-radius:10px;font-size:16px;cursor:pointer;">
                    Pay Now with UPI
                </button>
            </a>
            """,
            unsafe_allow_html=True,
        )
        st.caption("If the button does not open on laptop, scan the QR using PhonePe / Google Pay / Paytm mobile app.")


def mark_payment_paid(transaction_id):
    """Manual verification mode: stores transaction ID, but does not mark as Paid automatically."""
    booking = st.session_state.current_booking
    if not booking or not booking.get("booking_id"):
        return "No active booking found."

    transaction_id = str(transaction_id).strip()
    if not transaction_id:
        return "Please enter a valid UPI transaction/reference ID."

    conn = get_connection()
    c = conn.cursor()
    c.execute(
        """
        UPDATE bookings
        SET payment_status = 'Verification Pending',
            transaction_id = ?
        WHERE booking_id = ?
        """,
        (transaction_id, booking["booking_id"]),
    )
    conn.commit()
    conn.close()

    st.session_state.current_booking["payment_status"] = "Verification Pending"
    st.session_state.current_booking["transaction_id"] = transaction_id
    st.session_state.awaiting_payment = False

    return f"""
Transaction ID received: {transaction_id}

Payment Status: Verification Pending

Hotel staff will verify this transaction in the UPI app/bank statement.
After verification, staff can mark the booking as Paid using:
verify payment {booking['booking_id']}
"""


def verify_payment_manually(booking_id):
    """Hotel staff command after checking UPI/bank statement."""
    booking_id = str(booking_id).strip()
    if not booking_id:
        return "Please provide booking ID. Example: verify payment BK123456"

    conn = get_connection()
    c = conn.cursor()
    c.execute("UPDATE bookings SET payment_status = 'Paid' WHERE booking_id = ?", (booking_id,))
    updated = c.rowcount
    conn.commit()
    conn.close()

    if updated == 0:
        return f"Booking ID not found: {booking_id}"

    if st.session_state.current_booking and st.session_state.current_booking.get("booking_id") == booking_id:
        st.session_state.current_booking["payment_status"] = "Paid"

    return f"Payment manually verified. Booking {booking_id} marked as Paid."


# ---------------- EMAIL ----------------
def send_booking_email(to_email, guest_name, room_type, check_in, check_out, total_price, booking_id=None, customer_id=None, room_number=None):
    if not EMAIL_USER or not EMAIL_APP_PASSWORD:
        return False

    body = f"""
Dear {guest_name},

Your room booking is confirmed.

Booking ID: {booking_id}
Customer ID: {customer_id}
Room Number: {room_number}

Room Type: {room_type}
Check-in Date: {check_in}
Check-out Date: {check_out}
Total Amount: ₹{total_price}
Payment Status: Pending

Thank you for choosing our hotel.

Regards,
Hotel Team
"""
    try:
        yag = yagmail.SMTP(EMAIL_USER, EMAIL_APP_PASSWORD)
        yag.send(to=to_email, subject="Hotel Room Booking Confirmation", contents=body)
        return True
    except Exception as e:
        st.warning(f"Email not sent: {e}")
        return False


# ---------------- BOOKING / FOOD / SERVICE ----------------
def book_room_from_chat(guest_name, email, phone, room_type, check_in, check_out, guests):
    if not guest_name or not email or not phone:
        return "Missing guest name, email, or phone."
    if "@" not in email:
        return "Invalid email address."

    phone = re.sub(r"\D", "", str(phone))
    if not phone.isdigit() or len(phone) < 10:
        return "Invalid phone number."

    room_type = closest_room_type(room_type)
    if not room_type:
        return "Room type not found. Type `show rooms` to see room names."

    check_in = normalize_date(check_in)
    check_out = normalize_date(check_out)

    try:
        days = (pd.to_datetime(check_out) - pd.to_datetime(check_in)).days
    except Exception:
        return "Invalid date format. Use YYYY-MM-DD."

    if days <= 0:
        return "Check-out date must be after check-in date."

    conn = get_connection()
    c = conn.cursor()
    existing_booking = c.execute(
        """
        SELECT COUNT(*)
        FROM bookings
        WHERE room_type = ?
        AND NOT (check_out <= ? OR check_in >= ?)
        """,
        (room_type, check_in, check_out),
    ).fetchone()[0]

    if existing_booking > 0:
        conn.close()
        return "This room is already booked for selected dates."

    breakdown = calculate_payment_breakdown(room_type, check_in, check_out)
    if not breakdown:
        conn.close()
        return "Payment breakdown could not be calculated."

    total_price = breakdown["total"]
    profile = st.session_state.customer_profile
    customer_id = profile.get("customer_id") or generate_customer_id(phone)
    booking_id = generate_booking_id()
    room_number = allocate_room_number(room_type)

    c.execute(
        """
        INSERT INTO bookings
        (customer_id, booking_id, room_number, guest_name, email, phone, room_type, check_in, check_out, guests, total_price, payment_status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            customer_id,
            booking_id,
            room_number,
            guest_name,
            email,
            phone,
            room_type,
            check_in,
            check_out,
            guests,
            total_price,
            "Pending",
        ),
    )
    conn.commit()
    conn.close()

    st.session_state.customer_profile = {
        "customer_id": customer_id,
        "guest_name": guest_name,
        "email": email,
        "phone": phone,
        "room_number": room_number,
        "check_out": check_out,
    }
    st.session_state.current_booking = {
        "customer_id": customer_id,
        "booking_id": booking_id,
        "room_number": room_number,
        "guest_name": guest_name,
        "email": email,
        "phone": phone,
        "room_type": room_type,
        "check_in": check_in,
        "check_out": check_out,
        "guests": guests,
        "total_price": total_price,
        "payment_status": "Pending",
    }

    email_sent = send_booking_email(email, guest_name, room_type, check_in, check_out, total_price, booking_id, customer_id, room_number)
    email_status = "Confirmation email sent." if email_sent else "Email not sent."

    return f"""
✅ Room booked successfully.

Booking ID: {booking_id}
Customer ID: {customer_id}
Room Number: {room_number}

Guest Name: {guest_name}
Room Type: {room_type}
Check-in: {check_in}
Check-out: {check_out}
Guests: {guests}

Total Amount: ₹{total_price}
Payment Status: Pending

{email_status}

Type `pay now` to generate UPI QR code.
"""


def place_food_order(guest_name, room_number, food_item, quantity):
    conn = get_connection()
    conn.execute(
        "INSERT INTO food_orders (guest_name, room_number, food_item, quantity) VALUES (?, ?, ?, ?)",
        (guest_name, room_number, food_item, quantity),
    )
    conn.commit()
    conn.close()
    return f"Food order placed: {quantity} x {food_item} for room {room_number}."


def create_service_request(guest_name, room_number, service_type, message):
    conn = get_connection()
    conn.execute(
        "INSERT INTO service_requests (guest_name, room_number, service_type, message) VALUES (?, ?, ?, ?)",
        (guest_name, room_number, service_type, message),
    )
    conn.commit()
    conn.close()
    return f"Room service request created for room {room_number}: {service_type}."


# ---------------- GROQ / FILE / VOICE ----------------
def analyze_image_with_groq(uploaded_file, user_question):
    if not GROQ_API_KEY:
        return "Groq API Key not found."
    try:
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
                            "image_url": {"url": f"data:{uploaded_file.type};base64,{image_base64}"},
                        },
                    ],
                }
            ],
            max_tokens=500,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Image analysis failed: {e}"


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
    if not GROQ_API_KEY:
        return "Voice-to-text error: Groq API key missing."
    try:
        audio_bytes = audio["bytes"]
        temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".webm")
        temp_audio.write(audio_bytes)
        temp_audio.close()
        client = Groq(api_key=GROQ_API_KEY)
        with open(temp_audio.name, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(file=audio_file, model=GROQ_STT_MODEL)
        return transcription.text
    except Exception as e:
        return f"Voice-to-text error: {e}"


def stream_and_speak(response_text):
    st.write(response_text)


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
    client = Groq(api_key=GROQ_API_KEY)
    today = date.today()
    tomorrow = today + timedelta(days=1)
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {
                "role": "system",
                "content": f"""
You extract hotel {task} details.
Today is {today}. Tomorrow is {tomorrow}.
Return ONLY valid JSON. Do not add explanation.
Current data: {json.dumps(current_data)}
For booking JSON keys: guest_name, email, phone, room_type, check_in, check_out, guests
For food JSON keys: guest_name, room_number, food_item, quantity
For service JSON keys: guest_name, room_number, service_type, message
Use YYYY-MM-DD dates. If value is missing, use null.
""",
            },
            {"role": "user", "content": user_text},
        ],
        temperature=0,
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
User may type with spelling mistakes or incomplete text.
Today: {today}. Tomorrow: {tomorrow}.
Extract booking details and return ONLY valid JSON.
Current data: {json.dumps(current_data)}
Required keys: guest_name, email, phone, room_type, check_in, check_out, guests
Rules:
- Convert dates to YYYY-MM-DD.
- Understand today, tomorrow, day after tomorrow.
- Understand spelling mistakes like singel=single, delux=deluxe.
- If user gives email, save email.
- If user gives 10 digit number, save phone.
- If user says only booking request, do not use that as guest name.
- If value is missing, use null.
"""
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "system", "content": prompt}, {"role": "user", "content": user_text}],
        temperature=0,
    )
    return extract_json_from_text(response.choices[0].message.content)


# ---------------- DATA / INTENT ----------------
def update_dict(old, new):
    for key, value in new.items():
        if value not in [None, "", "null", "None"]:
            old[key] = normalize_date(value) if key in ["check_in", "check_out"] else value
    return old


def missing_fields(data):
    return [k for k, v in data.items() if v in [None, "", "null", "None"]]


def update_customer_profile(data):
    profile = st.session_state.customer_profile
    for key in ["guest_name", "email", "phone", "room_number", "check_out"]:
        if key in data and data[key] not in [None, "", "null", "None"]:
            profile[key] = data[key]
    if not profile.get("customer_id") and profile.get("phone"):
        profile["customer_id"] = generate_customer_id(profile["phone"])
    st.session_state.customer_profile = profile


def apply_customer_profile(data):
    profile = st.session_state.customer_profile
    for key in ["guest_name", "email", "phone", "room_number"]:
        if key in data and data[key] in [None, "", "null", "None"]:
            data[key] = profile.get(key)
    return data


def is_likely_name(text):
    text = text.strip().lower()
    bad_words = ["book", "booking", "room", "want", "need", "yes", "payment", "pay", "standard", "deluxe", "suite", "food", "service", "confirm", "proceed"]
    if any(word in text for word in bad_words):
        return False
    if "@" in text or re.search(r"\d", text):
        return False
    return 1 <= len(text.split()) <= 3 and len(text) > 1


def detect_action(user_text):
    text = user_text.lower()
    if text.startswith("verify payment"):
        return "PAYMENT"
    if any(word in text for word in ["pay", "payment", "upi", "transaction"]):
        return "PAYMENT"
    if any(word in text for word in ["checkout", "check out", "clear customer", "end stay"]):
        return "CHECKOUT"
    if any(word in text for word in ["book", "booking", "reserve", "reservation", "need a room", "want a room", "deluxe room", "standard room", "suite room", "room for"]):
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
# ---------------- CSS ----------------
st.markdown("""
<style>
/* GOOGLE FONT STYLE */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

/* FULL APP */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at top left, rgba(59,130,246,0.20), transparent 28%),
        radial-gradient(circle at top right, rgba(245,158,11,0.16), transparent 28%),
        linear-gradient(135deg, #eef4ff 0%, #f8fafc 45%, #ffffff 100%);
    color: #0f172a !important;
}

/* MAIN CONTAINER */
.block-container {
    padding-top: 1.5rem;
    padding-bottom: 7rem;
    max-width: 1250px;
}

/* FORCE MAIN TEXT DARK */
section.main p,
section.main span,
section.main div,
section.main label,
section.main h1,
section.main h2,
section.main h3,
section.main h4,
section.main h5,
section.main h6 {
    color: #0f172a !important;
}

/* SIDEBAR PREMIUM */
section[data-testid="stSidebar"] {
    background:
        radial-gradient(circle at top, rgba(245,158,11,0.18), transparent 28%),
        linear-gradient(180deg, #020617 0%, #0f1f4d 50%, #020617 100%);
    border-right: 1px solid rgba(255,255,255,0.14);
    box-shadow: 12px 0 35px rgba(15,23,42,0.22);
}

section[data-testid="stSidebar"] * {
    color: #ffffff !important;
}

/* SIDEBAR TITLE */
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #ffffff !important;
    font-weight: 900 !important;
}

/* SIDEBAR CAPTION VERSION */
section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
    color: #cbd5e1 !important;
    font-size: 12px !important;
}

/* RADIO NAVIGATION */
section[data-testid="stSidebar"] .stRadio > div {
    gap: 10px;
}

section[data-testid="stSidebar"] label {
    background: rgba(255,255,255,0.06);
    padding: 12px 14px;
    border-radius: 16px;
    margin-bottom: 8px;
    border: 1px solid rgba(255,255,255,0.08);
    transition: all 0.25s ease;
}

section[data-testid="stSidebar"] label:hover {
    background: rgba(245,158,11,0.18);
    border: 1px solid rgba(245,158,11,0.45);
    transform: translateX(4px);
}

/* SIDEBAR INPUT */
section[data-testid="stSidebar"] input {
    color: #0f172a !important;
    background: #ffffff !important;
    border-radius: 12px !important;
}

/* PREMIUM HEADER */
.main-title {
    font-size: 50px;
    font-weight: 950;
    letter-spacing: -1.5px;
    color: #08111f !important;
    text-align: left;
    margin-bottom: 4px;
    line-height: 1.05;
}

.main-title::before {
    content: "🏨";
    display: inline-flex;
    align-items: center;
    justify-content: center;
    margin-right: 14px;
    width: 58px;
    height: 58px;
    border-radius: 20px;
    background: linear-gradient(135deg, #1e3a8a, #2563eb);
    box-shadow: 0 14px 35px rgba(37,99,235,0.35);
}

.sub-title {
    font-size: 17px;
    color: #64748b !important;
    text-align: left;
    margin-bottom: 30px;
    font-weight: 500;
}

/* PREMIUM CARDS */
.metric-card {
    background:
        linear-gradient(135deg, rgba(30,58,138,0.96), rgba(37,99,235,0.92)),
        radial-gradient(circle at top right, rgba(245,158,11,0.30), transparent 35%);
    padding: 28px;
    color: white !important;
    border-radius: 28px;
    text-align: center;
    box-shadow: 0 18px 45px rgba(37,99,235,0.30);
    min-height: 150px;
    border: 1px solid rgba(255,255,255,0.20);
    transition: all 0.25s ease;
}

.metric-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 24px 55px rgba(37,99,235,0.40);
}

.metric-card * {
    color: white !important;
}

/* ROOM CARDS */
.room-card {
    background: rgba(255,255,255,0.88);
    backdrop-filter: blur(16px);
    padding: 22px;
    border-radius: 28px;
    box-shadow: 0 16px 45px rgba(15,23,42,0.10);
    margin-bottom: 24px;
    border: 1px solid rgba(226,232,240,0.95);
    transition: all 0.25s ease;
}

.room-card:hover {
    transform: translateY(-6px);
    box-shadow: 0 25px 60px rgba(15,23,42,0.16);
    border: 1px solid rgba(37,99,235,0.35);
}

.room-card h3 {
    font-size: 22px !important;
    font-weight: 850 !important;
    color: #0f172a !important;
}

.room-card h2 {
    color: #1e3a8a !important;
    font-weight: 900 !important;
}

.room-card p {
    color: #475569 !important;
    font-weight: 600 !important;
}

/* IMAGES */
img {
    border-radius: 26px !important;
    box-shadow: 0 14px 35px rgba(15,23,42,0.12);
}

/* CHAT MESSAGE CARD */
div[data-testid="stChatMessage"] {
    background: rgba(255,255,255,0.92) !important;
    backdrop-filter: blur(18px);
    border-radius: 28px !important;
    padding: 20px 22px !important;
    margin-bottom: 18px !important;
    box-shadow: 0 14px 40px rgba(15,23,42,0.09);
    border: 1px solid rgba(226,232,240,0.95);
}

/* CHAT TEXT FIX */
div[data-testid="stChatMessage"] p,
div[data-testid="stChatMessage"] span,
div[data-testid="stChatMessage"] div {
    color: #0f172a !important;
    font-weight: 500;
}

/* CHAT ICONS */
div[data-testid="stChatMessage"] svg {
    color: #1e3a8a !important;
}

/* USER CHAT MESSAGE STYLE */
div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    background: linear-gradient(135deg, #eef4ff, #ffffff) !important;
    border: 1px solid rgba(37,99,235,0.20);
}

/* ASSISTANT CHAT MESSAGE STYLE */
div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
    background: rgba(255,255,255,0.96) !important;
    border-left: 5px solid #f59e0b;
}

/* BUTTONS */
.stButton button {
    width: 100%;
    border-radius: 18px;
    font-weight: 850;
    height: 50px;
    background: linear-gradient(135deg, #1e3a8a, #2563eb);
    color: white !important;
    border: none;
    box-shadow: 0 12px 28px rgba(37,99,235,0.28);
    transition: all 0.25s ease;
}

.stButton button:hover {
    transform: translateY(-2px);
    box-shadow: 0 18px 40px rgba(37,99,235,0.38);
    background: linear-gradient(135deg, #1d4ed8, #3b82f6);
}

/* SUCCESS / INFO / WARNING BOXES */
.stAlert {
    border-radius: 22px !important;
    border: 1px solid rgba(226,232,240,0.9) !important;
    box-shadow: 0 10px 28px rgba(15,23,42,0.07);
}

.stAlert p,
.stAlert div,
.stAlert span {
    color: #0f172a !important;
    font-weight: 600;
}

/* EXPANDER */
div[data-testid="stExpander"] {
    background: rgba(255,255,255,0.92) !important;
    border-radius: 22px !important;
    border: 1px solid rgba(203,213,225,0.9) !important;
    box-shadow: 0 12px 32px rgba(15,23,42,0.08);
}

div[data-testid="stExpander"] * {
    color: #0f172a !important;
}

/* FILE UPLOADER */
div[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.95) !important;
    border-radius: 24px !important;
    padding: 18px !important;
    border: 1px dashed rgba(37,99,235,0.45) !important;
    box-shadow: 0 14px 35px rgba(15,23,42,0.08);
}

div[data-testid="stFileUploader"] * {
    color: #0f172a !important;
}

/* CHAT INPUT */
div[data-testid="stChatInput"] {
    background: transparent !important;
}

div[data-testid="stChatInput"] > div {
    border-radius: 24px !important;
    box-shadow: 0 18px 45px rgba(15,23,42,0.18);
    border: 1px solid rgba(148,163,184,0.35);
    background: rgba(255,255,255,0.95) !important;
}

div[data-testid="stChatInput"] textarea {
    background: transparent !important;
    color: #0f172a !important;
    font-size: 16px !important;
    font-weight: 500 !important;
}

div[data-testid="stChatInput"] textarea::placeholder {
    color: #64748b !important;
}

/* DATAFRAME */
div[data-testid="stDataFrame"] {
    background: white !important;
    border-radius: 20px !important;
    overflow: hidden;
    box-shadow: 0 12px 35px rgba(15,23,42,0.08);
}

/* STAFF LOGIN LOOK */
input[type="password"] {
    border-radius: 14px !important;
    border: 1px solid rgba(148,163,184,0.45) !important;
}

/* PAYMENT QR CARD */
[data-testid="stImage"] {
    text-align: center;
}

/* HORIZONTAL LINE */
hr {
    border: none;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(148,163,184,0.5), transparent);
    margin: 30px 0;
}

/* MOBILE RESPONSIVE */
@media screen and (max-width: 768px) {
    .block-container {
        padding-left: 0.85rem !important;
        padding-right: 0.85rem !important;
        padding-top: 1rem !important;
        max-width: 100% !important;
        padding-bottom: 7rem !important;
    }

    .main-title {
        font-size: 30px !important;
        text-align: center !important;
        letter-spacing: -0.5px;
    }

    .main-title::before {
        width: 46px;
        height: 46px;
        border-radius: 16px;
        margin-right: 8px;
    }

    .sub-title {
        font-size: 14px !important;
        text-align: center !important;
        margin-bottom: 18px !important;
    }

    div[data-testid="stChatMessage"] {
        padding: 15px !important;
        border-radius: 20px !important;
        margin-bottom: 14px !important;
    }

    div[data-testid="stChatMessage"] p {
        font-size: 14px !important;
        line-height: 1.5 !important;
    }

    .room-card {
        padding: 16px !important;
        border-radius: 20px !important;
    }

    .metric-card {
        min-height: 115px !important;
        padding: 18px !important;
        border-radius: 22px !important;
    }

    .stButton button {
        height: 45px !important;
        font-size: 14px !important;
        border-radius: 14px !important;
    }

    div[data-testid="stFileUploader"] {
        padding: 14px !important;
        border-radius: 18px !important;
    }

    div[data-testid="stChatInput"] > div {
        border-radius: 18px !important;
    }

    div[data-testid="stChatInput"] textarea {
        font-size: 14px !important;
    }

    section[data-testid="stSidebar"] {
        width: 86vw !important;
    }
}
</style>
""", unsafe_allow_html=True)
# ---------------- SESSION STATE ----------------
def init_session():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "staff_logged_in" not in st.session_state:
        st.session_state.staff_logged_in = False
    if "customer_profile" not in st.session_state:
        st.session_state.customer_profile = {"customer_id": None, "guest_name": None, "email": None, "phone": None, "room_number": None, "check_out": None}
    if "current_booking" not in st.session_state:
        st.session_state.current_booking = None
    if "awaiting_payment" not in st.session_state:
        st.session_state.awaiting_payment = False
    if "booking_data" not in st.session_state:
        st.session_state.booking_data = {"guest_name": None, "email": None, "phone": None, "room_type": None, "check_in": None, "check_out": None, "guests": None}
    if "food_data" not in st.session_state:
        st.session_state.food_data = {"guest_name": None, "room_number": None, "food_item": None, "quantity": None}
    if "service_data" not in st.session_state:
        st.session_state.service_data = {"guest_name": None, "room_number": None, "service_type": None, "message": None}
    if "voice_question" not in st.session_state:
        st.session_state.voice_question = ""


init_session()


# ---------------- PREMIUM HEADER + SIDEBAR ----------------
st.markdown("""
<div class="hero-card">
    <div class="hero-wrap">
        <div class="hero-left">
            <div class="hero-icon">✨</div>
            <div>
                <h1 class="hero-title">AI Hotel Agent</h1>
                <div class="hero-subtitle">Book rooms, order food, request services, analyze reviews, and get voice AI support.</div>
            </div>
        </div>
        <div class="hero-badge">🚀 Premium Hotel AI</div>
    </div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("""
<div class="luxe-logo">
    <div class="luxe-logo-icon">H</div>
    <div>
        <div class="luxe-logo-title">LUXE STAY</div>
        <div class="luxe-logo-sub">HOTELS</div>
    </div>
</div>
<div class="side-label">Navigation</div>
""", unsafe_allow_html=True)

st.sidebar.caption(APP_VERSION)
page = st.sidebar.radio("Go to", ["🏠 Home", "💬 AI Hotel Assistant", "🖼 Hotel Gallery"])

profile = st.session_state.customer_profile
if profile.get("guest_name"):
    avatar = str(profile.get("guest_name", "G"))[0].upper()
    st.sidebar.markdown(f"""
    <div class="side-label">Current Customer</div>
    <div class="customer-card">
        <div class="customer-top">
            <div class="customer-avatar">{avatar}</div>
            <div>
                <div class="customer-name">{profile.get('guest_name')}</div>
                <div class="customer-id">Customer ID: {profile.get('customer_id')}</div>
            </div>
        </div>
        <div class="customer-row"><div class="icon">✉️</div><div><div class="label">Email</div><div class="value">{profile.get('email')}</div></div></div>
        <div class="customer-row"><div class="icon">📞</div><div><div class="label">Phone</div><div class="value">{profile.get('phone')}</div></div></div>
        <div class="customer-row"><div class="icon">🛏️</div><div><div class="label">Room</div><div class="value">{profile.get('room_number')}</div></div></div>
        <div class="customer-row"><div class="icon">📅</div><div><div class="label">Checkout</div><div class="value">{profile.get('check_out')}</div></div></div>
    </div>
    """, unsafe_allow_html=True)
else:
    st.sidebar.markdown("""
    <div class="side-label">Current Customer</div>
    <div class="customer-card">
        <div class="customer-top">
            <div class="customer-avatar">?</div>
            <div>
                <div class="customer-name">No active customer</div>
                <div class="customer-id">Book a room to create profile</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.sidebar.markdown('<div class="side-label">Staff Access</div>', unsafe_allow_html=True)
with st.sidebar.container():
    staff_password_input = st.text_input("Staff Password", type="password", key="staff_password_input")
    col_login, col_logout = st.columns(2)
    with col_login:
        if st.button("🔐 Login", key="staff_login_btn"):
            if staff_password_input == STAFF_PASSWORD:
                st.session_state.staff_logged_in = True
                st.success("Staff login successful.")
            else:
                st.error("Wrong staff password.")
    with col_logout:
        if st.button("Logout", key="staff_logout_btn"):
            st.session_state.staff_logged_in = False
            st.rerun()
    if st.session_state.staff_logged_in:
        st.success("Staff Mode: ON")
    else:
        st.warning("Staff Mode: OFF")

st.sidebar.markdown("""
<div class="staff-card">
    <b>🎧 Need human support?</b><br>
    <span style="font-size:13px;color:rgba(255,255,255,0.70)!important;">Connect with hotel team</span>
</div>
""", unsafe_allow_html=True)

# ---------------- HOME ----------------
if page == "🏠 Home":
    safe_image("assets/hotel_lobby.jpg", "Luxury Hotel Lobby")
    st.markdown("""
    ## ⭐ Luxury Hotel Experience
    This is an AI-powered hotel system where customers can book rooms, order food, request room service,
    analyze reviews, upload files/images, and get support through one AI Hotel Agent.
    """)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="metric-card"><h2>🤖 AI Agent</h2></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card"><h2>📊 ML Models</h2><p>Room recommendation and sentiment analysis</p></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card"><h2>🎙 Voice AI</h2><p>Voice-to-text support</p></div>', unsafe_allow_html=True)

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
                <p>📶 Free WiFi</p><p>🍽 Breakfast Included</p><p>🛏 Comfortable Stay</p>
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

    col_clear, col_checkout = st.columns(2)
    with col_clear:
        if st.button("🔄 Clear Chat / Start New Chat"):
            st.session_state.messages = []
            st.session_state.booking_data = {"guest_name": None, "email": None, "phone": None, "room_type": None, "check_in": None, "check_out": None, "guests": None}
            st.session_state.food_data = {"guest_name": None, "room_number": None, "food_item": None, "quantity": None}
            st.session_state.service_data = {"guest_name": None, "room_number": None, "service_type": None, "message": None}
            st.rerun()
    with col_checkout:
        if st.button("✅ Checkout / Clear Customer"):
            st.session_state.customer_profile = {"customer_id": None, "guest_name": None, "email": None, "phone": None, "room_number": None, "check_out": None}
            st.session_state.current_booking = None
            st.session_state.awaiting_payment = False
            st.success("Customer checked out and details cleared.")
            st.rerun()

    st.info("Use Enter to send. I can book rooms, order food, request services, recommend rooms, analyze reviews, files and images.")

    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("🏨 Book Room"):
            st.session_state.chat_prompt = "I need a deluxe room for 2 guests tomorrow"
    with c2:
        if st.button("🍽 Order Food"):
            st.session_state.chat_prompt = "Send 2 chicken biryanis to my room"
    with c3:
        if st.button("🛎 Room Service"):
            st.session_state.chat_prompt = "Need room cleaning"

    with st.expander("📌 Example Commands"):
        st.write("""
show rooms
recommend, 5000, 2, 3
book room
Madhu,maddalamadhuram3@gmail.com,8688087482,Standard Single,2026-12-20,2026-12-22,1
pay now
paid T2606201443044287007623
verify payment BK123456
Send 2 chicken biryanis to my room
Need room cleaning
checkout
        """)

    audio = mic_recorder(start_prompt="🎙 Start Recording", stop_prompt="⏹ Stop Recording", just_once=True)
    if audio:
        voice_text = convert_voice_to_text(audio)
        if voice_text.startswith("Voice-to-text error"):
            st.error(voice_text)
        else:
            st.session_state.voice_question = voice_text
            st.success(f"Recognized text: {voice_text}")

    uploaded_file = st.file_uploader("Upload file or image for AI analysis", type=["txt", "pdf", "jpg", "jpeg", "png"])
    file_content = ""
    if uploaded_file is not None:
        if uploaded_file.type == "text/plain":
            file_content = uploaded_file.read().decode("utf-8")
            st.success("Text file uploaded successfully.")
        elif uploaded_file.type == "application/pdf":
            file_content = extract_pdf_text(uploaded_file)
            if file_content.startswith("PDF library not installed") or file_content.startswith("PDF extraction error"):
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
        st.markdown(f'<div class="suggested-prompt">✨ <b>Suggested prompt</b><br>{st.session_state.chat_prompt}</div>', unsafe_allow_html=True)

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
            command = user_question.lower().strip()
            intent = detect_action(user_question)

            # Force full comma booking details to booking module.
            if "," in user_question:
                parts = [x.strip() for x in user_question.split(",")]
                if len(parts) >= 7 and "@" in user_question and re.search(r"\d{10}", user_question):
                    intent = "BOOK_ROOM"

            # If payment is pending, direct UPI transaction/reference IDs go to payment handler.
            if st.session_state.current_booking and st.session_state.awaiting_payment:
                txn_text = user_question.strip()
                if txn_text.lower().startswith("paid") or re.match(r"^[A-Za-z0-9]{10,}$", txn_text):
                    intent = "PAYMENT"

            if intent == "PAYMENT":
                if command.startswith("verify payment"):
                    if not st.session_state.staff_logged_in:
                        response_text = "Staff access required. Please login from Staff Access section in sidebar."
                    else:
                        parts = user_question.split()
                        booking_id = parts[-1].strip() if len(parts) >= 3 else ""
                        response_text = verify_payment_manually(booking_id)
                    stream_and_speak(response_text)
                elif st.session_state.current_booking:
                    txn_text = user_question.strip()
                    if txn_text.lower().startswith("paid"):
                        transaction_id = txn_text[4:].strip()
                    elif re.match(r"^[A-Za-z0-9]{10,}$", txn_text):
                        transaction_id = txn_text
                    else:
                        transaction_id = ""

                    if transaction_id:
                        response_text = mark_payment_paid(transaction_id)
                        stream_and_speak(response_text)
                    else:
                        show_payment_breakdown_and_qr(st.session_state.current_booking)
                        response_text = "Scan the UPI QR code and type your transaction/reference ID after payment."
                        stream_and_speak(response_text)
                else:
                    response_text = "No active booking found for payment. Please book a room first."
                    stream_and_speak(response_text)

            elif intent == "CHECKOUT":
                st.session_state.customer_profile = {"customer_id": None, "guest_name": None, "email": None, "phone": None, "room_number": None, "check_out": None}
                st.session_state.current_booking = None
                st.session_state.awaiting_payment = False
                response_text = "Checkout completed. Customer details cleared."
                stream_and_speak(response_text)

            elif uploaded_file is not None and file_content == "IMAGE_UPLOADED" and intent == "CHAT" and any(word in command for word in ["image", "photo", "picture", "summarize", "summary", "describe", "what is in", "analyze this"]):
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
                    response_text = book_room_from_chat(parts[1].strip(), parts[2].strip(), parts[3].strip(), parts[4].strip(), parts[5].strip(), parts[6].strip(), int(parts[7].strip()))
                    if "Room booked successfully" in response_text:
                        show_payment_breakdown_and_qr(st.session_state.current_booking)
                else:
                    response_text = "Use format: book room, Name, email, phone, Room Type, check_in, check_out, guests"
                stream_and_speak(response_text)

            elif command.startswith("place food,"):
                parts = user_question.split(",")
                if len(parts) == 5:
                    response_text = place_food_order(parts[1].strip(), parts[2].strip(), parts[3].strip(), int(parts[4].strip()))
                else:
                    response_text = "Use format: place food, Name, Room Number, Food Item, Quantity"
                stream_and_speak(response_text)

            elif command.startswith("service,"):
                parts = user_question.split(",")
                if len(parts) == 5:
                    response_text = create_service_request(parts[1].strip(), parts[2].strip(), parts[3].strip(), parts[4].strip())
                else:
                    response_text = "Use format: service, Name, Room Number, Service Type, Message"
                stream_and_speak(response_text)

            elif command.startswith("analyze review,"):
                review_text = user_question.split(",", 1)[1].strip()
                result = analyze_review_from_chat(review_text)
                response_text = f"Review Sentiment: {result}"
                stream_and_speak(response_text)

            elif intent == "BOOK_ROOM" or any(v is not None for v in st.session_state.booking_data.values()):
                booking = st.session_state.booking_data
                if "," in user_question:
                    parts = [x.strip() for x in user_question.split(",")]
                    if len(parts) >= 7:
                        st.session_state.booking_data = {
                            "guest_name": parts[0],
                            "email": parts[1],
                            "phone": re.sub(r"\D", "", parts[2]),
                            "room_type": parts[3],
                            "check_in": normalize_date(parts[4]),
                            "check_out": normalize_date(parts[5]),
                            "guests": parts[6],
                        }
                else:
                    missing = missing_fields(booking)
                    if missing:
                        if "@" in user_question:
                            booking["email"] = user_question.strip()
                        elif re.search(r"\d{10}", user_question):
                            booking["phone"] = re.search(r"\d{10}", user_question).group()
                        elif re.match(r"^\d{4}-\d{2}-\d{2}$", user_question) or re.match(r"^\d{2}-\d{2}-\d{4}$", user_question):
                            if not booking.get("check_in"):
                                booking["check_in"] = normalize_date(user_question)
                            else:
                                booking["check_out"] = normalize_date(user_question)
                        elif is_likely_name(user_question) and not booking.get("guest_name"):
                            booking["guest_name"] = user_question.strip()
                        st.session_state.booking_data = booking

                    extracted = smart_booking_parser(user_question, st.session_state.booking_data)
                    st.session_state.booking_data = update_dict(st.session_state.booking_data, extracted)

                st.session_state.booking_data = apply_customer_profile(st.session_state.booking_data)
                booking = st.session_state.booking_data

                if booking.get("guest_name"):
                    name = str(booking["guest_name"]).lower()
                    bad_name_words = ["book", "booking", "room", "want", "need", "standard", "deluxe", "suite"]
                    if any(word in name for word in bad_name_words) or len(name.split()) > 3:
                        booking["guest_name"] = None

                update_customer_profile(booking)
                missing = missing_fields(booking)

                if missing:
                    response_text = f"""
Current Booking Data:

Name: {booking['guest_name']}
Email: {booking['email']}
Phone: {booking['phone']}
Room Type: {booking['room_type']}
Check In: {booking['check_in']}
Check Out: {booking['check_out']}
Guests: {booking['guests']}

Please provide: {', '.join(missing)}
"""
                    stream_and_speak(response_text)
                else:
                    response_text = book_room_from_chat(booking["guest_name"], booking["email"], booking["phone"], booking["room_type"], booking["check_in"], booking["check_out"], int(booking["guests"]))
                    if "Room booked successfully" in response_text:
                        st.session_state.awaiting_payment = True
                        show_payment_breakdown_and_qr(st.session_state.current_booking)
                        st.session_state.booking_data = {"guest_name": None, "email": None, "phone": None, "room_type": None, "check_in": None, "check_out": None, "guests": None}
                    stream_and_speak(response_text)

            elif intent == "FOOD_ORDER":
                extracted = extract_details_with_groq(user_question, "food", st.session_state.food_data)
                st.session_state.food_data = update_dict(st.session_state.food_data, extracted)
                st.session_state.food_data = apply_customer_profile(st.session_state.food_data)
                missing = missing_fields(st.session_state.food_data)
                if missing:
                    response_text = "Please provide: " + ", ".join(missing)
                else:
                    data = st.session_state.food_data
                    response_text = place_food_order(data["guest_name"], data["room_number"], data["food_item"], int(data["quantity"]))
                    st.session_state.food_data = {"guest_name": None, "room_number": None, "food_item": None, "quantity": None}
                stream_and_speak(response_text)

            elif intent == "ROOM_SERVICE":
                extracted = extract_details_with_groq(user_question, "service", st.session_state.service_data)
                st.session_state.service_data = update_dict(st.session_state.service_data, extracted)
                st.session_state.service_data = apply_customer_profile(st.session_state.service_data)
                missing = missing_fields(st.session_state.service_data)
                if missing:
                    response_text = "Please provide: " + ", ".join(missing)
                else:
                    data = st.session_state.service_data
                    response_text = create_service_request(data["guest_name"], data["room_number"], data["service_type"], data["message"])
                    st.session_state.service_data = {"guest_name": None, "room_number": None, "service_type": None, "message": None}
                stream_and_speak(response_text)

            elif file_content and file_content != "IMAGE_UPLOADED":
                if not GROQ_API_KEY:
                    response_text = "Groq API Key not found."
                else:
                    client = Groq(api_key=GROQ_API_KEY)
                    response = client.chat.completions.create(
                        model=GROQ_MODEL,
                        messages=[
                            {"role": "system", "content": "Analyze the uploaded file content and answer the user question."},
                            {"role": "user", "content": f"{user_question}\n\nFile content:\n{file_content}"},
                        ],
                    )
                    response_text = response.choices[0].message.content
                stream_and_speak(response_text)

            else:
                if not GROQ_API_KEY:
                    response_text = "Groq API Key not found."
                else:
                    client = Groq(api_key=GROQ_API_KEY)
                    response = client.chat.completions.create(
                        model=GROQ_MODEL,
                        messages=[
                            {"role": "system", "content": "You are an AI hotel agent. Answer briefly. For booking/payment/food/service, do not invent confirmations. Let Python system handle actual booking and payment."},
                            {"role": "user", "content": user_question},
                        ],
                    )
                    response_text = response.choices[0].message.content
                stream_and_speak(response_text)

        st.session_state.messages.append({"role": "assistant", "content": response_text})


# ---------------- GALLERY ----------------
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
            safe_image(get_room_image(room["room_type"]), room["room_type"])
            st.markdown(f"""
            <div class="room-card">
                <h3>{room['room_type']}</h3>
                <h2>₹{room['price']} / night</h2>
                <p>📶 Free WiFi</p><p>🍽 Breakfast Included</p><p>🛏 Comfortable Stay</p>
            </div>
            """, unsafe_allow_html=True)
