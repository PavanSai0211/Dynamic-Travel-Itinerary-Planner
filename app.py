# gradio_travel_planner.py

import os
import json
import sqlite3
import google.generativeai as genai
import requests
from dotenv import load_dotenv
from datetime import datetime
import logging
import re
import gradio as gr

# ---------------- Logging ------------------
logging.basicConfig(
    filename="travel_planner.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# --------------- Load Env Vars ---------------
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
WEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")

# -------------- Gemini Config ----------------
genai.configure(api_key=GOOGLE_API_KEY)

# --------------- SQLite Setup ----------------
DB_FILE = "trip_plans.db"

def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS trips (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            user_input TEXT,
            destination TEXT,
            json_data TEXT
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS trip_costs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            model TEXT,
            prompt_tokens INTEGER,
            completion_tokens INTEGER,
            total_tokens INTEGER,
            cost_usd REAL
        )
    """)
    conn.commit()
    conn.close()

init_db()

# -------- Store Trip Info --------
def store_trip(user_input, destination, json_data):
    try:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute("INSERT INTO trips (timestamp, user_input, destination, json_data) VALUES (?, ?, ?, ?)", (
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            user_input,
            destination,
            json.dumps(json_data)
        ))
        conn.commit()
        conn.close()
        logging.info(f"Trip stored: {destination}")
    except Exception as e:
        logging.error(f"Failed to store trip: {e}")

# -------- Store Token Costs --------
def store_token_cost(model, prompt_tokens, completion_tokens):
    if prompt_tokens <= 128000:
        input_cost = prompt_tokens * 0.075 / 1000000
    else:
        input_cost = prompt_tokens * 0.15 / 1000000

    if completion_tokens <= 128000:
        output_cost = completion_tokens * 0.30 / 1000000
    else:
        output_cost = completion_tokens * 0.60 / 1000000

    total_cost = round(input_cost + output_cost, 6)

    try:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute(""" 
            INSERT INTO trip_costs (timestamp, model, prompt_tokens, completion_tokens, total_tokens, cost_usd)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3],
            model,
            prompt_tokens,
            completion_tokens,
            prompt_tokens + completion_tokens,
            total_cost
        ))
        conn.commit()
        conn.close()
        logging.info(f"Token cost stored: {prompt_tokens}+{completion_tokens} = {total_cost}")
    except Exception as e:
        logging.error(f"Failed to store token cost: {e}")

# ------------- Weather API -------------------
def get_weather(city):
    try:
        url = f"https://api.openweathermap.org/data/2.5/forecast?q={city}&appid={WEATHER_API_KEY}&units=metric"
        response = requests.get(url)
        data = response.json()

        if data["cod"] != "200":
            return "Could not retrieve weather info."

        forecast = ""
        for i in range(0, 40, 8):
            day = data["list"][i]
            date = day["dt_txt"].split(" ")[0]
            desc = day["weather"][0]["description"].title()
            temp = day["main"]["temp"]
            humidity = day["main"]["humidity"]
            wind = day["wind"]["speed"]
            forecast += f"\n {date}: {desc}, {temp}°C, {humidity}%, {wind} km/h"

        return forecast
    except Exception as e:
        logging.error(f"Weather fetch error: {e}")
        return "Weather info unavailable."

# ----------- Greeting & Keyword Detection ----------- 
greeting_keywords = ["hi", "hello", "hey", "good morning", "good evening", "how are you"]
thank_keywords = ["thank you", "thanks", "thankyou", "thx", "appreciate", "grateful"]
non_travel_keywords = ["weather", "news", "joke", "movie", "recipe", "code", "sports"]

def is_greeting(text):
    return any(word in text.lower() for word in greeting_keywords)

def is_thank_you(text):
    return any(word in text.lower() for word in thank_keywords)

def is_non_travel_query(text):
    return any(word in text.lower() for word in non_travel_keywords)

def extract_destination(text):
    patterns = [r"trip to ([a-zA-Z\s]+)", r"visit ([a-zA-Z\s]+)", r"go to ([a-zA-Z\s]+)", r"([a-zA-Z\s]+)$"]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            destination = match.group(1).strip().title()
            if len(destination.split()) <= 3:
                return destination
    return None

# ----------- Trip Generation Logic ----------- 
chat_history = []
last_destination = None
stored_destinations = []

def generate_trip_response(user_input):
    global last_destination, chat_history, stored_destinations

    chat_history.append({"role": "user", "parts": [user_input]})
    destination = extract_destination(user_input)
    if not destination and last_destination:
        destination = last_destination

    if not destination:
        if is_thank_you(user_input):
            return "You're very welcome! 😊 I'm always here to help with your travel plans!"
        elif is_greeting(user_input):
            return "👋 Hello! I'm your friendly Travel Planner. Tell me where you'd like to go!"
        elif is_non_travel_query(user_input):
            return "I'm focused on helping with travel planning. Ask me about your next trip! 🌍"
        else:
            return "Please tell me a destination you'd like to travel to."

    last_destination = destination
    if destination not in stored_destinations:
        stored_destinations.append(destination)

    system_prompt = """
    You are an expert travel planner assistant.
    When a user asks about a place, respond with:
    1. A friendly markdown travel guide including:
        - Overview
        - Suggested itinerary
        - Attractions
        - Budget tips
        - Hotel and restaurant recommendations
    2. A structured JSON with the following keys:
        - destination
        - overview
        - itinerary
        - attractions
        - budget
        - hotels
        - restaurants
    DO NOT include weather info in the JSON — that will be added separately.
    If the user gives trip details (people, days), ask for missing info politely if needed.
    If the user does not specify number of people or number of days ask them politely.
    Always be friendly and follow previous trip context if no new destination is mentioned.
    If the user says thanks and all respond in a friendly manner saying i am always here to help you.
    dont answer questions that are not related to trip and planning, make sure to give a decent replay for not answering.
    """

    try:
        chat = genai.GenerativeModel("gemini-2.0-flash").start_chat(history=chat_history)
        gemini_response = chat.send_message([system_prompt, user_input]).text

        if "json" in gemini_response:
            markdown_part, json_part = gemini_response.split("json")
            json_part = json_part.split("```")[0].strip()
        else:
            markdown_part = gemini_response
            json_part = "{}"

        structured_data = json.loads(json_part)

        if "itinerary" in structured_data and destination:
            weather_data = get_weather(destination)
            markdown_part += f"\n\n🌦 Weather Forecast for {destination} (Next 5 days):\n{weather_data}"
            structured_data["weather"] = weather_data
            structured_data["destination"] = destination
            store_trip(user_input, destination, structured_data)

        prompt_tokens = sum(len(m["parts"][0].split()) for m in chat_history if m["role"] == "user")
        completion_tokens = len(gemini_response.split())
        store_token_cost("gemini-1.5-flash", prompt_tokens, completion_tokens)

        chat_history.append({"role": "assistant", "parts": [markdown_part]})
        return markdown_part

    except Exception as e:
        logging.error(f"Gemini response error: {e}")
        return "Sorry, something went wrong while planning your trip."

# ---------------- Gradio Interface ----------------
with gr.Blocks() as demo:
    gr.Markdown("# 🧳 Dynamic Travel Planner Chat")
    gr.Markdown("Chat with an AI travel assistant. Get full itinerary, weather, hotels & more.")

    chatbot = gr.Chatbot()
    msg = gr.Textbox(placeholder="Where are you planning to go?")

    def respond(message, history):
        reply = generate_trip_response(message)
        history.append((message, reply))
        return history, ""

    msg.submit(respond, [msg, chatbot], [chatbot, msg])

if __name__ == "__main__":
    demo.launch()