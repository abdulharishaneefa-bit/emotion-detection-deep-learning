from flask import Flask, render_template, request
import numpy as np
import joblib
import re
import os

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

# ---------------- LOAD MODEL ----------------
MODEL_PATH = "emotion_model_v2.keras"
TOKENIZER_PATH = "tokenizer_v2.pkl"

model = load_model(MODEL_PATH)
tokenizer = joblib.load(TOKENIZER_PATH)

MAXLEN = 50

emotion_labels = {
    0: "sadness",
    1: "joy",
    2: "love",
    3: "anger",
    4: "fear",
    5: "surprise"
}

emoji_map = {
    "sadness": "😢",
    "joy": "😊",
    "love": "❤️",
    "anger": "😠",
    "fear": "😨",
    "surprise": "😲"
}

message_map = {
    "sadness": "The text carries a tone of sadness or grief.",
    "joy": "The text radiates happiness and positivity!",
    "love": "The text expresses love, warmth, or affection.",
    "anger": "The text conveys anger or frustration.",
    "fear": "The text shows signs of fear or anxiety.",
    "surprise": "The text expresses surprise or astonishment."
}

sound_map = {
    "sadness": "sad.mp3",
    "joy": "joy.mp3",
    "love": "love.mp3",
    "anger": "angry.mp3",
    "fear": "fear.mp3",
    "surprise": "surprise.mp3"
}

# ---------------- TEXT CLEANING ----------------
contractions = {
    "won't": "will not",
    "can't": "cannot",
    "i'm": "i am",
    "i've": "i have",
    "i'll": "i will",
    "i'd": "i would",
    "it's": "it is",
    "that's": "that is",
    "there's": "there is",
    "they're": "they are",
    "don't": "do not",
    "doesn't": "does not",
    "didn't": "did not",
    "isn't": "is not",
    "aren't": "are not",
    "wasn't": "was not",
    "weren't": "were not",
    "hasn't": "has not",
    "haven't": "have not",
    "hadn't": "had not",
    "wouldn't": "would not",
    "couldn't": "could not",
    "shouldn't": "should not",
    "you're": "you are",
    "you've": "you have",
    "you'll": "you will",
    "he's": "he is",
    "she's": "she is",
    "let's": "let us",
    "what's": "what is",
    "ain't": "am not",
    "we're": "we are",
    "we've": "we have",
    "you'd": "you would"
}

def expand_contractions(text):
    for short, full in contractions.items():
        text = text.replace(short, full)
    return text

def clean_text(text):
    text = str(text).lower().strip()
    text = expand_contractions(text)
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+|#\w+", "", text)
    text = re.sub(r"[^a-z\s!?.,']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ---------------- KEYWORD SUPPORT ----------------
keyword_map = {
    "joy": [
        "happy", "happiness", "excited", "great", "amazing", "awesome", "best",
        "wonderful", "good", "smile", "enjoy", "fantastic", "glad", "joyful",
        "thrilled", "cheerful", "delighted", "perfect", "excellent", "fun",
        "laugh", "lol", "haha", "yay", "woohoo", "great day", "feeling good"
    ],
    "sadness": [
        "sad", "lonely", "cry", "crying", "broken", "depressed", "unhappy",
        "miss", "terrible", "bad", "hurt", "pain", "grief", "tears",
        "heartbroken", "miserable", "hopeless", "down", "alone", "empty",
        "disappointed"
    ],
    "anger": [
        "angry", "mad", "hate", "annoying", "frustrated", "irritated", "upset",
        "furious", "rage", "offended", "fed up", "sick of", "enough",
        "ridiculous", "unfair", "stupid", "idiot", "damn", "worst"
    ],
    "love": [
        "love", "care", "dear", "special", "heart", "affection", "like you",
        "miss you", "adore", "cherish", "romantic", "darling", "sweetheart",
        "forever", "together", "beautiful", "precious", "lovely", "my heart",
        "with you", "hugs"
    ],
    "fear": [
        "scared", "afraid", "fear", "worried", "nervous", "danger", "unsafe",
        "panic", "anxious", "terror", "frightened", "nightmare", "threat",
        "helpless", "stress", "uneasy", "shaking", "alarmed", "tense"
    ],
    "surprise": [
        "wow","shocked","unexpected","unbelievable","surprised","suddenly","omg",
        "no way","really","seriously","what","incredible","astonishing","astounding",
        "amazing","stunned","speechless","jaw drop","didnt expect","did not expect",
        "never thought","out of nowhere","cant believe","mind blown","whoa","holy",
        "oh my","i did not expect this","i didnt expect this"
    ]
}

def keyword_score(text):
    scores = {emotion: 0 for emotion in keyword_map}
    text_lower = text.lower()

    for emotion, words in keyword_map.items():
        for word in words:
            if word in text_lower:
                scores[emotion] += 2 if " " in word else 1

    return scores

# ---------------- PREDICTION ----------------
def predict_emotion(raw_text):
    cleaned = clean_text(raw_text)

    sequence = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(
        sequence,
        maxlen=MAXLEN,
        padding="post",
        truncating="post"
    )

    probs = model.predict(padded, verbose=0)[0]

    model_index = int(np.argmax(probs))
    model_emotion = emotion_labels[model_index]
    model_confidence = float(probs[model_index]) * 100

    kw_scores = keyword_score(cleaned)
    kw_total = sum(kw_scores.values())

    if kw_total > 0:
        kw_probs = np.array([
            kw_scores[emotion_labels[i]] / kw_total for i in range(6)
        ])
    else:
        kw_probs = np.zeros(6)

    if model_confidence >= 60:
        blended = 0.65 * probs + 0.35 * kw_probs
    else:
        blended = 0.50 * probs + 0.50 * kw_probs

    final_index = int(np.argmax(blended))
    final_emotion = emotion_labels[final_index]
    final_confidence = float(blended[final_index]) * 100

    # Short text correction
    if len(cleaned.split()) <= 3 and kw_total > 0:
        keyword_winner = max(kw_scores, key=kw_scores.get)
        if kw_scores[keyword_winner] > 0:
            final_emotion = keyword_winner
            final_confidence = max(final_confidence, 75.0)

    emoji = emoji_map.get(final_emotion, "🙂")
    message = message_map.get(final_emotion, "Emotion detected.")
    sound_file = sound_map.get(final_emotion, "")

    top3_indexes = sorted(range(6), key=lambda i: blended[i], reverse=True)[:3]

    top3_emotions = [
        {
            "emotion": emotion_labels[i].capitalize(),
            "confidence": round(float(blended[i]) * 100, 1)
        }
        for i in top3_indexes
    ]

    return final_emotion, emoji, round(final_confidence, 2), message, top3_emotions, sound_file

# ---------------- ROUTE ----------------
@app.route("/", methods=["GET", "POST"])
def home():
    emotion = None
    emoji = None
    confidence = None
    message = None
    sound_file = None
    top3_emotions = []
    user_text = ""

    if request.method == "POST":
        user_text = request.form.get("text", "").strip()

        if user_text:
            emotion, emoji, confidence, message, top3_emotions, sound_file = predict_emotion(user_text)

    return render_template(
        "index.html",
        emotion=emotion,
        emoji=emoji,
        confidence=confidence,
        message=message,
        user_text=user_text,
        top3_emotions=top3_emotions,
        sound_file=sound_file
    )

if __name__ == "__main__":
    app.run(debug=True)