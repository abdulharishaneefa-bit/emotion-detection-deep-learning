import os
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import re
import string
from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Embedding, LSTM, Dense, Dropout, Bidirectional,
    Input, GlobalMaxPooling1D, GlobalAveragePooling1D,
    Concatenate, BatchNormalization, Conv1D, MultiHeadAttention,
    LayerNormalization, Add
)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l2
import tensorflow as tf

# ─── CONFIG ───────────────────────────────────────────────────────────────────
MAXLEN      = 50
VOCAB_SIZE  = 20000
EMBED_DIM   = 256
BATCH_SIZE  = 32
EPOCHS      = 10
MODEL_PATH  = "emotion_model_v2.keras"
TOKENIZER_PATH = "tokenizer_v2.pkl"

emotion_labels = {0: "sadness", 1: "joy", 2: "love", 3: "anger", 4: "fear", 5: "surprise"}

# ─── TEXT CLEANING ─────────────────────────────────────────────────────────────
contractions = {
    "won't": "will not", "can't": "cannot", "i'm": "i am", "i've": "i have",
    "i'll": "i will", "i'd": "i would", "it's": "it is", "that's": "that is",
    "there's": "there is", "they're": "they are", "they've": "they have",
    "we're": "we are", "we've": "we have", "don't": "do not", "doesn't": "does not",
    "didn't": "did not", "isn't": "is not", "aren't": "are not", "wasn't": "was not",
    "weren't": "were not", "hasn't": "has not", "haven't": "have not",
    "hadn't": "had not", "wouldn't": "would not", "couldn't": "could not",
    "shouldn't": "should not", "you're": "you are", "you've": "you have",
    "you'll": "you will", "you'd": "you would", "he's": "he is", "she's": "she is",
    "let's": "let us", "what's": "what is", "who's": "who is", "how's": "how is",
    "here's": "here is", "where's": "where is", "when's": "when is",
    "ain't": "am not", "shan't": "shall not", "mayn't": "may not",
}

def expand_contractions(text):
    for key, val in contractions.items():
        text = text.replace(key, val)
    return text

def clean_text(text):
    text = str(text).lower().strip()
    text = expand_contractions(text)
    # Keep letters, spaces and basic punctuation for emotion signals
    text = re.sub(r"http\S+|www\S+", "", text)          # remove URLs
    text = re.sub(r"@\w+|#\w+", "", text)               # remove mentions/hashtags
    text = re.sub(r"[^a-z\s!?.,']", " ", text)          # keep core punctuation
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ─── AUGMENTATION ─────────────────────────────────────────────────────────────
def augment_text(text):
    """Simple synonym-free augmentation: random word drop."""
    words = text.split()
    if len(words) <= 3:
        return text
    drop_idx = np.random.randint(0, len(words))
    words.pop(drop_idx)
    return " ".join(words)

# ─── LOAD & PREPROCESS DATA ───────────────────────────────────────────────────
print("Loading data...")
train_df = pd.read_csv("training.csv")

# Support both 'label' and 'emotion' column names
label_col = "label" if "label" in train_df.columns else "emotion"
text_col  = "text"  if "text"  in train_df.columns else train_df.columns[0]

train_df = train_df[[text_col, label_col]].dropna()
train_df.columns = ["text", "label"]

# If labels are strings, encode them
if train_df["label"].dtype == object:
    label_map = {v: k for k, v in emotion_labels.items()}
    train_df["label"] = train_df["label"].map(label_map)
    train_df = train_df.dropna(subset=["label"])
    train_df["label"] = train_df["label"].astype(int)

train_df["text"] = train_df["text"].apply(clean_text)

# Data augmentation for minority classes
print("Augmenting minority classes...")
class_counts = Counter(train_df["label"])
max_count = max(class_counts.values())
augmented_rows = []

for label, count in class_counts.items():
    if count < max_count * 0.7:  # augment classes with less than 70% of max
        subset = train_df[train_df["label"] == label]
        needed = int(max_count * 0.7) - count
        for _ in range(needed):
            row = subset.sample(1).iloc[0]
            augmented_rows.append({"text": augment_text(row["text"]), "label": label})

if augmented_rows:
    aug_df = pd.DataFrame(augmented_rows)
    train_df = pd.concat([train_df, aug_df], ignore_index=True).sample(frac=1).reset_index(drop=True)

print(f"Total training samples: {len(train_df)}")
print("Class distribution:", Counter(train_df["label"]))

# Split
texts_np  = train_df["text"].to_numpy()
labels_np = train_df["label"].to_numpy()

X_train_raw, X_val_raw, y_train, y_val = train_test_split(
    texts_np,
    labels_np,
    test_size=0.15,
    stratify=labels_np,
    random_state=42
)
# Tokenizer
tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>", char_level=False)
tokenizer.fit_on_texts(X_train_raw)

X_train = pad_sequences(tokenizer.texts_to_sequences(X_train_raw), maxlen=MAXLEN, padding="post", truncating="post")
X_val   = pad_sequences(tokenizer.texts_to_sequences(X_val_raw),   maxlen=MAXLEN, padding="post", truncating="post")

y_train = np.array(y_train)
y_val   = np.array(y_val)

# Class weights
cw = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
class_weight_dict = dict(enumerate(cw))
print("Class weights:", class_weight_dict)

# ─── MODEL: CNN + BiLSTM + Self-Attention ─────────────────────────────────────
def build_model():
    inp = Input(shape=(MAXLEN,))

    # Embedding
    x = Embedding(VOCAB_SIZE, EMBED_DIM, mask_zero=False)(inp)
    x = Dropout(0.3)(x)

    # CNN branch — local n-gram features
    conv = Conv1D(128, 5, padding="same", activation="relu")(x)
    conv = BatchNormalization()(conv)

    # BiLSTM branch
    lstm_out = Bidirectional(LSTM(128, return_sequences=True, dropout=0.3, recurrent_dropout=0.2))(x)
    lstm_out = Bidirectional(LSTM(64,  return_sequences=True, dropout=0.3, recurrent_dropout=0.2))(lstm_out)

    # Combine CNN + BiLSTM
    merged = Add()([conv[:, :, :128], lstm_out[:, :, :128]])  # match dims

    # Self-Attention
    attn_out = MultiHeadAttention(num_heads=4, key_dim=32)(merged, merged)
    attn_out = LayerNormalization()(attn_out + merged)

    # Pooling
    avg_pool = GlobalAveragePooling1D()(attn_out)
    max_pool = GlobalMaxPooling1D()(attn_out)
    pooled = Concatenate()([avg_pool, max_pool])

    # Dense head
    x = Dense(256, activation="relu", kernel_regularizer=l2(1e-4))(pooled)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    x = Dense(128, activation="relu", kernel_regularizer=l2(1e-4))(x)
    x = Dropout(0.3)(x)
    out = Dense(6, activation="softmax")(x)

    model = Model(inp, out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

model = build_model()
model.summary()

# ─── CALLBACKS ────────────────────────────────────────────────────────────────
callbacks = [
    EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6, verbose=1),
    ModelCheckpoint(MODEL_PATH, monitor="val_accuracy", save_best_only=True, verbose=1)
]

# ─── TRAIN ────────────────────────────────────────────────────────────────────
print("\nTraining model...")
history = model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_val, y_val),
    callbacks=callbacks,
    class_weight=class_weight_dict,
    verbose=1
)

# ─── PLOTS ────────────────────────────────────────────────────────────────────
os.makedirs("static", exist_ok=True)

plt.figure(figsize=(9, 5))
plt.plot(history.history["accuracy"],     label="Train Accuracy", linewidth=2)
plt.plot(history.history["val_accuracy"], label="Val Accuracy",   linewidth=2, linestyle="--")
plt.title("Model Accuracy", fontsize=15)
plt.xlabel("Epochs"); plt.ylabel("Accuracy")
plt.legend(); plt.grid(True, alpha=0.4)
plt.tight_layout()
plt.savefig("static/accuracy_graph.png", dpi=150)
plt.show()

plt.figure(figsize=(9, 5))
plt.plot(history.history["loss"],     label="Train Loss", linewidth=2, color="tomato")
plt.plot(history.history["val_loss"], label="Val Loss",   linewidth=2, linestyle="--", color="orange")
plt.title("Model Loss", fontsize=15)
plt.xlabel("Epochs"); plt.ylabel("Loss")
plt.legend(); plt.grid(True, alpha=0.4)
plt.tight_layout()
plt.savefig("static/loss_graph.png", dpi=150)
plt.show()

# ─── FINAL EVAL ───────────────────────────────────────────────────────────────
loss, acc = model.evaluate(X_val, y_val, verbose=0)
print(f"\n✅ Validation Accuracy: {acc*100:.2f}%")

model.save(MODEL_PATH)
joblib.dump(tokenizer, TOKENIZER_PATH)
print("✅ Model and tokenizer saved!")