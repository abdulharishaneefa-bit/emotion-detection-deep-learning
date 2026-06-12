# System Architecture — Emotion Detection from Text with Emoji Interaction

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [System Architecture Diagram](#2-system-architecture-diagram)
3. [Module Descriptions](#3-module-descriptions)
4. [Deep Learning Model Architecture](#4-deep-learning-model-architecture)
5. [Text Preprocessing Pipeline](#5-text-preprocessing-pipeline)
6. [Data Flow](#6-data-flow)
7. [Emoji Fusion Logic](#7-emoji-fusion-logic)
8. [Hardware & Software Requirements](#8-hardware--software-requirements)

---

## 1. Architecture Overview

The system is structured as a five-layer application:

```
┌────────────────────────────────────────────┐
│          Layer 1 — User Interface           │
│   HTML · CSS · JavaScript · Audio API       │
├────────────────────────────────────────────┤
│          Layer 2 — Application Layer        │
│          Flask Backend (app.py)             │
├────────────────────────────────────────────┤
│       Layer 3 — Deep Learning Layer         │
│   Text Preprocessing → BiLSTM → Softmax    │
├────────────────────────────────────────────┤
│        Layer 4 — Visualization Layer        │
│      Accuracy · Loss · Confusion Matrix     │
├────────────────────────────────────────────┤
│          Layer 5 — Storage Layer            │
│       .h5 Model · .pkl Tokenizer · CSV      │
└────────────────────────────────────────────┘
```

---

## 2. System Architecture Diagram

```
                  ┌─────────────────────┐
                  │     User Input       │
                  │  (Text via Browser)  │
                  └──────────┬──────────┘
                             │  HTTP POST /predict
                  ┌──────────▼──────────┐
                  │    Flask Backend     │
                  │     (app.py)         │
                  └──────────┬──────────┘
                             │
                  ┌──────────▼──────────┐
                  │  Text Preprocessing  │
                  │  · Lowercase         │
                  │  · Clean / Strip     │
                  │  · Tokenization      │
                  │  · Sequence Padding  │
                  └──────────┬──────────┘
                             │  Integer-padded sequence
                  ┌──────────▼──────────┐
                  │  BiLSTM Model        │
                  │  (emotion_lstm.h5)   │
                  │                      │
                  │  Embedding Layer     │
                  │       ↓              │
                  │  BiLSTM Layer        │
                  │       ↓              │
                  │  Dropout Layer       │
                  │       ↓              │
                  │  Dense Layer         │
                  │       ↓              │
                  │  Softmax [6 classes] │
                  └──────────┬──────────┘
                             │  argmax → emotion label + confidence
                  ┌──────────▼──────────┐
                  │   Emoji Mapping      │
                  │        Layer         │
                  └──────────┬──────────┘
                             │
          ┌──────────────────┼──────────────────┐
          │                  │                  │
 ┌────────▼──────┐  ┌────────▼──────┐  ┌────────▼──────┐
 │ Emoji Display │  │ Sound Effects │  │  Emoji Fusion  │
 │  + Confidence │  │ (MP3 Playback)│  │    System      │
 └───────────────┘  └───────────────┘  └───────────────┘

                  ┌──────────────────────┐
                  │  Performance Graphs  │
                  │  · Accuracy Graph    │
                  │  · Loss Graph        │
                  │  · Confusion Matrix  │
                  └──────────────────────┘
```

---

## 3. Module Descriptions

### Module 1 — User Interface

Handles all user-facing interactions: text input, prediction display, emoji animations, sound playback, and emoji fusion.

| Property | Value |
|---|---|
| Technologies | HTML5, CSS3, JavaScript |
| Audio | JavaScript `Audio()` API with MP3 files |
| Emoji Fusion | Client-side combination logic with predefined outcome map |
| Communication | `fetch()` POST requests to Flask `/predict` endpoint |

**Responsibilities:**
- Accept free-form text from the user
- Submit text to Flask backend via AJAX
- Render predicted emotion, emoji, and confidence score
- Trigger emotion-matched audio playback
- Handle emoji selection and fusion interaction

---

### Module 2 — Flask Backend

The central coordination layer that receives prediction requests, runs preprocessing, invokes the model, and returns structured JSON responses.

| Property | Value |
|---|---|
| Framework | Flask |
| Entry Point | `app.py` |
| Endpoint | `POST /predict` |
| Response Format | JSON `{ emotion, confidence, emoji }` |

**Responsibilities:**
- Load trained model (`emotion_lstm_model.h5`) on startup
- Load fitted tokenizer (`tokenizer.pkl`) on startup
- Accept and route prediction requests
- Return structured prediction results

---

### Module 3 — Text Preprocessing

Transforms raw user text into a fixed-length integer sequence suitable for the BiLSTM model.

| Step | Operation | Tool |
|---|---|---|
| 1 | Lowercase conversion | Python `str.lower()` |
| 2 | Special character removal | `re` (regex) |
| 3 | Tokenization | Keras `Tokenizer` |
| 4 | Sequence padding | Keras `pad_sequences()` |

**Pipeline:**

```
"I am SO happy today!!!"
        │
lowercase
        │
"i am so happy today"
        │
tokenizer.texts_to_sequences()
        │
[4, 12, 87, 23, 156]
        │
pad_sequences(maxlen=100)
        │
[0, 0, 0, ..., 4, 12, 87, 23, 156]  ← shape: (1, 100)
```

---

### Module 4 — Deep Learning Module

The BiLSTM model that learns semantic patterns in text to classify emotions.

See Section 4 for the full model architecture.

| Property | Value |
|---|---|
| Model Type | Bidirectional LSTM |
| Framework | TensorFlow / Keras |
| Saved Format | `.h5` (HDF5) |
| Output | 6-class probability distribution |
| Training Callback | EarlyStopping on `val_loss` |

---

### Module 5 — Emoji Fusion Module

Provides an interactive emoji combination feature independent of the emotion prediction system.

| Property | Value |
|---|---|
| Interaction | User selects two emojis |
| Output | Predefined fusion result emoji |
| Implementation | Client-side JavaScript lookup table |

---

### Module 6 — Sound Effect Module

Plays emotion-matched audio when a prediction is returned, and provides additional sounds for emoji fusion interactions.

| Emotion / Action | Audio File |
|---|---|
| Joy | `happy.mp3` |
| Sadness | `sad.mp3` |
| Anger | `angry.mp3` |
| Love | `love.mp3` |
| Fear | `fear.mp3` |
| Surprise | `surprise.mp3` |
| Fusion: Laugh | `laugh.mp3` |
| Fusion: Cool | `cool.mp3` |
| Fusion: Sleep | `sleep.mp3` |
| Fusion: Party | `party.mp3` |

---

### Module 7 — Visualization Module

Generates performance graphs during model training and stores them as static images served by Flask.

| Graph | Description | Library |
|---|---|---|
| Accuracy Graph | Training vs. validation accuracy per epoch | Matplotlib |
| Loss Graph | Training vs. validation loss per epoch | Matplotlib |
| Confusion Matrix | 6×6 heatmap of predictions vs. true labels | Seaborn |

---

## 4. Deep Learning Model Architecture

```
Input
  │  shape: (batch_size, max_sequence_length)
  │
  ▼
┌──────────────────────────────────────────┐
│             Embedding Layer              │
│                                          │
│  vocab_size × embedding_dim              │
│  Maps each word index to a dense vector  │
│  shape out: (batch, seq_len, embed_dim)  │
└──────────────────┬───────────────────────┘
                   │
┌──────────────────▼───────────────────────┐
│        Bidirectional LSTM Layer           │
│                                          │
│  Forward LSTM  ──────────────────────►   │
│                        ┌─────────────┐   │
│                        │concatenate  │   │
│                        └──────┬──────┘   │
│  Backward LSTM ◄─────────────────────    │
│                                          │
│  Output shape: (batch, 2 × lstm_units)   │
│  Captures both past and future context   │
└──────────────────┬───────────────────────┘
                   │
┌──────────────────▼───────────────────────┐
│              Dropout Layer               │
│   rate: 0.5  (prevents overfitting)      │
└──────────────────┬───────────────────────┘
                   │
┌──────────────────▼───────────────────────┐
│              Dense Layer                 │
│   units: 64, activation: ReLU            │
└──────────────────┬───────────────────────┘
                   │
┌──────────────────▼───────────────────────┐
│           Softmax Output Layer           │
│   units: 6 (one per emotion class)       │
│   activation: softmax                    │
│                                          │
│   Output: [p_joy, p_sadness, p_love,     │
│            p_anger, p_fear, p_surprise]  │
└──────────────────┬───────────────────────┘
                   │
                argmax
                   │
          Predicted emotion label
          + Confidence score (max probability)
```

**Training Configuration:**

| Parameter | Value |
|---|---|
| Loss Function | Categorical Crossentropy |
| Optimizer | Adam |
| Batch Size | 32 (default) |
| Early Stopping | Monitor `val_loss`, patience = 3 |
| Test Accuracy | ~96% |

**Why Bidirectional LSTM?**

In emotion analysis, the emotional tone of a word depends on surrounding context in both directions. A standard LSTM only sees left-to-right context. A BiLSTM processes the sequence both forward and backward, then concatenates both outputs — giving the model a complete view of the sentence before making its prediction.

Example:
```
"I thought it would be great, but it wasn't at all."
                                              ↑
                  Forward LSTM misses this negation when encoding early words.
                  Backward LSTM captures it and corrects the representation.
```

---

## 5. Text Preprocessing Pipeline

```
Raw Input Text
       │
       ▼
  str.lower()               "I Am SO Angry!" → "i am so angry!"
       │
       ▼
  re.sub(r'[^a-z\s]', '')  "i am so angry!" → "i am so angry"
       │
       ▼
  tokenizer.texts_to_sequences()
                             "i am so angry" → [4, 12, 87, 99]
       │
       ▼
  pad_sequences(maxlen=100, padding='post')
                             [4, 12, 87, 99] → [4, 12, 87, 99, 0, 0, ..., 0]
                                                shape: (1, 100)
       │
       ▼
  model.predict()
                             → [0.02, 0.01, 0.01, 0.94, 0.01, 0.01]
       │
       ▼
  argmax → index 3 → "Anger" (confidence: 94%)
```

The tokenizer is fitted on the training corpus during `train_model.py` and saved as `tokenizer.pkl` for consistent vocabulary mapping at inference time.

---

## 6. Data Flow

```
Step 1   User enters text in the browser input field

Step 2   JavaScript sends POST /predict with { "text": "..." }

Step 3   Flask receives request and extracts text

Step 4   Text preprocessing: lowercase → clean → tokenize → pad

Step 5   Padded sequence fed into BiLSTM model

Step 6   Softmax layer outputs 6 probabilities

Step 7   argmax selects highest probability → emotion label

Step 8   Emotion label mapped to emoji + confidence score

Step 9   JSON response returned: { emotion, emoji, confidence }

Step 10  Frontend renders emoji, confidence bar, and emotion label

Step 11  JavaScript plays corresponding emotion .mp3 audio

Step 12  User can optionally trigger emoji fusion interaction
```

---

## 7. Emoji Fusion Logic

The emoji fusion system is a client-side interactive feature independent of the ML pipeline.

```
User selects Emoji A + Emoji B
             │
             ▼
  JavaScript fusion lookup table
  { "A+B": "result_emoji", ... }
             │
             ▼
  Predefined fusion result displayed
             │
             ▼
  Optional fusion sound played (laugh / cool / party / sleep)
```

Fusion outcomes are predefined combinations — the system maps specific emoji pairs to creative result emojis, providing a playful interactive layer on top of the core prediction feature.

---

## 8. Hardware & Software Requirements

### Hardware

| Component | Minimum | Recommended |
|---|---|---|
| Processor | Intel Core i3 | Intel Core i5/i7 |
| RAM | 4 GB | 8 GB+ |
| Storage | 2 GB free | SSD |
| GPU | — | NVIDIA (CUDA, for training) |

### Software

| Package | Purpose |
|---|---|
| Python 3.10+ | Runtime |
| Flask | Web framework |
| TensorFlow / Keras | Model training and inference |
| Pandas | Dataset loading and manipulation |
| NumPy | Numerical operations |
| Scikit-learn | Train/test split, metrics |
| Matplotlib | Accuracy and loss graphs |
| Seaborn | Confusion matrix heatmap |
| Joblib | Tokenizer serialization (`.pkl`) |

---

*Architecture documented by Abdul Haris H — M.Tech CSE, Government Engineering College Thrissur, APJ Abdul Kalam Technological University*