<div align="center">

<h1>🎭 Emotion Detection from Text</h1>
<h3>with Emoji Interaction — Powered by Deep Learning</h3>

<p><strong>BiLSTM-based text emotion classifier with emoji fusion, audio feedback, and performance analytics</strong></p>

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-Deep%20Learning-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![Keras](https://img.shields.io/badge/Keras-BiLSTM-D00000?style=for-the-badge&logo=keras&logoColor=white)](https://keras.io)
[![Flask](https://img.shields.io/badge/Flask-Web%20App-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com)
[![Accuracy](https://img.shields.io/badge/Test%20Accuracy-~96%25-brightgreen?style=for-the-badge)](https://github.com)
[![License](https://img.shields.io/badge/License-Academic-blue?style=for-the-badge)](LICENSE)

<br/>

> Language carries emotion — this system reads it. Enter any text and watch a deep learning model decode the feeling behind the words, respond with emojis, play emotion-matched audio, and show you exactly how confident it is.

</div>

---

## 📌 Overview

**Emotion Detection from Text with Emoji Interaction** is a deep learning web application that classifies the emotional content of free-form text into one of six categories — then brings that prediction to life through emoji mapping, emoji fusion, sound effects, and model performance visualizations.

At its core sits a **Bidirectional LSTM (BiLSTM)** network trained on labeled text data, achieving ~96% test accuracy. The Flask backend serves predictions in real time, while the interactive frontend provides an engaging, multimedia-rich user experience.

Built as an M.Tech project at **Government Engineering College Thrissur**, APJ Abdul Kalam Technological University.

---

## 🎯 Supported Emotion Classes

| Emotion | Emoji | Description |
|---|---|---|
| **Joy** | 😊 | Happiness, excitement, positivity |
| **Sadness** | 😢 | Grief, disappointment, sorrow |
| **Love** | ❤️ | Affection, warmth, care |
| **Anger** | 😠 | Frustration, rage, irritation |
| **Fear** | 😨 | Anxiety, dread, nervousness |
| **Surprise** | 😲 | Shock, astonishment, disbelief |

---

## ✨ Features

**Emotion Detection**
- Predicts emotion from any user-entered text
- Displays confidence score alongside the prediction
- Maps emotion to corresponding emoji instantly

**Deep Learning Model**
- Bidirectional LSTM architecture captures context in both directions
- Word tokenization + sequence padding pipeline
- Early stopping to prevent overfitting
- ~96% test accuracy on held-out data

**Emoji Interaction**
- Emotion-to-emoji mapping for every prediction
- Interactive emoji combination/fusion system
- Multiple predefined emoji fusion outcomes

**Sound Effects**
- Emotion-matched audio playback (MP3)
- Bonus sounds: laugh, cool, sleep, party

**Performance Visualization**
- Training accuracy graph
- Training loss graph
- Confusion matrix (6 × 6)

---

## 🏗️ System Architecture

```
                        ┌─────────────────────┐
                        │     User Input       │
                        │  (Text via Browser)  │
                        └──────────┬──────────┘
                                   │  HTTP POST
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
                                   │  Padded sequence
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
                        │  Softmax Output      │
                        │  [6 probabilities]   │
                        └──────────┬──────────┘
                                   │  argmax → emotion label
                        ┌──────────▼──────────┐
                        │   Emoji Mapping      │
                        │   Layer              │
                        └──────────┬──────────┘
                                   │
                   ┌───────────────┼───────────────┐
                   │               │               │
          ┌────────▼──────┐ ┌──────▼──────┐ ┌──────▼──────┐
          │  Emoji Display│ │Sound Effects│ │ Emoji Fusion │
          │  + Confidence │ │  (MP3)      │ │  System      │
          └───────────────┘ └─────────────┘ └─────────────┘
                   │
          ┌────────▼──────────────┐
          │  Performance Graphs   │
          │  Accuracy · Loss · CM │
          └───────────────────────┘
```

---

## 🧠 Model Architecture

```
Input Text
    │
    ▼
┌─────────────────────────────────────┐
│         Embedding Layer             │
│  vocab_size × embedding_dim         │
│  Converts word indices → dense vecs │
└──────────────────┬──────────────────┘
                   │
┌──────────────────▼──────────────────┐
│      Bidirectional LSTM Layer       │
│                                     │
│  Forward LSTM ──────────────────►   │
│                      ┌──────────┐   │
│                      │  concat  │   │
│                      └────┬─────┘   │
│  Backward LSTM ◄────────────────    │
│                                     │
│  Captures context from BOTH sides   │
└──────────────────┬──────────────────┘
                   │
┌──────────────────▼──────────────────┐
│           Dropout Layer             │
│    Regularization to prevent        │
│    overfitting during training      │
└──────────────────┬──────────────────┘
                   │
┌──────────────────▼──────────────────┐
│            Dense Layer              │
│     Fully connected, ReLU           │
└──────────────────┬──────────────────┘
                   │
┌──────────────────▼──────────────────┐
│          Softmax Output             │
│   6 neurons → 6 probabilities       │
│  [joy, sadness, love, anger,        │
│   fear, surprise]                   │
└──────────────────┬──────────────────┘
                   │
                argmax
                   │
           Predicted Emotion
           + Confidence Score
```

**Why BiLSTM?**

Standard LSTMs only read text left-to-right. A Bidirectional LSTM reads the sequence in both directions simultaneously, giving the model access to full sentence context when predicting — critical for emotion, where a word's meaning often depends on what comes *after* it just as much as before.

---

## 🔄 Prediction Pipeline

```
1. Raw text input
        │
2. Lowercase + clean (remove special chars)
        │
3. Keras Tokenizer → integer sequence
        │
4. pad_sequences() → fixed-length array
        │
5. BiLSTM model.predict() → [p1, p2, p3, p4, p5, p6]
        │
6. argmax → emotion index → emotion label
        │
7. Emoji mapping → emoji + confidence score
        │
8. Audio trigger → play emotion.mp3
        │
9. Return JSON response to frontend
```

---

## 📊 Model Performance

| Metric | Value |
|---|---|
| Test Accuracy | ~96% |
| Loss Function | Categorical Crossentropy |
| Optimizer | Adam |
| Early Stopping | Yes (monitors val_loss) |
| Emotion Classes | 6 |

Performance graphs generated during training:

- **Accuracy Graph** — training vs validation accuracy per epoch
- **Loss Graph** — training vs validation loss per epoch
- **Confusion Matrix** — 6×6 heatmap of classification results

---

## 🗂️ Project Structure

```
emotion_emoji_project/
│
├── app.py                      # Flask application + prediction endpoint
├── train_model.py              # Model training script
├── emotion_lstm_model.h5       # Trained BiLSTM model weights
├── tokenizer.pkl               # Saved Keras tokenizer
├── training.csv                # Labeled emotion dataset
│
├── templates/
│   └── index.html              # Main frontend page
│
└── static/
    ├── style.css               # Frontend styles
    ├── accuracy_graph.png      # Training accuracy plot
    ├── loss_graph.png          # Training loss plot
    ├── confusion_matrix.png    # Confusion matrix heatmap
    └── sounds/
        ├── happy.mp3           # Joy sound
        ├── sad.mp3             # Sadness sound
        ├── angry.mp3           # Anger sound
        ├── love.mp3            # Love sound
        ├── fear.mp3            # Fear sound
        ├── surprise.mp3        # Surprise sound
        ├── laugh.mp3           # Emoji fusion sound
        ├── cool.mp3            # Emoji fusion sound
        ├── sleep.mp3           # Emoji fusion sound
        └── party.mp3           # Emoji fusion sound
```

---

## ⚙️ Installation & Usage

### 1. Install Dependencies

```bash
pip install flask tensorflow pandas numpy scikit-learn matplotlib seaborn joblib
```

### 2. Train the Model

```bash
python train_model.py
```

This will generate:
- `emotion_lstm_model.h5` — trained model
- `tokenizer.pkl` — fitted tokenizer
- `static/accuracy_graph.png`
- `static/loss_graph.png`
- `static/confusion_matrix.png`

### 3. Run the Application

```bash
python app.py
```

Open in your browser:

```
http://127.0.0.1:5000
```

---

## 🛠️ Technology Stack

| Layer | Technology |
|---|---|
| **Language** | Python 3.10+ |
| **Web Framework** | Flask |
| **Deep Learning** | TensorFlow + Keras |
| **Model** | Bidirectional LSTM |
| **Data Processing** | Pandas, NumPy, Scikit-learn |
| **Tokenization** | Keras Tokenizer |
| **Visualization** | Matplotlib, Seaborn |
| **Audio** | MP3 files via JavaScript Audio API |
| **Frontend** | HTML5, CSS3, JavaScript |

---

## 💻 Hardware Requirements

| Component | Minimum | Recommended |
|---|---|---|
| Processor | Intel Core i3 | Intel Core i5/i7 |
| RAM | 4 GB | 8 GB+ |
| Storage | 2 GB free | SSD |
| GPU | — | NVIDIA (for faster training) |

---

## 🔮 Future Enhancements

- [ ] BERT / RoBERTa-based emotion classification
- [ ] Multi-label emotion detection (text can express multiple emotions)
- [ ] Real-time speech-to-emotion pipeline
- [ ] Multilingual emotion detection
- [ ] Cloud deployment (Heroku / AWS / Render)
- [ ] Mobile application (Flutter / React Native)
- [ ] Emotion trend tracking over conversation history
- [ ] REST API for third-party integration

---

## 👨‍💻 Author

**Abdul Haris H**
M.Tech — Computer Science and Engineering
Government Engineering College Thrissur
APJ Abdul Kalam Technological University

---

## 📄 License

This project is developed for academic and educational purposes.

---

<div align="center">
  <sub>Built with ❤️ using TensorFlow, Keras BiLSTM, and Flask</sub>
</div>