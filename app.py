import os
import subprocess
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import whisper
import numpy as np
from pydub import AudioSegment

# Import your existing scripts
#import transcription
import summarization  # import full module for dynamic file detection
import sentiment_analysis

# ====================================================
# Define Base Directory (consistent with other scripts)
# ====================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

app = Flask(__name__)

# ====================================================
# ✅ Load Whisper model globally (only once)
# ====================================================
print("🚀 Loading Whisper model (tiny)...")
model = whisper.load_model("tiny", device="cpu")
print("✅ Whisper model loaded successfully!")

# ====================================================
# Utility Function
# ====================================================
def get_latest_transcript_file():
    """Return path of latest transcript file (_transcript.txt) from data/ folder."""
    if not os.path.exists(DATA_DIR):
        return None
    transcript_files = [
        f for f in os.listdir(DATA_DIR)
        if f.lower().endswith("_transcript.txt")
    ]
    if not transcript_files:
        return None
    transcript_files.sort(key=lambda f: os.path.getmtime(os.path.join(DATA_DIR, f)), reverse=True)
    latest_file = os.path.join(DATA_DIR, transcript_files[0])
    return latest_file

# ====================================================
# Routes
# ====================================================

@app.route('/')
def index():
    return render_template('index.html')

# -----------------------------
# Upload audio file
# -----------------------------
@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['file']
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(DATA_DIR, filename)
        file.save(filepath)
        return jsonify({"message": f"✅ File uploaded successfully: {filename}"})
    return jsonify({"error": "❌ No file uploaded"}), 400

# -----------------------------
# Transcription route
# -----------------------------
@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    try:
        # Find the most recent uploaded audio file
        audio_files = [
            f for f in os.listdir(DATA_DIR)
            if f.lower().endswith((".m4a", ".wav", ".mp3"))
        ]
        if not audio_files:
            return jsonify({"status": "error", "message": "❌ No uploaded audio file found."}), 400

        audio_files.sort(key=lambda f: os.path.getmtime(os.path.join(DATA_DIR, f)), reverse=True)
        latest_audio = os.path.join(DATA_DIR, audio_files[0])

        # Convert to mono 16kHz for Whisper
        audio = AudioSegment.from_file(latest_audio)
        audio = audio.set_channels(1).set_frame_rate(16000)
        audio_array = np.array(audio.get_array_of_samples()).astype(np.float32) / 32768.0

        # 🔥 Transcribe using preloaded model
        result = model.transcribe(audio_array, fp16=False)
        text = result.get("text", "").strip()

        # Save transcript file for downstream tasks
        transcript_path = latest_audio.rsplit(".", 1)[0] + "_transcript.txt"
        with open(transcript_path, "w", encoding="utf-8") as f:
            f.write(text)

        return jsonify({
            "status": "success",
            "message": "✅ Transcription complete!",
            "transcription": text,
            "transcript_file": transcript_path
        })

    except Exception as e:
        return jsonify({"status": "error", "message": f"❌ Transcription failed: {str(e)}"})

# -----------------------------
# Summarization route (Dynamic)
# -----------------------------
@app.route('/summarize', methods=['POST'])
def summarize():
    try:
        latest_transcript = get_latest_transcript_file()
        if not latest_transcript:
            return jsonify({"status": "error", "message": "❌ No transcript file found for summarization."}), 400

        summary_path = latest_transcript.replace("_transcript.txt", "_summary.txt")

        # Call summarization logic
        summarization.summarize_text(latest_transcript, summary_path)

        if not os.path.exists(summary_path):
            return jsonify({"status": "error", "message": "❌ Summarization failed or no output generated."}), 500

        with open(summary_path, "r", encoding="utf-8") as f:
            summary_content = f.read()

        return jsonify({
            "status": "success",
            "message": "✅ Summarization complete!",
            "summary_file": summary_path,
            "summary_text": summary_content
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# -----------------------------
# Sentiment Analysis route
# -----------------------------
@app.route('/analyze_sentiment', methods=['POST'])
def analyze_sentiment():
    try:
        base_path = os.path.join(BASE_DIR, "data")

        # Find the most recent transcript dynamically
        transcript_files = [
            f for f in os.listdir(base_path)
            if f.lower().endswith("_transcript.txt")
        ]

        if not transcript_files:
            return jsonify({"status": "error", "message": "❌ No transcript file found. Please transcribe first."}), 400

        # Get latest transcript file
        transcript_files.sort(key=lambda f: os.path.getmtime(os.path.join(base_path, f)), reverse=True)
        latest_transcript = os.path.join(base_path, transcript_files[0])

        print(f"✅ Using transcript for sentiment: {latest_transcript}")

        # Call sentiment analysis with dynamic filename
        from sentiment_analysis import analyze_sentiment_file
        sentiment_result = analyze_sentiment_file(latest_transcript)

        return jsonify({
            "status": "success",
            "message": "✅ Sentiment analysis complete!",
            "sentiment": sentiment_result
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

# -----------------------------
# Debug route - to view available files
# -----------------------------
@app.route('/_debug/list_files', methods=['GET'])
def list_files():
    files = []
    if os.path.exists(DATA_DIR):
        files = sorted(os.listdir(DATA_DIR), key=lambda f: os.path.getmtime(os.path.join(DATA_DIR, f)), reverse=True)
    return jsonify({"data_dir": DATA_DIR, "files": files})

# ====================================================
# Run Flask App
# ====================================================
if __name__ == "__main__":
    app.run(debug=True)
