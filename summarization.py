# backend/summarization.py
import os
from transformers import pipeline

def get_latest_transcript_file(data_dir):
    transcript_files = [
        f for f in os.listdir(data_dir)
        if f.lower().endswith("_transcript.txt")
    ]
    if not transcript_files:
        return None
    # newest first
    transcript_files.sort(key=lambda f: os.path.getmtime(os.path.join(data_dir, f)), reverse=True)
    return os.path.join(data_dir, transcript_files[0])

def summarize_text(input_file, output_file):
    if not os.path.exists(input_file):
        print(f"❌ Transcript file not found: {input_file}")
        return False

    with open(input_file, "r", encoding="utf-8") as f:
        transcript = f.read().strip()
    if not transcript:
        print("⚠️ Transcript is empty — nothing to summarize.")
        return False

    print("🧠 Loading summarization model (T5-small)...")
    summarizer = pipeline("summarization", model="t5-small", tokenizer="t5-small")

    print("📝 Summarizing transcript...")
    summary = summarizer(transcript, max_length=130, min_length=30, do_sample=False)[0]['summary_text']

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(summary.strip())

    print(f"\n✅ Summary saved to: {output_file}")
    return True

def main():
    base_path = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(base_path, exist_ok=True)

    latest = get_latest_transcript_file(base_path)
    if not latest:
        print("❌ No transcript file found in data/ to summarize.")
        return None

    summary_file = latest.replace("_transcript.txt", "_summary.txt")
    ok = summarize_text(latest, summary_file)
    return summary_file if ok else None

if __name__ == "__main__":
    print("Running summarization.main()")
    print("Summary file:", main())
