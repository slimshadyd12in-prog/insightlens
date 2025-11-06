import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

def analyze_sentiment_file(transcript_path):
    base_path = os.path.dirname(transcript_path)
    sentiment_json_path = os.path.splitext(transcript_path)[0] + "_sentiment.json"

    # 🔹 Check transcript
    if not os.path.exists(transcript_path):
        raise FileNotFoundError(f"Transcript file not found: {transcript_path}")
    print(f"✅ Found transcript: {transcript_path}")

    with open(transcript_path, "r", encoding="utf-8") as f:
        transcript = f.read().strip()
    if not transcript:
        raise ValueError("Transcript file is empty. Please check your ASR output.")

    # 🔹 Detect device
    device = 0 if torch.cuda.is_available() else -1
    device_name = "GPU" if device == 0 else "CPU"
    print(f"🔥 Using {device_name} for sentiment analysis")

    # 🔹 Load model
    model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    print(f"✅ Loading sentiment model ({model_name})...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    sentiment_analyzer = pipeline(
        "sentiment-analysis",
        model=model,
        tokenizer=tokenizer,
        device=device
    )

    # 🔹 Split transcript into sentences
    sentences = [s.strip() for s in transcript.replace("\n", " ").split(".") if s.strip()]
    print(f"🔍 Split transcript into {len(sentences)} sentences for analysis")

    sentiment_results = []
    for sentence in sentences:
        result = sentiment_analyzer(sentence)[0]
        sentiment_results.append({
            "sentence": sentence,
            "label": result["label"],
            "score": round(result["score"], 4)
        })

    # 🔹 Aggregate
    labels = [r["label"] for r in sentiment_results]
    overall_sentiment = max(set(labels), key=labels.count)

    output = {
        "overall_sentiment": overall_sentiment,
        "sentence_sentiments": sentiment_results
    }

    with open(sentiment_json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=4, ensure_ascii=False)

    print(f"✅ Sentiment JSON saved to: {sentiment_json_path}")
    return output


# Keep this only for manual run support
if __name__ == "__main__":
    base_path = os.path.join(os.path.dirname(__file__), "data")
    transcript_files = [f for f in os.listdir(base_path) if f.endswith("_transcript.txt")]
    if transcript_files:
        latest_transcript = os.path.join(base_path, sorted(transcript_files, key=lambda f: os.path.getmtime(os.path.join(base_path, f)), reverse=True)[0])
        analyze_sentiment_file(latest_transcript)
    else:
        print("❌ No transcript found.")
