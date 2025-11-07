import os
import subprocess
from datetime import datetime
from pydub import AudioSegment
import numpy as np
import whisper
import torch

# Optional DirectML for AMD/Intel
try:
    import torch_directml
    has_dml = True
except ImportError:
    has_dml = False


def get_available_device():
    """Detects and returns the best available device (CUDA / DirectML / CPU)."""
    if torch.cuda.is_available():
        print("⚡ Using NVIDIA GPU (CUDA)")
        return "cuda"
    elif has_dml:
        print("🔥 Using AMD/Intel GPU via DirectML")
        return "cpu"  # Whisper does not support torch_directml directly, fallback CPU
    else:
        print("💻 Using CPU")
        return "cpu"


def main():
    # Automatically detect the latest uploaded audio file in /data/
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"❌ Data folder not found: {data_dir}")

    supported_ext = (".m4a", ".wav", ".mp3")
    audio_files = [f for f in os.listdir(data_dir) if f.lower().endswith(supported_ext)]

    if not audio_files:
        raise FileNotFoundError("❌ No audio file found in the data/ folder. Please upload one via the web interface.")

    # Pick most recent audio file
    audio_files = sorted(audio_files, key=lambda f: os.path.getmtime(os.path.join(data_dir, f)), reverse=True)
    audio_path = os.path.join(data_dir, audio_files[0])
    print(f"✅ Found audio file: {audio_path}")

    # Create unique output names based on timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    wav_path = os.path.splitext(audio_path)[0] + f"_{timestamp}.wav"
    txt_path = os.path.splitext(audio_path)[0] + f"_{timestamp}_transcript.txt"

    ffmpeg_path = r"C:\ffmpeg\bin\ffmpeg.exe"

    # Convert to WAV
    print("🔄 Converting audio → .wav using ffmpeg...")
    subprocess.run(
        [ffmpeg_path, "-y", "-i", audio_path, wav_path],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True
    )
    print(f"✅ WAV file ready at: {wav_path}")

    # Preprocess
    audio = AudioSegment.from_file(wav_path)
    audio = audio.set_channels(1).set_frame_rate(16000)
    audio_array = np.array(audio.get_array_of_samples()).astype(np.float32) / 32768.0
    print("🎵 Audio loaded and converted to mono 16kHz.")

    # Load Whisper model
    device = get_available_device()
    print("✅ Loading Whisper model...")
    model = whisper.load_model("base", device=device)

    # Transcribe
    print("🎧 Transcribing audio...")
    result = model.transcribe(audio_array, fp16=False)
    text = result.get("text", "").strip()

    print("\n===== TRANSCRIPTION RESULT =====\n")
    print(text if text else "[No text detected]")
    print("\n=================================\n")

    # Save transcript
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"✅ Transcription saved to: {txt_path}")

    return txt_path  # ✅ Return path for next scripts


if __name__ == "__main__":
    main()
