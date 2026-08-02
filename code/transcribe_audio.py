import os
import sys
import json
import whisper

def transcribe_all():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    repo_dir = os.path.abspath(os.path.join(base_dir, ".."))
    audio_dir = os.path.join(repo_dir, "dataset", "media", "audio")
    
    print("Loading Whisper model...")
    model = whisper.load_model("tiny")
    
    results = {}
    for filename in sorted(os.listdir(audio_dir)):
        if filename.endswith(".mp3"):
            vn_id = os.path.splitext(filename)[0]
            file_path = os.path.join(audio_dir, filename)
            print(f"Transcribing {vn_id} ({filename})...")
            try:
                res = model.transcribe(file_path)
                text = res.get("text", "").strip()
                results[vn_id] = {
                    "text": text,
                    "language": res.get("language", "en")
                }
                print(f"  {vn_id}: {text}")
            except Exception as e:
                print(f"  Error on {vn_id}: {e}")
                results[vn_id] = {"text": "", "error": str(e)}
                
    out_path = os.path.join(base_dir, "audio_transcriptions.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Transcriptions saved to {out_path}")

if __name__ == "__main__":
    transcribe_all()
