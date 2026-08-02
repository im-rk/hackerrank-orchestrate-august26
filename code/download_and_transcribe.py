import os
import sys
import json

# Add imageio_ffmpeg's ffmpeg binary to PATH
try:
    import imageio_ffmpeg
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    ffmpeg_dir = os.path.dirname(ffmpeg_exe)
    if ffmpeg_dir not in os.environ["PATH"]:
        os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ["PATH"]
    print(f"Added ffmpeg from {ffmpeg_exe} to PATH")
except Exception as e:
    print(f"imageio_ffmpeg note: {e}")

try:
    import static_ffmpeg
    static_ffmpeg.add_paths()
    print("static_ffmpeg paths added")
except Exception as e:
    pass

import whisper

def transcribe():
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
                
    # Update extracted_media_cache.json with both images and audio
    cache_path = os.path.join(base_dir, "extracted_media_cache.json")
    cache_data = {"images": {}, "audio": {}}
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cache_data = json.load(f)
        except Exception:
            pass
            
    cache_data["audio"] = results
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cache_data, f, indent=2, ensure_ascii=False)
        
    print(f"\nAll voice notes transcribed! Updated cache at {cache_path}")

if __name__ == "__main__":
    transcribe()
