import os
import sys

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

import json
import easyocr
import whisper
import pandas as pd

def extract_all():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    repo_dir = os.path.abspath(os.path.join(base_dir, ".."))
    dataset_dir = os.path.join(repo_dir, "dataset")
    
    extracted_data = {
        "images": {},
        "audio": {}
    }
    
    # 1. OCR for images
    print("Initializing EasyOCR with verbose=False...")
    reader = easyocr.Reader(['en'], gpu=False, verbose=False)
    
    images_csv = os.path.join(dataset_dir, "images.csv")
    if os.path.exists(images_csv):
        img_df = pd.read_csv(images_csv)
        for _, row in img_df.iterrows():
            img_id = row['image_id']
            rel_path = row['file_path']
            abs_path = os.path.join(dataset_dir, rel_path)
            if os.path.exists(abs_path):
                print(f"Running OCR on {img_id}: {rel_path}...")
                try:
                    results = reader.readtext(abs_path)
                    text_lines = [res[1] for res in results]
                    extracted_data["images"][img_id] = {
                        "text": " ".join(text_lines),
                        "lines": text_lines
                    }
                    print(f"  {img_id} extracted: {' '.join(text_lines)[:100]}...")
                except Exception as e:
                    print(f"  Warning: OCR failed on {img_id}: {e}")
                    extracted_data["images"][img_id] = {
                        "text": "",
                        "lines": [],
                        "error": str(e)
                    }
            else:
                print(f"  Warning: {abs_path} not found")
                
    # 2. Transcription for audio
    print("\nInitializing Whisper...")
    model = whisper.load_model("tiny")
    
    vn_csv = os.path.join(dataset_dir, "voice_notes.csv")
    if os.path.exists(vn_csv):
        vn_df = pd.read_csv(vn_csv)
        for _, row in vn_df.iterrows():
            vn_id = row['voice_note_id']
            rel_path = row['file_path']
            abs_path = os.path.join(dataset_dir, rel_path)
            if os.path.exists(abs_path):
                print(f"Transcribing {vn_id}: {rel_path}...")
                try:
                    result = model.transcribe(abs_path)
                    extracted_data["audio"][vn_id] = {
                        "text": result.get("text", "").strip(),
                        "language": result.get("language", "en")
                    }
                    print(f"  {vn_id} transcribed: {result.get('text', '').strip()}...")
                except Exception as e:
                    print(f"  Error transcribing {vn_id}: {e}")
                    extracted_data["audio"][vn_id] = {
                        "text": "",
                        "error": str(e)
                    }
            else:
                print(f"  Warning: {abs_path} not found")
                
    cache_path = os.path.join(base_dir, "extracted_media_cache.json")
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(extracted_data, f, indent=2, ensure_ascii=False)
    print(f"\nExtraction complete! Saved to {cache_path}")

if __name__ == "__main__":
    extract_all()
