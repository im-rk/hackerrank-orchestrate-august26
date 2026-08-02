import os
import json

def inspect():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(base_dir, "messages_deep_analysis.json"), "r", encoding="utf-8") as f:
        data = json.load(f)
        
    for item in data:
        mid = item["mid"]
        ctype = item["ctype"]
        text = item["text"]
        gname = item["gname"]
        is_admin = item["sender_is_admin"]
        bname = item["bname"]
        fcount = item["fcount"]
        
        print(f"[{item['idx']:03d}] {mid} | {ctype} | Admin:{is_admin} | Biz:{bname} | Grp:{gname} | Fwd:{fcount}")
        print(f"      Text: {text}")
        print()

if __name__ == "__main__":
    inspect()
