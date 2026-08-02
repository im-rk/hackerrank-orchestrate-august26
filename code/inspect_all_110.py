import os
import json

def inspect_all():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(base_dir, "messages_deep_analysis.json"), "r", encoding="utf-8") as f:
        data = json.load(f)
        
    for item in data:
        idx = item["idx"]
        mid = item["mid"]
        uid = item["uid"]
        ctype = item["ctype"]
        text = item["text"]
        gname = item["gname"]
        gtype = item["gtype"]
        is_admin = item["sender_is_admin"]
        muted = item["group_muted"]
        bid = item["bid"]
        bname = item["bname"]
        bcat = item["bcat"]
        bver = item["bver"]
        brep = item["brep"]
        ub_rel = item["ub_rel"]
        ub_optout = item["ub_optout"]
        ub_dism = item["ub_dismissed"]
        mtype = item["mtype"]
        fcount = item["fcount"]
        
        print(f"[{idx:03d}] {mid} | User:{uid} | CType:{ctype} | GName:{gname} (Admin:{is_admin}, Muted:{muted}) | Biz:{bname} (Ver:{bver}, Rep:{brep}, Rel:{ub_rel}, OptOut:{ub_optout}, Dism:{ub_dism}) | Media:{mtype} | Fwd:{fcount}")
        print(f"      Text: {text}")
        print("-" * 70)

if __name__ == "__main__":
    inspect_all()
