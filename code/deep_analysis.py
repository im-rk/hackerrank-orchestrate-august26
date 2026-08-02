import os
import sys
import json
import pandas as pd

def analyze_all():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    repo_dir = os.path.abspath(os.path.join(base_dir, ".."))
    dataset_dir = os.path.join(repo_dir, "dataset")
    
    messages_df = pd.read_csv(os.path.join(dataset_dir, "messages.csv"))
    users_df = pd.read_csv(os.path.join(dataset_dir, "users.csv")).set_index("user_id")
    groups_df = pd.read_csv(os.path.join(dataset_dir, "groups.csv")).set_index("group_id")
    group_members_df = pd.read_csv(os.path.join(dataset_dir, "group_members.csv"))
    biz_df = pd.read_csv(os.path.join(dataset_dir, "business_accounts.csv")).set_index("business_id")
    user_biz_df = pd.read_csv(os.path.join(dataset_dir, "user_business_history.csv"))
    history_df = pd.read_csv(os.path.join(dataset_dir, "message_history.csv"))
    events_df = pd.read_csv(os.path.join(dataset_dir, "message_events.csv"))
    
    # Load cache
    cache_path = os.path.join(base_dir, "extracted_media_cache.json")
    media_cache = {}
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            media_cache = json.load(f)
            
    print(f"Loaded {len(messages_df)} incoming messages.")
    
    rows = []
    for idx, row in messages_df.iterrows():
        mid = row['message_id']
        uid = row['user_id']
        ctype = row['conversation_type']
        gid = str(row['group_id']) if pd.notna(row['group_id']) else ""
        bid = str(row['business_id']) if pd.notna(row['business_id']) else ""
        sid = str(row['sender_user_id']) if pd.notna(row['sender_user_id']) else ""
        mtype = str(row['media_type']) if pd.notna(row['media_type']) else ""
        media_id = str(row['media_id']) if pd.notna(row['media_id']) else ""
        fcount = int(row['forwarded_count']) if pd.notna(row['forwarded_count']) else 0
        raw_text = str(row['message_text']) if pd.notna(row['message_text']) else ""
        
        extracted = media_cache.get(media_id, "")
        full_text = (raw_text + " " + extracted).strip()
        
        u_info = users_df.loc[uid].to_dict() if uid in users_df.index else {}
        g_info = groups_df.loc[gid].to_dict() if gid in groups_df.index else {}
        b_info = biz_df.loc[bid].to_dict() if bid in biz_df.index else {}
        
        # group member info
        gm_match = group_members_df[(group_members_df['group_id'] == gid) & (group_members_df['user_id'] == uid)]
        gm_user = gm_match.iloc[0].to_dict() if not gm_match.empty else {}
        
        gm_sender_match = group_members_df[(group_members_df['group_id'] == gid) & (group_members_df['user_id'] == sid)]
        gm_sender = gm_sender_match.iloc[0].to_dict() if not gm_sender_match.empty else {}
        
        # user business info
        ub_match = user_biz_df[(user_biz_df['user_id'] == uid) & (user_biz_df['business_id'] == bid)]
        ub_info = ub_match.iloc[0].to_dict() if not ub_match.empty else {}
        
        # history with sender/group/biz
        user_hist = history_df[history_df['user_id'] == uid]
        
        rows.append({
            "idx": idx + 1,
            "mid": mid,
            "uid": uid,
            "ctype": ctype,
            "gid": gid,
            "gname": g_info.get("group_name", ""),
            "gtype": g_info.get("group_type", ""),
            "sender_is_admin": gm_sender.get("role", "") == "admin",
            "group_muted": gm_user.get("group_muted_by_user", 0) == 1,
            "bid": bid,
            "bname": b_info.get("brand_name", ""),
            "bcat": b_info.get("category", ""),
            "bver": b_info.get("verified", 0),
            "brep": b_info.get("user_reports_30d", 0),
            "ub_rel": ub_info.get("why_user_knows_account", ""),
            "ub_optout": pd.notna(ub_info.get("promotions_opted_out_at")),
            "ub_dismissed": ub_info.get("messages_dismissed_30d", 0),
            "sid": sid,
            "mtype": mtype,
            "media_id": media_id,
            "fcount": fcount,
            "text": full_text
        })
        
    out_df = pd.DataFrame(rows)
    out_df.to_json(os.path.join(base_dir, "messages_deep_analysis.json"), orient="records", indent=2)
    print(f"Saved deep analysis of {len(out_df)} messages to code/messages_deep_analysis.json")

if __name__ == "__main__":
    analyze_all()
