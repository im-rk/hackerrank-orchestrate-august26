import os
import sys
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from router import MessageRouter
from features import ContextEngine
from evidence import EvidenceRetriever

def audit():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    repo_dir = os.path.abspath(os.path.join(base_dir, ".."))
    dataset_dir = os.path.join(repo_dir, "dataset")
    
    messages_path = os.path.join(dataset_dir, "messages.csv")
    df = pd.read_csv(messages_path)
    
    router = MessageRouter(dataset_dir)
    context = router.context
    
    print(f"Total messages to audit: {len(df)}")
    
    for idx, row in df.iterrows():
        mid = row['message_id']
        uid = row['user_id']
        ctype = row['conversation_type']
        gid = row['group_id'] if pd.notna(row['group_id']) else ""
        bid = row['business_id'] if pd.notna(row['business_id']) else ""
        sid = row['sender_user_id'] if pd.notna(row['sender_user_id']) else ""
        mtype = row['media_type'] if pd.notna(row['media_type']) else ""
        media_id = row['media_id'] if pd.notna(row['media_id']) else ""
        fcount = row['forwarded_count'] if pd.notna(row['forwarded_count']) else 0
        text = str(row['message_text']) if pd.notna(row['message_text']) else ""
        
        media_text = router.get_media_text(mtype, media_id)
        decision = router.route_message(row)
        
        print(f"--- [{idx+1:03d}/110] {mid} ({uid} | {ctype}) ---")
        if gid: print(f"    Group: {gid} | Admin: {context.is_group_admin(gid, sid)} | Muted: {context.is_group_muted_by_user(gid, uid)}")
        if bid: print(f"    Business: {bid} | Rel: {context.user_business.get((str(uid), str(bid)), {})}")
        if sid: print(f"    Sender: {sid}")
        if mtype: print(f"    Media: {mtype} ({media_id}) -> '{media_text[:80]}...'")
        if text: print(f"    Text: '{text[:80]}...'")
        print(f"    => PREDICTED: Action={decision['action']} | Type={decision['message_type']} | Conf={decision['confidence']} | Ev={decision['evidence_message_ids']}")
        print(f"       Reason: {decision['reason']}")

if __name__ == "__main__":
    audit()
