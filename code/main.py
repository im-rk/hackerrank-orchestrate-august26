import os
import sys
import pandas as pd
from router import MessageRouter

def run():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    repo_dir = os.path.abspath(os.path.join(base_dir, ".."))
    dataset_dir = os.path.join(repo_dir, "dataset")
    
    messages_path = os.path.join(dataset_dir, "messages.csv")
    output_path = os.path.join(dataset_dir, "output.csv")
    
    print(f"Loading incoming messages from {messages_path}...")
    df = pd.read_csv(messages_path)
    
    router = MessageRouter(dataset_dir)
    results = []
    
    for idx, row in df.iterrows():
        decision = router.route_message(row)
        results.append({
            "message_id": row["message_id"],
            "action": decision["action"],
            "message_type": decision["message_type"],
            "reason": decision["reason"],
            "confidence": round(float(decision["confidence"]), 2),
            "evidence_message_ids": decision["evidence_message_ids"]
        })
        
    out_df = pd.DataFrame(results)
    out_df.to_csv(output_path, index=False)
    root_output_path = os.path.join(repo_dir, "output.csv")
    out_df.to_csv(root_output_path, index=False)
    print(f"Successfully processed {len(out_df)} messages and saved predictions to {output_path} and {root_output_path}")

if __name__ == "__main__":
    run()
