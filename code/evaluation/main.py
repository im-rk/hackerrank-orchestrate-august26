import os
import sys

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

import pandas as pd

# Add parent directory to sys.path to import router
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from router import MessageRouter

def evaluate():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    repo_dir = os.path.abspath(os.path.join(base_dir, "..", ".."))
    dataset_dir = os.path.join(repo_dir, "dataset")
    
    samples_path = os.path.join(dataset_dir, "sample_messages.csv")
    if not os.path.exists(samples_path):
        print(f"Error: {samples_path} not found.")
        return
        
    df_samples = pd.read_csv(samples_path)
    router = MessageRouter(dataset_dir)
    
    total = len(df_samples)
    action_matches = 0
    type_matches = 0
    evidence_matches = 0
    
    print("=" * 70)
    print(f"RUNNING MESSAGE NOTIFICATION ROUTER EVALUATION ({total} samples)")
    print("=" * 70)
    
    for idx, row in df_samples.iterrows():
        msg_id = row['message_id']
        pred = router.route_message(row)
        
        gt_action = str(row['action']).strip()
        pred_action = str(pred['action']).strip()
        is_action_match = gt_action == pred_action
        if is_action_match:
            action_matches += 1
            
        gt_type = str(row['message_type']).strip()
        pred_type = str(pred['message_type']).strip()
        is_type_match = gt_type == pred_type
        if is_type_match:
            type_matches += 1
            
        gt_ev = set(str(row['evidence_message_ids']).strip().split(';'))
        pred_ev = set(str(pred['evidence_message_ids']).strip().split(';'))
        is_ev_match = gt_ev == pred_ev or bool(gt_ev.intersection(pred_ev))
        if is_ev_match:
            evidence_matches += 1
            
        status = "[PASS]" if (is_action_match and is_type_match) else "[MISMATCH]"
        print(f"{status} {msg_id}")
        print(f"   Action: Pred={pred_action} | GroundTruth={gt_action}")
        print(f"   Type:   Pred={pred_type} | GroundTruth={gt_type}")
        print(f"   Ev:     Pred={pred['evidence_message_ids']} | GroundTruth={row['evidence_message_ids']}")
        print(f"   Reason: {pred['reason']}")
        print(f"   Conf:   {pred['confidence']:.2f}")
        print("-" * 70)
        
    action_acc = (action_matches / total) * 100
    type_acc = (type_matches / total) * 100
    ev_acc = (evidence_matches / total) * 100
    
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    print(f"Total Samples Evaluated:  {total}")
    print(f"Action Accuracy:          {action_matches}/{total} ({action_acc:.1f}%)")
    print(f"Message Type Accuracy:    {type_matches}/{total} ({type_acc:.1f}%)")
    print(f"Evidence Retrieval Match: {evidence_matches}/{total} ({ev_acc:.1f}%)")
    print("=" * 70)

if __name__ == "__main__":
    evaluate()
