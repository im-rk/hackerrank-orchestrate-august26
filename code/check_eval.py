import os
import sys
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from router import MessageRouter

def check():
    dataset_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dataset"))
    samples_path = os.path.join(dataset_dir, "sample_messages.csv")
    df = pd.read_csv(samples_path)
    
    router = MessageRouter(dataset_dir)
    
    for idx, row in df.iterrows():
        mid = row['message_id']
        pred = router.route_message(row)
        gt_a = row['action']
        gt_t = row['message_type']
        gt_e = row['evidence_message_ids']
        
        pa = pred['action']
        pt = pred['message_type']
        pe = pred['evidence_message_ids']
        
        ok = (pa == gt_a) and (pt == gt_t)
        mark = "OK" if ok else "FAIL"
        print(f"[{mark}] {mid} | A: pred={pa} gt={gt_a} | T: pred={pt} gt={gt_t} | E: pred={pe} gt={gt_e}")

if __name__ == "__main__":
    check()
