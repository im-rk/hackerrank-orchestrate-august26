import os
import pandas as pd

def analyze_history():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    repo_dir = os.path.abspath(os.path.join(base_dir, ".."))
    dataset_dir = os.path.join(repo_dir, "dataset")
    
    msgs = pd.read_csv(os.path.join(dataset_dir, "messages.csv"))
    samples = pd.read_csv(os.path.join(dataset_dir, "sample_messages.csv"))
    history = pd.read_csv(os.path.join(dataset_dir, "message_history.csv"))
    events = pd.read_csv(os.path.join(dataset_dir, "message_events.csv"))
    
    print(f"Messages count: {len(msgs)}")
    print(f"Samples count: {len(samples)}")
    print(f"History count: {len(history)}")
    print(f"Events count: {len(events)}")
    
    # Merge history with events
    hist_events = pd.merge(history, events, on=['user_id', 'message_id'], how='left')
    print("Merged history & events shape:", hist_events.shape)
    print("Events columns:", events.columns.tolist())
    print("\nSample evidence mapping inspection:")
    for idx, row in samples.head(10).iterrows():
        ev_ids = str(row['evidence_message_ids']).split(';')
        print(f"\nSample: {row['message_id']} | User: {row['user_id']} | Action: {row['action']} | Ev: {row['evidence_message_ids']}")
        print(f"  Text: {str(row['message_text'])[:60]}...")
        for evid in ev_ids:
            if evid != 'none':
                h_match = hist_events[hist_events['message_id'] == evid]
                if not h_match.empty:
                    h_row = h_match.iloc[0]
                    print(f"  -> Match {evid}: opened={h_row['message_opened']}, replied={h_row['message_replied']}, dismissed={h_row['notification_dismissed']}, muted={h_row['muted_after_message']}")
                    print(f"     History text: {str(h_row['message_text'])[:60]}...")
                else:
                    print(f"  -> Match {evid} NOT FOUND in history!")

if __name__ == "__main__":
    analyze_history()
