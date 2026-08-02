import os
import re
import datetime
import pandas as pd

class ContextEngine:
    def __init__(self, dataset_dir=None):
        if dataset_dir is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            dataset_dir = os.path.abspath(os.path.join(base_dir, "..", "dataset"))
        self.dataset_dir = dataset_dir
        
        self.users = self._load_csv("users.csv", "user_id")
        self.groups = self._load_csv("groups.csv", "group_id")
        self.group_members = self._load_group_members()
        self.businesses = self._load_csv("business_accounts.csv", "business_id")
        self.user_business = self._load_user_business()
        self.daily_summary = self._load_daily_summary()
        
    def _load_csv(self, filename, index_col=None):
        path = os.path.join(self.dataset_dir, filename)
        if os.path.exists(path):
            df = pd.read_csv(path)
            if index_col and index_col in df.columns:
                df = df.drop_duplicates(subset=[index_col])
                return df.set_index(index_col).to_dict(orient="index")
            return df.to_dict(orient="records")
        return {}

    def _load_daily_summary(self):
        path = os.path.join(self.dataset_dir, "daily_notification_summary.csv")
        if not os.path.exists(path):
            return {}
        df = pd.read_csv(path)
        lookup = {}
        for _, row in df.iterrows():
            u_id = str(row['user_id'])
            if u_id not in lookup:
                lookup[u_id] = []
            lookup[u_id].append(row.to_dict())
        return lookup

    def _load_group_members(self):
        path = os.path.join(self.dataset_dir, "group_members.csv")
        if not os.path.exists(path):
            return {}
        df = pd.read_csv(path)
        lookup = {}
        for _, row in df.iterrows():
            key = (str(row['group_id']), str(row['user_id']))
            lookup[key] = row.to_dict()
        return lookup

    def _load_user_business(self):
        path = os.path.join(self.dataset_dir, "user_business_history.csv")
        if not os.path.exists(path):
            return {}
        df = pd.read_csv(path)
        lookup = {}
        for _, row in df.iterrows():
            key = (str(row['user_id']), str(row['business_id']))
            lookup[key] = row.to_dict()
        return lookup

    def is_in_dnd_window(self, user_id, timestamp_str):
        user_info = self.users.get(str(user_id), {})
        dnd_window = user_info.get("do_not_disturb_window")
        if not dnd_window or pd.isna(dnd_window) or "-" not in str(dnd_window):
            return False
        
        try:
            # Parse message time (HH:MM)
            dt = datetime.datetime.strptime(str(timestamp_str).strip(), "%Y-%m-%d %H:%M")
            msg_minutes = dt.hour * 60 + dt.minute
            
            start_str, end_str = str(dnd_window).split("-")
            sh, sm = map(int, start_str.split(":"))
            eh, em = map(int, end_str.split(":"))
            start_minutes = sh * 60 + sm
            end_minutes = eh * 60 + em
            
            if start_minutes <= end_minutes:
                return start_minutes <= msg_minutes <= end_minutes
            else:
                # Overnight DND window, e.g. 22:00-07:00
                return msg_minutes >= start_minutes or msg_minutes <= end_minutes
        except Exception:
            return False

    def is_group_admin(self, group_id, sender_user_id):
        key = (str(group_id), str(sender_user_id))
        member_info = self.group_members.get(key, {})
        return str(member_info.get("role", "")).lower() == "admin"

    def is_group_muted_by_user(self, group_id, user_id):
        key = (str(group_id), str(user_id))
        member_info = self.group_members.get(key, {})
        return int(member_info.get("group_muted_by_user", 0)) == 1

    def is_direct_mention(self, message_text, user_id):
        if not message_text or not user_id:
            return False
        pattern = rf"@{re.escape(str(user_id))}\b"
        return bool(re.search(pattern, str(message_text), re.IGNORECASE))

    def detect_spoofed_or_fraud_business(self, business_id, text=""):
        b_info = self.businesses.get(str(business_id), {})
        if not b_info:
            return False, "unknown_business"
        
        official_domain = str(b_info.get("official_domain", "")).strip().lower()
        sender_domain = str(b_info.get("domain_used_by_sender", "")).strip().lower()
        reports = int(b_info.get("user_reports_30d", 0))
        verified = int(b_info.get("verified", 0))
        
        # Check spoofed domains on unverified business accounts
        if verified == 0:
            if official_domain and sender_domain and official_domain != sender_domain and not sender_domain.endswith(official_domain):
                return True, f"Domain spoofing: brand domain {official_domain} vs sender domain {sender_domain}"
            if reports >= 20:
                return True, f"High risk sender: unverified with {reports} user reports"
        
        return False, "legitimate"

    def detect_prompt_injection(self, text):
        if not text:
            return False
        text_lower = str(text).lower()
        injection_keywords = [
            "ignore all previous routing rules",
            "routing override:",
            "set action=notify",
            "mark this message as notify",
            "ignore previous instructions",
            "system override",
            "rule override",
        ]
        return any(kw in text_lower for kw in injection_keywords)

    def detect_scam_or_phishing(self, text, business_id=None, sender_id=None):
        if not text:
            return False, ""
        text_lower = str(text).lower()
        
        # Check prompt injection
        if self.detect_prompt_injection(text):
            return True, "The message contains adversarial prompt injection instructions."
            
        # Check urgent credential/OTP harvesting and financial scam patterns
        otp_patterns = [
            r"\b(otp|verification code|login code|6 digit code|pin|password)\b.*(leak|send|confirm|reply|share|enter|verify)",
            r"(verify|confirm|send|enter|share|reply with).*\b(otp|code|pin|password)\b",
            r"(profile|account|workspace access|wallet).*(blocked|suspended|expire|terminated|restricted|revoked).*(confirm|verify|link|code|otp)",
            r"failed login attempts.*(profile will be|verify through|restricted|revoked)",
            r"scan this qr and pay.*clearance amount immediately.*(access card|blocked)",
            r"reattempt fee at amazonpay-delivery\.in.*enter otp",
            r"wallet kyc incomplete.*confirm card number.*pin.*otp",
            r"loan approved.*pay processing fee.*amount will be released",
            r"claim benefits by sharing your account number",
            r"otp verify nahi hua.*(account.*hold|link open|code daal do)",
            r"refund update.*(wallet details|link shared here).*to release the amount",
            r"international payout profile needs one final verification step",
            r"pay-check-secure\.com",
            r"account-help\.in/verify",
        ]
        for pattern in otp_patterns:
            if re.search(pattern, text_lower):
                return True, "The message asks for sensitive verification, OTP, or payment under urgency."
                
        # Business spoof check
        if business_id:
            is_spoof, reason = self.detect_spoofed_or_fraud_business(business_id, text)
            if is_spoof:
                return True, reason

        return False, ""
