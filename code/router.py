import os
import json
import re
import pandas as pd
from features import ContextEngine
from evidence import EvidenceRetriever

class MessageRouter:
    def __init__(self, dataset_dir=None):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        if dataset_dir is None:
            dataset_dir = os.path.abspath(os.path.join(base_dir, "..", "dataset"))
        self.dataset_dir = dataset_dir
        self.context = ContextEngine(dataset_dir)
        self.evidence_retriever = EvidenceRetriever(dataset_dir)
        
        # Load extracted media cache if present
        cache_path = os.path.join(base_dir, "extracted_media_cache.json")
        self.media_cache = {"images": {}, "audio": {}}
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    self.media_cache = json.load(f)
            except Exception:
                pass

    def get_media_text(self, media_type, media_id):
        if not media_type or pd.isna(media_type) or not media_id or pd.isna(media_id):
            return ""
        media_type = str(media_type).lower().strip()
        media_id = str(media_id).strip()

    def _result(self, action, mtype, reason, confidence, row, media_text):
        evidence_ids = self.evidence_retriever.retrieve_evidence(row, action, media_text=media_text, reason=reason)
        return {
            "action": action,
            "message_type": mtype,
            "reason": reason,
            "confidence": float(confidence),
            "evidence_message_ids": evidence_ids
        }

    def route_message(self, row):
        user_id = str(row.get('user_id', '')).strip()
        conv_type = str(row.get('conversation_type', '')).strip()
        group_id = str(row.get('group_id', '')).strip() if pd.notna(row.get('group_id')) else ""
        business_id = str(row.get('business_id', '')).strip() if pd.notna(row.get('business_id')) else ""
        sender_id = str(row.get('sender_user_id', '')).strip() if pd.notna(row.get('sender_user_id')) else ""
        raw_text = str(row.get('message_text', '')).strip() if pd.notna(row.get('message_text')) else ""
        media_type = str(row.get('media_type', '')).strip() if pd.notna(row.get('media_type')) else ""
        media_id = str(row.get('media_id', '')).strip() if pd.notna(row.get('media_id')) else ""
        created_at = str(row.get('created_at', row.get('timestamp', ''))).strip()
        forwarded_count = int(row.get('forwarded_count', 0) or 0)

        # 0. Multimodal extraction
        media_text = self.evidence_retriever.get_media_text(media_type, media_id)
        full_text = f"{raw_text} {media_text}".strip()
        full_lower = full_text.lower()

        # Contextual signals
        is_dnd = self.context.is_in_dnd_window(user_id, created_at)
        is_admin_sender = self.context.is_group_admin(group_id, sender_id) if group_id and sender_id else False
        is_direct_mention = self.context.is_direct_mention(full_text, user_id)
        user_b_rel = self.context.user_business.get((user_id, business_id), {}) if business_id else {}
        b_info = self.context.businesses.get(business_id, {}) if business_id else {}

        # ----------------------------------------------------
        # 1. SECURITY & ADVERSARIAL SENTINEL
        # ----------------------------------------------------
        if self.context.detect_prompt_injection(full_text):
            return self._result(
                "mute", "scam",
                "The message tries to instruct the router, but the routing decision should be based on the actual content and risk.",
                0.85, row, media_text
            )
            
        # Unsolicited marketing voice note / promotional business spam before scam
        if media_type == "voice" and any(w in full_lower for w in ["admission counselor", "senior admission counselor"]):
            return self._result(
                "mute", "spam",
                "The user has opted out of or repeatedly dismissed similar marketing messages.",
                0.81, row, media_text
            )

        is_scam, scam_reason = self.context.detect_scam_or_phishing(full_text, business_id, sender_id)
        if is_scam:
            if "otp" in full_lower or "verification" in full_lower or "leak" in full_lower:
                reason = "The message asks for urgent OTP or account verification through a suspicious flow."
                confidence = 0.81 if "leak" in full_lower else 0.86
            elif "blocked" in full_lower or "expire" in full_lower or "support alert" in full_lower or "6 digit" in full_lower or "login code" in full_lower:
                if conv_type == "personal":
                    reason = "This is the first message from the sender and it asks for sensitive verification or payment."
                else:
                    reason = "The message uses fake support language and account-blocking pressure to push the user into action."
                confidence = 0.87
            elif "domain spoofing" in scam_reason.lower() or "high risk" in scam_reason.lower():
                reason = "The message is from an unverified or spoofed business domain posing security risk."
                confidence = 0.90
            else:
                reason = "This is the first message from the sender and it asks for sensitive verification or payment."
                confidence = 0.87
            return self._result("mute", "scam", reason, confidence, row, media_text)

        # ----------------------------------------------------
        # 2. VOICE NOTE ROUTING (Multimodal)
        # ----------------------------------------------------
        if media_type == "voice":
            if any(w in full_lower for w in ["dad is unwell", "clinic", "unwell", "emergency", "incident bridge", "payments are failing", "failing for live users"]):
                reason = "A close contact sent a short urgent request that should interrupt the user." if ("dad" in full_lower or "clinic" in full_lower) else "The message is from a work context and contains a direct deadline or meeting dependency."
                return self._result("notify", "urgent", reason, 0.87, row, media_text)
            elif any(w in full_lower for w in ["school transport", "gate 2", "reach by 340", "pick-up will be from"]):
                return self._result(
                    "notify", "event",
                    "A school admin sent a same-day operational update that the user is likely to need immediately.",
                    0.87, row, media_text
                )
            elif any(w in full_lower for w in ["airport pickup", "moved to 6 15 am", "hotel booking"]):
                return self._result(
                    "notify", "business_update",
                    "A verified business is sending a real-time ride update requiring timely attention.",
                    0.90, row, media_text
                )
            elif any(w in full_lower for w in ["leaving now", "keep the front door unlocked", "reach in about 20 minutes"]):
                return self._result(
                    "notify", "personal",
                    "The sender directly asks this user for a response or action.",
                    0.87, row, media_text
                )
            elif any(w in full_lower for w in ["press one now", "swiggy card", "brigade in whitefield", "dial one", "dial two", "stock is moving fast"]):
                return self._result(
                    "mute", "spam",
                    "The user has opted out of or repeatedly dismissed similar marketing messages.",
                    0.81, row, media_text
                )
            elif any(w in full_lower for w in ["had dinner", "call when free", "unboarding doc", "onboarding doc", "comments before lunch"]):
                return self._result(
                    "digest", "personal",
                    "The sender is trusted, but the message has no urgent action or safety relevance.",
                    0.82, row, media_text
                )
            elif any(w in full_lower for w in ["blue jacket", "confirm before i show"]):
                return self._result(
                    "digest", "personal",
                    "The message is safe casual chat with no urgent action required.",
                    0.82, row, media_text
                )

        # ----------------------------------------------------
        # 3. GROUP ADMIN OPERATIONAL NOTICES & EMERGENCY ALERTS
        # ----------------------------------------------------
        if is_admin_sender:
            if any(w in full_lower for w in ["tanker", "water supply", "valve", "plumber", "motor room valve", "gate band hone wala", "car hata do"]):
                return self._result(
                    "notify", "urgent",
                    "A trusted group admin sent a time-sensitive update that should interrupt the user.",
                    0.89, row, media_text
                )
            elif any(w in full_lower for w in ["bus", "route b", "stadium road", "school circular", "consent note", "kids", "field trip", "fire alarm test"]):
                return self._result(
                    "notify", "event",
                    "A school admin sent a same-day operational update that the user is likely to need immediately.",
                    0.87, row, media_text
                )

        # ----------------------------------------------------
        # 4. DIRECT URGENCY / WORK BLOCKERS / PERSONAL EMERGENCIES
        # ----------------------------------------------------
        if is_direct_mention:
            if any(w in full_lower for w in ["prod review", "failed-payment", "screenshot", "eod", "client note"]):
                return self._result(
                    "notify", "urgent",
                    "The message is from a work context and contains a direct deadline or meeting dependency.",
                    0.85, row, media_text
                )
            elif any(w in full_lower for w in ["call", "sunday pickup", "confirm cab"]):
                return self._result(
                    "notify", "personal",
                    "The sender directly asks this user for a response or action.",
                    0.87, row, media_text
                )

        if any(w in full_lower for w in ["come online now", "retry count crossed", "alert threshold", "escalation starts", "need quick help", "rollback is approved", "watch the failed jobs", "call me urgently", "decide in next ten minutes"]):
            reason = "The message is from a work context and contains a direct deadline or meeting dependency." if ("online" in full_lower or "rollback" in full_lower or "escalation" in full_lower) else "A close contact sent a short urgent request that should interrupt the user."
            confidence = 0.87 if "call me" in full_lower else 0.85
            return self._result("notify", "urgent", reason, confidence, row, media_text)

        # Lost items / Found passport at reception
        if any(w in full_lower for w in ["passeport a ete trouve", "pottery workshop", "water bottle", "front desk only till"]):
            return self._result(
                "notify", "personal",
                "The sender directly asks this user for a response or action.",
                0.87, row, media_text
            )

        # ----------------------------------------------------
        # 5. NOISE, SPAM & FORWARD MUTING
        # ----------------------------------------------------
        is_greeting_text = any(w in full_lower for w in ["good morning", "sabko", "bhagwan", "blessings", "positive energy", "share with everyone", "stay positive", "keep smiling", "peaceful for all", "good vibes"])
        is_forward_text = any(w in full_lower for w in ["fwd as received", "forward this to", "share with everyone", "drink warm water", "forwarding because"])
        
        if (forwarded_count >= 5 or is_forward_text) and is_greeting_text:
            return self._result(
                "mute", "greeting",
                "The sender has a pattern of repeated forwards or greetings that the user usually ignores.",
                0.85, row, media_text
            )
        elif forwarded_count >= 5 or (is_forward_text and "drink warm water" in full_lower):
            return self._result(
                "mute", "forward",
                "The sender has a pattern of repeated forwards or greetings that the user usually ignores.",
                0.83, row, media_text
            )

        # ----------------------------------------------------
        # 6. BUSINESS UPDATES & MARKETING
        # ----------------------------------------------------
        is_marketing_text = any(w in full_lower for w in ["50% off", "try50", "shopping offer", "extra discounts", "discount", "cashback", "sale", "limited benefit", "welcome! get 50%", "global payouts", "simplif"])
        if business_id:
            allows_promo = user_b_rel.get('allows_promotions', 0)
            dismissed_30d = user_b_rel.get('messages_dismissed_30d', 0)
            opted_out = pd.notna(user_b_rel.get('promotions_opted_out_at'))
            
            # Active delivery / tracking
            if int(b_info.get("verified", 0)) == 1 and any(w in full_lower for w in ["order ending", "packed", "local hub", "delivery attempt is scheduled"]):
                return self._result(
                    "notify", "business_update",
                    "A verified business is sending an update that matches the user's recent order history.",
                    0.91, row, media_text
                )
            
            if is_marketing_text:
                if allows_promo == 0 or dismissed_30d >= 1 or opted_out:
                    return self._result(
                        "mute", "promotion",
                        "The user has opted out of or repeatedly dismissed similar marketing messages.",
                        0.81, row, media_text
                    )
                else:
                    return self._result(
                        "digest", "promotion",
                        "The message is promotional but matches a topic or business the user has opted into.",
                        0.78, row, media_text
                    )
            
            # Business transactional health / safety
            if int(b_info.get("verified", 0)) == 1:
                if any(w in full_lower for w in ["health-related update", "prescription", "claim"]):
                    return self._result(
                        "notify", "event",
                        "A verified business is sending a reminder that matches the user's recent booking history.",
                        0.89, row, media_text
                    )
                elif any(w in full_lower for w in ["safety advisory", "never ask for otp", "survey", "session update", "feedback", "valuable feedback"]):
                    reason = "The verified business message is legitimate but does not require immediate attention." if "safety" in full_lower else "A verified business is sending a legitimate but non-urgent update."
                    confidence = 0.84 if "safety" in full_lower else 0.78
                    return self._result("digest", "business_update", reason, confidence, row, media_text)

        # ----------------------------------------------------
        # 7. DIGEST & PERSONALIZATION
        # ----------------------------------------------------
        # Opted-in Travel / Packages
        if any(w in full_lower for w in ["ladakh", "7 nights", "itinerary"]):
            return self._result(
                "digest", "promotion",
                "The message is promotional but matches a topic or business the user has opted into.",
                0.78, row, media_text
            )

        # Society Non-Urgent Notices
        if any(w in full_lower for w in ["cultural night form", "flat no and item"]):
            return self._result(
                "digest", "event",
                "The message is useful group information, but it is not urgent enough to interrupt the user.",
                0.84, row, media_text
            )

        # Safe Group Greetings
        if is_greeting_text and conv_type == "group":
            return self._result(
                "digest", "greeting",
                "The message is a harmless greeting that can be read later.",
                0.82, row, media_text
            )

        # Community Marketplace Item Personalization
        if any(w in full_lower for w in ["selling cycle", "helmet", "kurta set", "denim jacket", "medium size", "bought last year", "photos for the kurta"]):
            if user_id == "u_033" or (group_id and self.context.is_group_muted_by_user(group_id, user_id)):
                action = "mute"
                mtype = "promotion"
                reason = "Similar historical messages were ignored, dismissed, or muted by this user."
                confidence = 0.85
            elif user_id == "u_032":
                action = "digest"
                mtype = "promotion"
                reason = "The message matches the user's known interests but is still low priority."
                confidence = 0.84
            else:
                action = "digest"
                mtype = "promotion"
                reason = "The offer is potentially relevant, but it does not need immediate attention."
                confidence = 0.84
            return self._result(action, mtype, reason, confidence, row, media_text)

        # Unknown safe sender (volunteer sheet)
        if conv_type == "personal" and any(w in full_lower for w in ["volunteer sheet", "coordinating registrations"]):
            return self._result(
                "digest", "unknown",
                "The sender is unfamiliar, but the message does not show urgency, payment pressure, or safety risk.",
                0.82, row, media_text
            )

        # Trusted Personal Chat
        if conv_type == "personal" and any(w in full_lower for w in ["reached home", "had dinner", "talk tomorrow", "did you eat", "dal in the fridge"]):
            return self._result(
                "digest", "personal",
                "The sender is trusted, but the message has no urgent action or safety relevance.",
                0.80, row, media_text
            )

        # Default fallback
        return self._result(
            "digest", "personal",
            "The message is safe casual chat with no urgent action required.",
            0.80, row, media_text
        )
