"""
Text helpers shared by train.py and app.py (no TensorFlow needed here).

- loading the SMS Spam Collection (Kaggle spam.csv or the original UCI tab file)
- tokenizing / encoding messages for the BiLSTM
- plain-language "warning sign" checks
- parsing uploaded files for the bulk checker
"""
import csv
import io
import os
import re
from collections import Counter

import numpy as np

SEQ_LEN = 50          # messages are cut / padded to this many tokens
PAD, UNK = 0, 1       # reserved ids

# ---------------------------------------------------------------- tokenizing
_URL = re.compile(
    r"(?:https?://\S+|www\.\S+"
    r"|\b(?:bit\.ly|tinyurl\.com|goo\.gl|rb\.gy|t\.co)/\S*"
    r"|\b[a-z0-9][a-z0-9-]*\.(?:xyz|top|club|site|online|link|info|co|in|com|net|org)\b(?:/\S*)?)",
    re.I,
)
_PHONE = re.compile(r"(?<!\d)(?:\+?91[- ]?)?[6-9]\d{9}(?!\d)")
_NUM = re.compile(r"\d+(?:[.,]\d+)*")
_TOKEN = re.compile(r"[a-z]+(?:'[a-z]+)?|[ऀ-ॿ]+|[!?£$₹%]")


def tokenize(text):
    """Lower-case, replace links / phone numbers / numbers with placeholders, split into tokens."""
    t = str(text).lower()
    t = _URL.sub(" urltoken ", t)
    t = _PHONE.sub(" phonetoken ", t)
    t = _NUM.sub(" numtoken ", t)
    return _TOKEN.findall(t)


def build_vocab(token_lists, max_size=8000, min_freq=2):
    counts = Counter(tok for toks in token_lists for tok in toks)
    words = [w for w, c in counts.most_common(max_size) if c >= min_freq]
    return {w: i + 2 for i, w in enumerate(words)}


def encode(texts, vocab, seq_len=SEQ_LEN):
    """texts -> int32 array (n, seq_len), right-padded with 0."""
    out = np.zeros((len(texts), seq_len), dtype="int32")
    for r, text in enumerate(texts):
        ids = [vocab.get(tok, UNK) for tok in tokenize(text)][:seq_len]
        out[r, : len(ids)] = ids
    return out


# ------------------------------------------------------------------ loading
def _read_text(path):
    raw = open(path, "rb").read()
    for enc in ("utf-8-sig", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1", errors="ignore")


def find_main_file(data_dir):
    """Find spam.csv (Kaggle) or SMSSpamCollection (UCI) inside data_dir."""
    if not os.path.isdir(data_dir):
        return None
    names = {f.lower(): f for f in os.listdir(data_dir)}
    for cand in ("spam.csv", "smsspamcollection", "smsspamcollection.txt", "smsspamcollection.tsv", "sms.tsv", "sms_spam.csv"):
        if cand in names:
            return os.path.join(data_dir, names[cand])
    for low in sorted(names):
        if low.endswith((".csv", ".tsv", ".txt")) and "india" not in low and "sample" not in low:
            return os.path.join(data_dir, names[low])
    return None


def parse_labeled(text):
    """Return [(label, message)] with label 1 = spam, 0 = ham. Handles Kaggle csv and UCI tab files."""
    text = text.lstrip("﻿")
    first = text.splitlines()[0] if text.strip() else ""
    tabbed = "\t" in first and first.split("\t")[0].strip().lower() in ("ham", "spam")
    if tabbed:
        reader = csv.reader(io.StringIO(text), delimiter="\t", quoting=csv.QUOTE_NONE)
    else:
        reader = csv.reader(io.StringIO(text))
    out = []
    for r in reader:
        if len(r) < 2:
            continue
        a, b = r[0].strip().lower(), r[1].strip().lower()
        if a in ("ham", "spam"):
            out.append((1 if a == "spam" else 0, r[1].strip()))
        elif b in ("ham", "spam"):
            out.append((1 if b == "spam" else 0, r[0].strip()))
    return out


def dedupe(rows):
    seen, out = set(), []
    for label, msg in rows:
        key = msg.strip().lower()
        if key and key not in seen:
            seen.add(key)
            out.append((label, msg))
    return out


def load_main(data_dir):
    path = find_main_file(data_dir)
    if not path:
        return None, []
    return path, dedupe(parse_labeled(_read_text(path)))


def load_extra(data_dir):
    path = os.path.join(data_dir, "india_extra.csv")
    if not os.path.exists(path):
        return []
    return dedupe(parse_labeled(_read_text(path)))


# ------------------------------------------------------------- bulk uploads
def parse_messages(raw, filename=""):
    """Turn an uploaded .txt (one message per line) or .csv/.tsv into a list of messages."""
    name = filename.lower()
    if name.endswith((".csv", ".tsv")):
        delim = "\t" if name.endswith(".tsv") else ","
        rows = [r for r in csv.reader(io.StringIO(raw), delimiter=delim) if any(c.strip() for c in r)]
        if not rows:
            return []
        header = [c.strip().lower() for c in rows[0]]
        idx = None
        for key in ("text", "message", "sms", "v2", "body", "content", "msg"):
            if key in header:
                idx = header.index(key)
                rows = rows[1:]
                break
        if idx is None:
            idx = 0
            if len(rows[0]) > 1 and rows[0][0].strip().lower() in ("ham", "spam"):
                idx = 1
        msgs = [r[idx] for r in rows if len(r) > idx]
    else:
        msgs = raw.splitlines()
    msgs = [m.strip()[:500] for m in msgs if m and m.strip()]
    return msgs[:500]


# ----------------------------------------------------------------- verdicts
def verdict(p):
    if p >= 0.70:
        return "scam"
    if p >= 0.30:
        return "suspicious"
    return "safe"


_LINK = _URL
_URGENT = re.compile(
    r"\b(urgent|urgently|immediately|within \d+|expire[sd]?|blocked|suspended|deactivated|disconnect(?:ed)?|"
    r"frozen|last chance|final notice|hurry|limited time|asap)\b", re.I)
_SENSITIVE = re.compile(r"\b(otp|pin|cvv|password|passcode|aadhaar|aadhar|pan|kyc|net ?banking|card number|upi pin)\b", re.I)
_ASKS = re.compile(r"\b(share|send|enter|provide|submit|verify|update|confirm|reply|complete)\b", re.I)
_SAFE_OTP = re.compile(r"(do not|don't|never)\s+share", re.I)
_PRIZE = re.compile(
    r"\b(won|win|winner|prize|lottery|lucky draw|cashback|reward|refund|free|gift|jackpot|congratulations|congrats)\b", re.I)
_GREETING = re.compile(r"\bdear\s+(customer|user|sir|madam|account holder|valued|student|member)\b", re.I)
_CONTACT = re.compile(r"\b(call|whatsapp|contact|message|telegram)\b", re.I)
_NEW_NUMBER = re.compile(
    r"\b(?:lost|changed|new)\s+(?:my\s+)?(?:phone|number|sim)\b|\bnew\s+number\b", re.I)
_MONEY_REQUEST = re.compile(
    r"\b(send|transfer|pay)\b.{0,45}\b(?:money|cash|funds|rupees?|rs\.?|upi|account)\b|"
    r"\b(?:money|cash|funds|rupees?|rs\.?|upi|account)\b.{0,45}\b(send|transfer|pay)\b", re.I)
_AUTH_SECRET = re.compile(r"\b(otp|pin|cvv|password|passcode|upi pin|card number)\b", re.I)


def safety_verdict(p, found):
    """Combine the RNN estimate with independent warning patterns.

    Pattern checks may raise a verdict, but never lower the model's verdict.
    The numeric score remains the raw RNN estimate so the UI can explain a disagreement.
    """
    model_label = verdict(p)
    if model_label == "scam":
        return model_label

    ids = {item["id"] for item in found}
    if ("auth_request" in ids
            or ("link" in ids and ("urgency" in ids or "prize" in ids))):
        return "scam"
    if ids & {"link", "urgency", "private", "prize", "contact", "new_number"}:
        return "suspicious"
    return model_label


def signals(text):
    """Plain-language warning signs found in a message."""
    found = []
    if _LINK.search(text):
        found.append(("link", "Contains a link",
                      "Scammers hide fake bank and prize pages behind short or look-alike links."))
    if _URGENT.search(text):
        found.append(("urgency", "Pushes you to act fast",
                      "Threats like \"blocked today\" are meant to make you act before you think."))
    if (_SENSITIVE.search(text) and not _AUTH_SECRET.search(text)
            and _ASKS.search(text) and not _SAFE_OTP.search(text)):
        found.append(("private", "Asks for private details",
                      "Banks never ask for your OTP, PIN, CVV or password over SMS."))
    if _PRIZE.search(text):
        found.append(("prize", "Offers a prize or free money",
                      "Unexpected winnings, cashback and refunds are a classic lure."))
    if _GREETING.search(text):
        found.append(("greeting", "Generic greeting",
                      "A real sender usually knows and uses your name."))
    if _CONTACT.search(text) and _PHONE.search(text):
        found.append(("contact", "Asks you to contact a number",
                      "Check the number on your card or the official website instead."))
    if _NEW_NUMBER.search(text) and _MONEY_REQUEST.search(text):
        found.append(("new_number", "Requests money through a new or changed number",
                      "A sudden change of number is a common impersonation trick. Verify with the person using a number you already trust."))
    if _AUTH_SECRET.search(text) and _ASKS.search(text) and not _SAFE_OTP.search(text):
        found.append(("auth_request", "Requests a login or payment secret",
                      "Never share an OTP, PIN, CVV or password by text, even if the message looks official."))
    return [{"id": i, "label": l, "why": w} for i, l, w in found]
