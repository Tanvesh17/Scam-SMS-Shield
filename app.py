"""
Flask backend for Scam SMS Shield.

    python app.py     then open  http://127.0.0.1:5000
(needs model.keras + vocab.json from train.py)
"""
import json
import os
import threading

from flask import Flask, jsonify, request, send_from_directory

import textutil as tu

BASE = os.path.dirname(os.path.abspath(__file__))
MAX_WORDS_EXPLAINED = 80


def explain(predictor, text):
    """Score the message, then drop each word in turn to see which words raise the scam score most."""
    words = text.split()
    cap = words[:MAX_WORDS_EXPLAINED]
    variants = [text] + [" ".join(words[:i] + words[i + 1:]) for i in range(len(cap))]
    probs = predictor.predict(variants)
    p = probs[0]
    impacts = [max(0.0, p - q) for q in probs[1:]]
    if p < 0.30:  # nothing worth highlighting in a message that looks fine
        impacts = [0.0] * len(impacts)
    tokens = [{"w": w, "i": round(min(1.0, imp / 0.30), 2)} for w, imp in zip(cap, impacts)]
    tokens += [{"w": w, "i": 0} for w in words[MAX_WORDS_EXPLAINED:]]
    return p, tokens


def create_app(predictor, base=BASE):
    app = Flask(__name__, static_folder=os.path.join(base, "static"))
    app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024
    lock = threading.Lock()

    class Safe:  # one prediction at a time keeps Keras happy under the dev server
        def predict(self, texts):
            with lock:
                return predictor.predict(texts)

    model = Safe()

    @app.get("/")
    def index():
        return send_from_directory(app.static_folder, "index.html")

    @app.get("/api/stats")
    def stats():
        path = os.path.join(base, "metrics.json")
        if not os.path.exists(path):
            return jsonify(error="No metrics yet."), 404
        with open(path, encoding="utf-8") as f:
            return jsonify(json.load(f))

    @app.post("/api/check")
    def check():
        data = request.get_json(silent=True) or {}
        text = (data.get("text") or "").strip()[:1000]
        if not text:
            return jsonify(error="Paste a message first."), 400
        p, tokens = explain(model, text)
        found = tu.signals(text)
        final_verdict = tu.safety_verdict(p, found)
        return jsonify(
            score=round(p * 100),
            model_score=round(p * 100),
            verdict=final_verdict,
            adjusted=final_verdict != tu.verdict(p),
            tokens=tokens,
            signals=found,
        )

    @app.post("/api/bulk")
    def bulk():
        if request.files.get("file"):
            f = request.files["file"]
            raw = f.read()
            try:
                text = raw.decode("utf-8-sig")
            except UnicodeDecodeError:
                text = raw.decode("latin-1")
            msgs = tu.parse_messages(text, f.filename or "")
        else:
            data = request.get_json(silent=True) or {}
            msgs = [str(m).strip()[:500] for m in data.get("messages", []) if str(m).strip()][:500]
        if not msgs:
            return jsonify(error="No messages found. Use a .txt file with one message per line, or a .csv with a 'text' column."), 400

        probs = []
        for i in range(0, len(msgs), 200):
            probs += model.predict(msgs[i:i + 200])
        results = []
        for message, probability in zip(msgs, probs):
            found = tu.signals(message)
            final_verdict = tu.safety_verdict(probability, found)
            results.append({
                "text": message,
                "score": round(probability * 100),
                "model_score": round(probability * 100),
                "verdict": final_verdict,
                "adjusted": final_verdict != tu.verdict(probability),
            })
        summary = {"total": len(results)}
        for v in ("scam", "suspicious", "safe"):
            summary[v] = sum(1 for r in results if r["verdict"] == v)
        return jsonify(results=results, summary=summary)

    @app.errorhandler(413)
    def too_big(_):
        return jsonify(error="That file is too large. Keep it under 2 MB."), 413

    return app


if __name__ == "__main__":
    from predictor import Predictor

    application = create_app(Predictor(BASE))
    print("Open http://127.0.0.1:5000 in your browser")
    application.run(host="127.0.0.1", port=5000, debug=False)
