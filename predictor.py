"""Loads the trained BiLSTM and scores messages. Used by app.py."""
import json
import os

import textutil as tu


class Predictor:
    def __init__(self, base):
        model_path = os.path.join(base, "model.keras")
        vocab_path = os.path.join(base, "vocab.json")
        if not (os.path.exists(model_path) and os.path.exists(vocab_path)):
            raise SystemExit("model.keras / vocab.json not found. Run  python train.py  first.")

        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        from tensorflow import keras  # imported here so a missing TensorFlow gives a clear message

        self.model = keras.models.load_model(model_path)
        with open(vocab_path, encoding="utf-8") as f:
            self.vocab = json.load(f)
        cfg_path = os.path.join(base, "config.json")
        self.seq_len = tu.SEQ_LEN
        if os.path.exists(cfg_path):
            with open(cfg_path, encoding="utf-8") as f:
                self.seq_len = json.load(f).get("seq_len", tu.SEQ_LEN)

    def predict(self, texts):
        """list of messages -> list of scam probabilities (0..1)."""
        x = tu.encode(texts, self.vocab, self.seq_len)
        probs = self.model.predict(x, batch_size=256, verbose=0)
        return [float(p) for p in probs.ravel()]
