"""
Train the BiLSTM scam-SMS classifier.

    python train.py              # uses ./data (spam.csv + india_extra.csv)
    python train.py --epochs 6   # faster

Creates: model.keras, vocab.json, config.json, metrics.json, training.png, confusion.png
"""
import argparse
import json
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

import textutil as tu

PROBES = [
    "Dear customer, your account will be blocked today. Update your PAN now at http://sbi-update.xyz",
    "Congratulations!! You won a free iPhone. Click the link to claim your prize now.",
    "URGENT! Your bank KYC has expired. Share the OTP sent to your phone to avoid suspension.",
    "Your OTP is 482913. Do not share it with anyone.",
    "Hey, lunch at 1? Meet me at the canteen.",
    "Please submit the lab record by Friday. Bring the printed copy.",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data", help="folder that holds spam.csv (default: ./data)")
    ap.add_argument("--epochs", type=int, default=12)
    args = ap.parse_args()

    # ------------------------------------------------------------- data
    path, rows = tu.load_main(args.data)
    if not rows:
        raise SystemExit(
            f"Could not find the SMS dataset in '{args.data}'.\n"
            "Put spam.csv (Kaggle) or SMSSpamCollection (UCI) inside the data folder."
        )
    extra = tu.load_extra(args.data)
    print(f"Dataset file : {path}")
    print(f"Messages     : {len(rows)} ({sum(l for l, _ in rows)} spam) after removing duplicates")
    print(f"India extras : {len(extra)} messages" if extra else "India extras : none found (data/india_extra.csv)")

    texts = [t for _, t in rows]
    y = np.array([l for l, _ in rows])
    x_tr, x_tmp, y_tr, y_tmp = train_test_split(texts, y, test_size=0.2, stratify=y, random_state=42)
    x_va, x_te, y_va, y_te = train_test_split(x_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=42)

    # India-style messages go into training only, so the test score stays an honest standard benchmark
    x_tr = x_tr + [t for _, t in extra]
    y_tr = np.concatenate([y_tr, np.array([l for l, _ in extra], dtype=int)]) if extra else y_tr
    order = np.random.RandomState(42).permutation(len(x_tr))
    x_tr = [x_tr[i] for i in order]
    y_tr = y_tr[order]

    vocab = tu.build_vocab([tu.tokenize(t) for t in x_tr])
    vocab_size = len(vocab) + 2
    print(f"Train {len(x_tr)} | validation {len(x_va)} | test {len(x_te)} | vocabulary {vocab_size}")

    X_tr, X_va, X_te = (tu.encode(s, vocab) for s in (x_tr, x_va, x_te))

    # ------------------------------------------------------------ model
    try:
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers
    except ImportError:
        import sys
        raise SystemExit(
            f"TensorFlow is not installed for this Python ({sys.version.split()[0]}).\n"
            "Run: pip install tensorflow\n"
            "If pip says no matching distribution, install Python 3.12 and use it for this project."
        )

    keras.utils.set_random_seed(42)
    model = keras.Sequential(
        [
            layers.Input(shape=(tu.SEQ_LEN,)),
            layers.Embedding(vocab_size, 64, mask_zero=True),
            layers.Bidirectional(layers.LSTM(64, dropout=0.2)),
            layers.Dense(32, activation="relu"),
            layers.Dropout(0.4),
            layers.Dense(1, activation="sigmoid"),
        ],
        name="scam_sms_bilstm",
    )
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="binary_crossentropy", metrics=["accuracy"])
    model.summary()

    hist = model.fit(
        X_tr, y_tr.astype("float32"),
        validation_data=(X_va, y_va.astype("float32")),
        epochs=args.epochs, batch_size=32,
        class_weight={0: 1.0, 1: 2.0},          # missing a scam is worse than a false alarm
        callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)],
        verbose=2,
    )

    # --------------------------------------------------------- evaluate
    probs = model.predict(X_te, batch_size=256, verbose=0).ravel()
    pred = (probs >= 0.5).astype(int)
    metrics = {
        "accuracy": round(100 * accuracy_score(y_te, pred), 1),
        "precision": round(100 * precision_score(y_te, pred, zero_division=0), 1),
        "recall": round(100 * recall_score(y_te, pred, zero_division=0), 1),
        "f1": round(100 * f1_score(y_te, pred, zero_division=0), 1),
        "test_size": int(len(y_te)),
        "train_size": int(len(y_tr)),
        "india_extra": int(len(extra)),
    }
    print("\nTEST RESULTS (messages the model never saw)")
    for k in ("accuracy", "precision", "recall", "f1"):
        print(f"  {k:<9}: {metrics[k]}%")
    cm = confusion_matrix(y_te, pred)
    print("  confusion matrix [[ham ok, ham->spam], [spam->ham, spam ok]]:", cm.tolist())

    # ------------------------------------------------------------- save
    model.save("model.keras")
    with open("vocab.json", "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False)
    with open("config.json", "w", encoding="utf-8") as f:
        json.dump({"seq_len": tu.SEQ_LEN}, f)
    with open("metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print("\nSaved model.keras, vocab.json, config.json, metrics.json")

    # ---------------------------------------------------------- sanity
    print("\nSanity check on new messages (scam risk):")
    p = model.predict(tu.encode(PROBES, vocab), verbose=0).ravel()
    for text, prob in zip(PROBES, p):
        print(f"  {prob * 100:5.1f}%  {tu.verdict(float(prob)):<10} {text[:70]}")

    # ------------------------------------------------------------ plots
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        ax[0].plot(hist.history["accuracy"], label="train")
        ax[0].plot(hist.history["val_accuracy"], label="validation")
        ax[0].set_title("Accuracy"); ax[0].set_xlabel("epoch"); ax[0].legend()
        ax[1].plot(hist.history["loss"], label="train")
        ax[1].plot(hist.history["val_loss"], label="validation")
        ax[1].set_title("Loss"); ax[1].set_xlabel("epoch"); ax[1].legend()
        fig.tight_layout(); fig.savefig("training.png", dpi=120); plt.close(fig)

        fig, a = plt.subplots(figsize=(4, 4))
        a.imshow(cm, cmap="Blues")
        a.set_xticks([0, 1]); a.set_xticklabels(["safe", "scam"]); a.set_yticks([0, 1]); a.set_yticklabels(["safe", "scam"])
        a.set_xlabel("predicted"); a.set_ylabel("actual"); a.set_title("Confusion matrix")
        for i in range(2):
            for j in range(2):
                a.text(j, i, int(cm[i, j]), ha="center", va="center", color="black")
        fig.tight_layout(); fig.savefig("confusion.png", dpi=120); plt.close(fig)
        print("Saved training.png and confusion.png")
    except Exception as e:
        print("Skipped plots:", e)


if __name__ == "__main__":
    main()
