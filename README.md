# Scam SMS Shield (BiLSTM + Flask)

Paste an SMS and a BiLSTM (a type of RNN) tells you if it is a scam, highlights the words that raised the alarm and lists the warning signs. A second tab checks a whole file of messages at once.

## Folder layout

```
scam-sms-shield/
├── data/
│   ├── spam.csv           <- YOU add this (SMS Spam Collection from Kaggle)
│   └── india_extra.csv    <- included: India-style scam / genuine messages
├── static/index.html      frontend
├── textutil.py            tokenizer, data loading, warning-sign checks
├── train.py               trains the BiLSTM
├── predictor.py           loads the trained model
├── app.py                 Flask backend (serves the frontend too)
├── sample_messages.csv    a file to try in the "Check many messages" tab
└── requirements.txt
```

## 1. Check your Python / TensorFlow first

```
python --version
pip install -r requirements.txt
python -c "import tensorflow as tf; print(tf.__version__)"
```

- It prints a version number: you are good, go to step 2.
- `No matching distribution found for tensorflow`, or an import error: your Python is too new for the TensorFlow build pip can find. Install **Python 3.12** from python.org, then create the project environment with it:

```
py -3.12 -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

(Mac/Linux: `python3.12 -m venv venv` and `source venv/bin/activate`.)

## 2. Put the dataset in place

Copy `spam.csv` into the `data/` folder, next to `india_extra.csv`. The original UCI file (`SMSSpamCollection`) also works.

## 3. Train (1 to 3 minutes on CPU)

```
python train.py
```

It prints the test accuracy, precision and recall and writes `model.keras`, `vocab.json`, `metrics.json`, `training.png` and `confusion.png`. Show the two PNGs to your teacher.

## 4. Run the app

```
python app.py
```

Open http://127.0.0.1:5000

## Demo script (2 minutes)

1. Show the header line with the test accuracy and the share of scams caught.
2. Click the first sample (a fake KYC message). The stamp lands, the scam words are highlighted, the warning signs are listed.
3. Click the OTP sample. It is marked safe because "do not share" is how real banks word it.
4. Type your own scam message live.
5. Switch to "Check many messages", click "Try a sample batch" or upload `sample_messages.csv`, then download the CSV.

## How it works (viva notes)

- **Dataset**: SMS Spam Collection, about 5,500 English SMS labelled ham or spam. Duplicates are removed, then 80 / 10 / 10 split into train, validation and test.
- **India-style messages**: `data/india_extra.csv` holds about 340 messages (fake KYC, UPI, parcel, job scams and genuine bank / OTP / college messages) that were written from templates for this project. They are used for training only, so the test score is on the standard dataset. Say this openly if asked.
- **Preprocessing**: lower-case, links become `urltoken`, phone numbers `phonetoken`, numbers `numtoken`, then words are turned into ids (vocabulary of the 8,000 most common words) and padded to 50 tokens.
- **Embedding**: each word id becomes a 64-number vector that the model learns.
- **LSTM**: an RNN with a memory cell, so it reads the message in order and remembers earlier words. **Bidirectional** means one LSTM reads left to right and one right to left.
- **Dropout**: randomly switches off neurons while training to reduce overfitting.
- **Sigmoid output**: one number between 0 and 1, the probability of scam. Below 30% is safe, 30 to 70% suspicious, above 70% scam.
- **Safety cross-check**: warning-pattern rules run alongside the RNN. A link, urgent request, prize lure, or request to pay a new number can raise a low RNN result to suspicious; direct requests for OTPs, PINs or similar secrets, and combinations such as prize links with urgency, can raise it to high risk. The displayed RNN estimate stays separate so disagreements are visible.
- **Class weight**: scams are about 13% of the data, so a missed scam is counted twice as costly.
- **Highlighted words**: the app removes one word at a time and re-scores the message. Words whose removal drops the score the most are highlighted. This is called occlusion.
- **Warning signs**: simple rules (links, urgency, asking for OTP / PIN, prize language) shown next to the model's score.

## Limits (be upfront)

English only for now, so Hindi and Hinglish scams are future scope. The model gives a probability, not proof, and new scam wording can fool it. Use it as a helper, not as the only check.

## Troubleshooting

| Problem | Fix |
|---|---|
| `Could not find the SMS dataset` | `spam.csv` is not inside the `data/` folder. |
| `TensorFlow is not installed for this Python` | See step 1 (use Python 3.12). |
| `model.keras / vocab.json not found` | Run `python train.py` before `python app.py`. |
| Page says it can't reach the checker | `python app.py` is not running, or you closed the terminal. |
| Port 5000 is busy | Change `port=5000` at the bottom of `app.py`. |
