import os
import json
import random
import time
import subprocess
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from torch.utils.data import TensorDataset, DataLoader
from sentence_transformers import SentenceTransformer

import matplotlib.pyplot as plt

import requests

HF_API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"
HF_HEADERS = {"Authorization": f"Bearer {""}"}
# Insert access toke for code to function

def run_hf(prompt):
    response = requests.post(HF_API_URL, headers=HF_HEADERS, json={"inputs": prompt})
    return response.json()[0]["generated_text"]


#--------------------------
# Config and reproducibility
#--------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Efficiency knobs
ROUNDS = 3                 # number of adversarial rounds
N_GEN_FALSE_PER_ROUND = 60 # minimal generation per round (false claims)
N_GEN_TRUE_PER_ROUND  = 40 # optional true claims per round
MIX_RATIO = 0.5            # generator proportion relative to base dataset size
HARD_TOP_K = 40            # how many hard examples to use for training
VAL_CHECK = True           # print val accuracy during training
EPOCHS = 8                 # detector epochs per round
BATCH_SIZE = 32
LR = 3e-4

# Paths
BASE_TRAIN = "base_train.jsonl"
BASE_VAL   = "base_val.jsonl"
BASE_TEST  = "base_test.jsonl"
CACHE_DIR  = "gen_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

#--------------------------
# IO helpers
#--------------------------
def save_jsonl(items, path):
    with open(path, "w", encoding="utf-8") as f:
        for it in items: f.write(json.dumps(it) + "\n")

def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def ensure_base_splits(sample_n=1500):
    if all(os.path.exists(p) for p in [BASE_TRAIN, BASE_VAL, BASE_TEST]):
        print("Loaded existing base splits.")
        return read_jsonl(BASE_TRAIN), read_jsonl(BASE_VAL), read_jsonl(BASE_TEST)

    dataset = load_dataset("HiTZ/This-is-not-a-dataset")
    df = dataset["train"].to_pandas().sample(sample_n, random_state=SEED)
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=SEED, stratify=df["label"])
    val_df, test_df   = train_test_split(temp_df, test_size=0.5, random_state=SEED, stratify=temp_df["label"])

    def wrap_df(dfx):
        return [{"text": r["sentence"], "label": int(r["label"]), "difficulty": 0, "domain": "dataset"} for _, r in dfx.iterrows()]
    base_train = wrap_df(train_df)
    base_val   = wrap_df(val_df)
    base_test  = wrap_df(test_df)

    save_jsonl(base_train, BASE_TRAIN)
    save_jsonl(base_val, BASE_VAL)
    save_jsonl(base_test, BASE_TEST)
    print("Created and saved base splits.")
    return base_train, base_val, base_test

#--------------------------
# Embeddings and detector
#--------------------------
embedder = SentenceTransformer("all-MiniLM-L6-v2")

def embed(texts, batch_size=256):
    # Batched encoding for speed/memory efficiency
    vecs = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i+batch_size]
        vecs.append(embedder.encode(chunk, convert_to_numpy=True))
    return np.vstack(vecs) if vecs else np.zeros((0, 384))

class DetectorNN(nn.Module):
    def __init__(self, input_dim=384, hidden_dim=256, dropout=0.3):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(hidden_dim // 2, 2)

    def forward(self, x):
        x = self.drop1(self.relu1(self.fc1(x)))
        x = self.drop2(self.relu2(self.fc2(x)))
        return self.fc3(x)

def train_detector(train_texts, train_labels, val_texts=None, val_labels=None,
                   epochs=EPOCHS, lr=LR, batch_size=BATCH_SIZE):
    X_train = embed(train_texts)
    ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                       torch.tensor(train_labels, dtype=torch.long))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    model = DetectorNN(input_dim=X_train.shape[1])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    for ep in range(epochs):
        model.train()
        total_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
        print(f"Epoch {ep+1}/{epochs}, Loss: {total_loss/len(loader.dataset):.4f}")

        if VAL_CHECK and val_texts is not None:
            model.eval()
            X_val = embed(val_texts)
            X_val_t = torch.tensor(X_val, dtype=torch.float32)
            with torch.no_grad():
                preds = torch.argmax(model(X_val_t), dim=1).numpy()
            acc = accuracy_score(val_labels, preds)
            print(f"  Val Accuracy: {acc:.3f}")

    return model

def evaluate_detector(model, test_texts, test_labels, title="Detector"):
    X_test = embed(test_texts)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    with torch.no_grad():
        logits = model(X_test_t)
        preds = torch.argmax(logits, dim=1).numpy()
        probs = torch.softmax(logits, dim=1).numpy()
    acc = accuracy_score(test_labels, preds)
    print(f"\n=== {title} ===")
    print(classification_report(test_labels, preds))
    print("Confusion Matrix:\n", confusion_matrix(test_labels, preds))
    print("Accuracy:", acc)
    return preds, probs, acc

#--------------------------
# Local generator (efficient, cached)
#--------------------------
def run_ollama(model_name, prompt):
    r = subprocess.run(["ollama", "run", model_name], input=prompt.encode(), capture_output=True)
    return r.stdout.decode().strip()

def gen_prompt(label, difficulty, domain):
    base = f"Produce a single-sentence claim about {domain} at difficulty {difficulty}. "
    if label == 0:
        base += "Make it plausible but factually false. Include named entities or numbers when relevant."
    else:
        base += "Make it factually correct and verifiable. Include named entities or numbers when relevant."
    return base

def generate_claims(model_name="generator", n_false=60, n_true=40, difficulty=1, domain="mixed", cache_key=""):
    # Use cache if available
    cache_path = os.path.join(CACHE_DIR, f"claims_{cache_key}_d{difficulty}.jsonl")
    if os.path.exists(cache_path):
        print(f"Loaded cached claims: {cache_path}")
        return read_jsonl(cache_path)

    claims = []
    # Generate false claims first (adversarial focus)
    prompt_false = gen_prompt(label=0, difficulty=difficulty, domain=domain)
    for i in range(n_false):
        text = run_ollama(model_name, prompt_false)
        claims.append({"text": text, "label": 0, "difficulty": difficulty, "domain": domain})
        if (i+1) % 10 == 0: print(f"Gen false {i+1}/{n_false}")

    # Optionally generate true claims (smaller count)
    if n_true > 0:
        prompt_true = gen_prompt(label=1, difficulty=difficulty, domain=domain)
        for i in range(n_true):
            text = run_ollama(model_name, prompt_true)
            claims.append({"text": text, "label": 1, "difficulty": difficulty, "domain": domain})
            if (i+1) % 10 == 0: print(f"Gen true {i+1}/{n_true}")

    save_jsonl(claims, cache_path)
    print(f"Saved generated claims to {cache_path}")
    return claims

#--------------------------
# Hard-example mining
#--------------------------
def mine_hard_examples(claims, preds, probs, top_k=HARD_TOP_K, conf_margin=0.15):
    # Attach predictions
    for i, c in enumerate(claims):
        c["pred"] = int(preds[i])
        c["conf"] = float(probs[i][c["pred"]])

    # Misclassified first
    misclassified = [c for c in claims if c["pred"] != c["label"]]
    # Then low-confidence correct predictions (near decision boundary)
    low_conf = [c for c in claims if c["pred"] == c["label"] and c["conf"] < (0.5 + conf_margin)]

    # Prioritize false claims that fooled the detector
    hard_neg = [c for c in misclassified if c["label"] == 0]
    # Fill remaining slots with other misclassified, then low-conf
    pool = hard_neg + [c for c in misclassified if c["label"] == 1] + low_conf

    # Deduplicate by text
    seen = set()
    unique_pool = []
    for c in pool:
        if c["text"] not in seen:
            unique_pool.append(c)
            seen.add(c["text"])

    return unique_pool[:top_k]

#--------------------------
# Mix builder (base + adversarial)
#--------------------------
def build_training_set(base, adversarial, mix_ratio=MIX_RATIO, balance=True):
    n_base = len(base)
    n_adv  = min(len(adversarial), int(n_base * mix_ratio))
    combined = base + adversarial[:n_adv]

    if not balance:
        texts = [c["text"] for c in combined]
        labels = [int(c["label"]) for c in combined]
        return texts, labels

    pos = [c for c in combined if c["label"] == 1]
    neg = [c for c in combined if c["label"] == 0]
    m = min(len(pos), len(neg))
    if m == 0:
        texts = [c["text"] for c in combined]
        labels = [int(c["label"]) for c in combined]
        return texts, labels

    balanced = pos[:m] + neg[:m]
    random.shuffle(balanced)
    texts = [c["text"] for c in balanced]
    labels = [int(c["label"]) for c in balanced]
    return texts, labels

#--------------------------
# Adversarial loop controller
#--------------------------
def adversarial_loop(rounds=ROUNDS, gen_model="generator", domain="mixed"):
    # Load base splits
    base_train, base_val, base_test = ensure_base_splits(sample_n=1500)

    # Initialize detector on base only (round 0)
    train_texts_0 = [c["text"] for c in base_train]
    train_labels_0 = [int(c["label"]) for c in base_train]
    val_texts = [c["text"] for c in base_val]
    val_labels = [int(c["label"]) for c in base_val]
    test_texts = [c["text"] for c in base_test]
    test_labels = [int(c["label"]) for c in base_test]

    print("\n=== Round 0: Base training ===")
    detector = train_detector(train_texts_0, train_labels_0, val_texts, val_labels)
    preds, probs, acc = evaluate_detector(detector, test_texts, test_labels, title="Round 0 (Base)")

    metrics_log = [{"round": 0, "accuracy": acc, "mix_ratio": 0.0}]

    # Adversarial rounds
    for r in range(1, rounds + 1):
        print(f"\n=== Round {r} ===")
        difficulty = min(1 + r // 2, 3)  # light escalation
        # Efficient generation: small batches, cached
        claims = generate_claims(
            model_name=gen_model,
            n_false=N_GEN_FALSE_PER_ROUND,
            n_true=N_GEN_TRUE_PER_ROUND,
            difficulty=difficulty,
            domain=domain,
            cache_key=f"r{r}"
        )

        # Detect on generated claims
        gen_texts = [c["text"] for c in claims]
        gen_labels = [int(c["label"]) for c in claims]
        preds_g, probs_g, _ = evaluate_detector(detector, gen_texts, gen_labels, title=f"Generator claims (r{r})")

        # Mine hard examples
        hard = mine_hard_examples(claims, preds_g, probs_g, top_k=HARD_TOP_K)
        print(f"Selected hard examples: {len(hard)}")

        # Build training set: base + hard examples
        train_texts, train_labels = build_training_set(base_train, hard, mix_ratio=MIX_RATIO, balance=True)

        # Retrain detector (or fine-tune) on mixed set
        detector = train_detector(train_texts, train_labels, val_texts, val_labels)

        # Evaluate on fixed test set
        preds_t, probs_t, acc_t = evaluate_detector(detector, test_texts, test_labels, title=f"Round {r} (Adversarial)")
        metrics_log.append({"round": r, "accuracy": acc_t, "mix_ratio": MIX_RATIO, "difficulty": difficulty, "hard_examples": len(hard)})

    # Save metrics
    save_jsonl(metrics_log, "adv_metrics.jsonl")
    print("\nSaved adversarial metrics to adv_metrics.jsonl")

    # Plot accuracy per round
    dfm = pd.DataFrame(metrics_log)
    plt.figure(figsize=(7,5))
    plt.plot(dfm["round"], dfm["accuracy"], marker="o")
    plt.xlabel("Round")
    plt.ylabel("Test Accuracy")
    plt.title("Detector accuracy across adversarial rounds")
    plt.grid(True)
    plt.show()

#--------------------------
# Entry point
#--------------------------
if __name__ == "__main__":
    adversarial_loop(rounds=ROUNDS, gen_model="generator", domain="general")
