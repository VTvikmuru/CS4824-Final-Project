import os
import json
import random
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
import seaborn as sns

#--------------------------
# Stable seed
#--------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

#--------------------------
# Paths
#--------------------------
BASE_TRAIN_PATH = "base_train.jsonl"
BASE_VAL_PATH   = "base_val.jsonl"
BASE_TEST_PATH  = "base_test.jsonl"
FAKES_TXT_PATH  = "generated_fakes_50.txt"
FAKES_JSONL_PATH= "generated_fakes_50.jsonl"

#--------------------------
# Utilities: IO
#--------------------------
def save_jsonl(items, path):
    with open(path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it) + "\n")

def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def load_fakes_txt(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

def wrap_claims(sentences, label=0, difficulty=1, domain="general"):
    return [{"text": s, "label": label, "difficulty": difficulty, "domain": domain} for s in sentences]

#--------------------------
# Dataset splits (stable)
#--------------------------
def create_or_load_base_splits(sample_n=1000):
    # If splits exist, load them
    if all(os.path.exists(p) for p in [BASE_TRAIN_PATH, BASE_VAL_PATH, BASE_TEST_PATH]):
        print("Loaded existing base splits.")
        return read_jsonl(BASE_TRAIN_PATH), read_jsonl(BASE_VAL_PATH), read_jsonl(BASE_TEST_PATH)

    # Otherwise, create from HF dataset (sample for speed)
    dataset = load_dataset("HiTZ/This-is-not-a-dataset")
    df = dataset["train"].to_pandas().sample(sample_n, random_state=SEED)

    # Fixed splits with stratification
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=SEED, stratify=df["label"])
    val_df, test_df   = train_test_split(temp_df, test_size=0.5, random_state=SEED, stratify=temp_df["label"])

    # Wrap into schema
    def wrap_df(dfx):
        return [{"text": r["sentence"], "label": int(r["label"]), "difficulty": 0, "domain": "dataset"} for _, r in dfx.iterrows()]

    base_train = wrap_df(train_df)
    base_val   = wrap_df(val_df)
    base_test  = wrap_df(test_df)

    save_jsonl(base_train, BASE_TRAIN_PATH)
    save_jsonl(base_val,   BASE_VAL_PATH)
    save_jsonl(base_test,  BASE_TEST_PATH)

    print("Created and saved base splits.")
    return base_train, base_val, base_test

#--------------------------
# Pre-generate claims to JSONL (from existing fakes txt)
#--------------------------
def ensure_fakes_jsonl():
    if os.path.exists(FAKES_JSONL_PATH):
        print(f"Loaded existing {FAKES_JSONL_PATH}")
        return read_jsonl(FAKES_JSONL_PATH)
    if not os.path.exists(FAKES_TXT_PATH):
        raise FileNotFoundError(f"Missing {FAKES_TXT_PATH}. Place your fake sentences file in the working directory.")

    fakes = load_fakes_txt(FAKES_TXT_PATH)
    wrapped = wrap_claims(fakes, label=0, difficulty=1, domain="generator")
    save_jsonl(wrapped, FAKES_JSONL_PATH)
    print(f"Saved {len(wrapped)} claims to {FAKES_JSONL_PATH}")
    return wrapped

#--------------------------
# Embeddings and model
#--------------------------
embedder = SentenceTransformer("all-MiniLM-L6-v2")
def embed(texts):
    return embedder.encode(texts, convert_to_numpy=True)

class ImprovedNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, dropout=0.3):
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

def train_nn(train_texts, train_labels, val_texts=None, val_labels=None, epochs=10, lr=3e-4, batch_size=32):
    X_train = embed(train_texts)
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(train_labels, dtype=torch.long)
    loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=True)

    model = ImprovedNN(X_train.shape[1])
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

        # Optional simple val check
        if val_texts and val_labels:
            model.eval()
            X_val = embed(val_texts)
            X_val_t = torch.tensor(X_val, dtype=torch.float32)
            with torch.no_grad():
                preds = torch.argmax(model(X_val_t), dim=1).numpy()
            acc = accuracy_score(val_labels, preds)
            print(f"  Val Accuracy: {acc:.3f}")

    return model

#--------------------------
# Mixing and balancing
#--------------------------
def build_training_set(base, gen, mix_ratio=0.5, balance=True):
    """
    mix_ratio: number of generator samples relative to base size (e.g., 0.5 -> add 50% of base size).
    Ensures class balance if balance=True.
    """
    n_base = len(base)
    n_gen  = min(len(gen), int(n_base * mix_ratio))
    combined = base + gen[:n_gen]

    if not balance:
        texts  = [c["text"] for c in combined]
        labels = [int(c["label"]) for c in combined]
        return texts, labels

    pos = [c for c in combined if c["label"] == 1]
    neg = [c for c in combined if c["label"] == 0]
    m = min(len(pos), len(neg))
    if m == 0:
        # Fallback if one class is missing
        texts  = [c["text"] for c in combined]
        labels = [int(c["label"]) for c in combined]
        return texts, labels

    balanced = pos[:m] + neg[:m]
    random.shuffle(balanced)
    texts  = [c["text"] for c in balanced]
    labels = [int(c["label"]) for c in balanced]
    return texts, labels

#--------------------------
# Evaluation
#--------------------------
def evaluate(model, test_texts, test_labels, title="Detector"):
    X_test = embed(test_texts)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    with torch.no_grad():
        preds = torch.argmax(model(X_test_t), dim=1).numpy()

    print(f"\n=== {title} ===")
    print(classification_report(test_labels, preds))
    cm = confusion_matrix(test_labels, preds)
    print("Confusion Matrix:\n", cm)
    acc = accuracy_score(test_labels, preds)
    print("Accuracy:", acc)
    return acc, cm, preds

def plot_confusion(cm, labels=("FAKE","TRUE"), title="Confusion Matrix"):
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

#--------------------------
# Experiment: ratios of generator vs dataset
#--------------------------
def run_ratio_experiment(base_train, base_val, base_test, gen_fakes, ratios=(0, 0.25, 0.5, 0.75, 1.0), epochs=10):
    val_texts  = [c["text"] for c in base_val]
    val_labels = [int(c["label"]) for c in base_val]
    test_texts  = [c["text"] for c in base_test]
    test_labels = [int(c["label"]) for c in base_test]

    results = []
    for r in ratios:
        print(f"\n=== Mix Ratio {int(r*100)}% generator ===")
        train_texts, train_labels = build_training_set(base_train, gen_fakes, mix_ratio=r, balance=True)
        model = train_nn(train_texts, train_labels, val_texts, val_labels, epochs=epochs, lr=3e-4, batch_size=32)
        acc, cm, preds = evaluate(model, test_texts, test_labels, title=f"Mix {int(r*100)}%")
        results.append({"mix_ratio": r, "accuracy": acc})

    df_results = pd.DataFrame(results)
    print("\n=== Final Results ===")
    print(df_results)

    plt.figure(figsize=(8,6))
    plt.plot(df_results["mix_ratio"], df_results["accuracy"], marker="o", label="Improved NN")
    plt.xlabel("Generator Mix Ratio")
    plt.ylabel("Accuracy")
    plt.title("Detector Performance vs Generator Mix Ratio")
    plt.grid(True)
    plt.legend()
    plt.show()

    return df_results

#--------------------------
# Main
#--------------------------
if __name__ == "__main__":
    # 1) Create or load fixed dataset splits
    base_train, base_val, base_test = create_or_load_base_splits(sample_n=1000)

    # 2) Ensure fakes JSONL exists from your pre-generated text
    gen_fakes = ensure_fakes_jsonl()

    # 3) Run mix ratio experiment
    results_df = run_ratio_experiment(
        base_train=base_train,
        base_val=base_val,
        base_test=base_test,
        gen_fakes=gen_fakes,
        ratios=(0, 0.25, 0.5, 0.75, 1.0),
        epochs=30
    )

    # Optional: save results
    results_df.to_csv("ratio_results.csv", index=False)
    print("Saved results to ratio_results.csv")
