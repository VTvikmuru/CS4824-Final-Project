import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import subprocess

# Stable seed
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Load dataset
dataset = load_dataset("HiTZ/This-is-not-a-dataset")
df = dataset["train"].to_pandas().sample(1000, random_state=SEED)

sentences = df["sentence"].tolist()
labels = df["label"].astype(int).tolist()

train_texts, test_texts, train_labels, test_labels = train_test_split(sentences, labels, test_size=0.2, random_state=SEED)

#------------------------------------------------------------------------------------------------------

# Ollama generator wrapper
def run_ollama(model, prompt):
    """Run an Ollama model with a given prompt and return output."""
    result = subprocess.run(
        ["ollama", "run", model],
        input=prompt.encode(),
        capture_output=True
    )
    return result.stdout.decode().strip()

def generate_fake_samples(num_samples=50):
    fakes = []
    for i in range(num_samples):
        fake = run_ollama("generator", "Write a fake news style sentence.")
        fakes.append(fake)
        
        # Progress printout every sample
        print(f"Generated {i+1}/{num_samples} fake samples")
    
    return fakes

def load_generated_samples(filename="generated_fakes.txt"):
    with open(filename, "r", encoding="utf-8") as f:
        samples = [line.strip() for line in f if line.strip()]
    return samples

#------------------------------------------------------------------------------------------------------

# Simple NN Detector
class SimpleNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 2)
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

#------------------------------------------------------------------------------------------------------

# Experiment function
def run_experiment(mix_ratios=[0,0.25,0.5,0.75,1.0], fake_samples=10):
    results = []
    vectorizer = TfidfVectorizer(max_features=5000)

    for mix in mix_ratios:
        print(f"\n=== Mix Ratio {mix*100:.0f}% Generator ===")

        # Only generate fakes if mix > 0
        fakes = []
        if mix > 0:
            fakes = generate_fake_samples(fake_samples)
            fake_labels = [0]*len(fakes)

            # load_generated_samples("generated_fakes.txt")
            # fake_labels = [0]*len(fakes)
        else:
            fake_labels = []

        num_fake = int(len(train_texts)*mix)
        mixed_sentences = train_texts + (fakes[:num_fake] if mix > 0 else [])
        mixed_labels = train_labels + (fake_labels[:num_fake] if mix > 0 else [])


        # Vectorize
        X_train = vectorizer.fit_transform(mixed_sentences)
        X_test = vectorizer.transform(test_texts)

        # Baseline Logistic Regression
        baseline = LogisticRegression(max_iter=1000)
        baseline.fit(X_train, mixed_labels)
        baseline_preds = baseline.predict(X_test)
        baseline_acc = accuracy_score(test_labels, baseline_preds)

        print("Baseline Accuracy:", baseline_acc)

        # Neural Network Detector
        input_dim = X_train.shape[1]
        model = SimpleNN(input_dim)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        X_train_tensor = torch.tensor(X_train.toarray(), dtype=torch.float32)
        y_train_tensor = torch.tensor(mixed_labels, dtype=torch.long)
        X_test_tensor = torch.tensor(X_test.toarray(), dtype=torch.float32)
        y_test_tensor = torch.tensor(test_labels, dtype=torch.long)

        EPOCHS = 3
        for epoch in range(EPOCHS):
            optimizer.zero_grad()
            outputs = model(X_train_tensor)
            loss = criterion(outputs, y_train_tensor)
            loss.backward()
            optimizer.step()
            print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {loss.item():.4f}")

        with torch.no_grad():
            test_outputs = model(X_test_tensor)
            preds = torch.argmax(test_outputs, axis=1).numpy()

        nn_acc = accuracy_score(test_labels, preds)
        print("NN Accuracy:", nn_acc)

        results.append({"mix": mix, "baseline_acc": baseline_acc, "nn_acc": nn_acc})

    return pd.DataFrame(results)

#------------------------------------------------------------------------------------------------------

# Run
results_df = run_experiment()

print("\n=== Final Results ===")
print(results_df)

#------------------------------------------------------------------------------------------------------

# Plot
plt.figure(figsize=(8,6))
plt.plot(results_df["mix"], results_df["baseline_acc"], marker="o", label="Logistic Regression")
plt.plot(results_df["mix"], results_df["nn_acc"], marker="s", label="Neural Network")
plt.xlabel("Generator Mix Ratio")
plt.ylabel("Accuracy")
plt.title("Detector Performance vs Generator Mix Ratio")
plt.legend()
plt.grid(True)
plt.show()
