import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Stable seed
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Load dataset
dataset = load_dataset("HiTZ/This-is-not-a-dataset")
df = dataset["train"].to_pandas().sample(1000, random_state=SEED)  # sample for speed

sentences = df["sentence"].tolist()
labels = df["label"].astype(int).tolist()

train_texts, test_texts, train_labels, test_labels = train_test_split(
    sentences, labels, test_size=0.2, random_state=SEED
)

#------------------------------------------------------------------------------------------------------

# Vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X_train = vectorizer.fit_transform(train_texts)
X_test = vectorizer.transform(test_texts)

# NN
class ImprovedNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, dropout=0.3):
        super(ImprovedNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim//2)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(hidden_dim//2, 2)
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.drop1(x)
        x = self.relu2(self.fc2(x))
        x = self.drop2(x)
        return self.fc3(x)

input_dim = X_train.shape[1]
model = ImprovedNN(input_dim)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-4)

#------------------------------------------------------------------------------------------------------

# Mini-batch training
from torch.utils.data import TensorDataset, DataLoader
X_train_tensor = torch.tensor(X_train.toarray(), dtype=torch.float32)
y_train_tensor = torch.tensor(train_labels, dtype=torch.long)
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

X_test_tensor = torch.tensor(X_test.toarray(), dtype=torch.float32)
y_test_tensor = torch.tensor(test_labels, dtype=torch.long)

EPOCHS = 3000
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        outputs = model(xb)
        loss = criterion(outputs, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
    avg_loss = total_loss / len(train_dataset)
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.4f}")

#------------------------------------------------------------------------------------------------------

# Evaluation
model.eval()
with torch.no_grad():
    test_outputs = model(X_test_tensor)
    preds = torch.argmax(test_outputs, axis=1).numpy()

print("\n=== Improved Neural Network Detector ===")
print(classification_report(test_labels, preds))
cm = confusion_matrix(test_labels, preds)
print("Confusion Matrix:\n", cm)

#------------------------------------------------------------------------------------------------------

# Visualization
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["FAKE","TRUE"], yticklabels=["FAKE","TRUE"])
plt.title("Confusion Matrix - Improved NN Detector")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
