import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ─────────────────────────────────────────────────────────────────────
# A) Multi-Class EEGNet Definition
#   Adjust kernel sizes/padding if your data shape differs.
# ─────────────────────────────────────────────────────────────────────
class EEGNet(nn.Module):
    """
    Multi-class EEGNet adapted for a shape of (N, 1, 120, 64).
    If your data has a different shape (e.g. 5 channels, 128 samples),
    you must adjust the conv/pool sizes and final in_features.
    """
    def __init__(self, n_classes=5):
        super(EEGNet, self).__init__()

        # Layer 1
        self.conv1 = nn.Conv2d(in_channels=1,
                               out_channels=16,
                               kernel_size=(1, 64),
                               padding=0)
        self.batchnorm1 = nn.BatchNorm2d(16, affine=False)

        # Layer 2
        self.padding1 = nn.ZeroPad2d((16, 17, 0, 1))  # (left, right, top, bottom)
        self.conv2 = nn.Conv2d(in_channels=1,
                               out_channels=4,
                               kernel_size=(2, 32))
        self.batchnorm2 = nn.BatchNorm2d(4, affine=False)
        self.pooling2 = nn.MaxPool2d(kernel_size=2, stride=4)

        # Layer 3
        self.padding2 = nn.ZeroPad2d((2, 1, 4, 3))
        self.conv3 = nn.Conv2d(in_channels=4,
                               out_channels=4,
                               kernel_size=(8, 4))
        self.batchnorm3 = nn.BatchNorm2d(4, affine=False)
        self.pooling3 = nn.MaxPool2d(kernel_size=(2, 4))

        # Fully Connected
        # NOTE: The final flatten dimension (4*2*7=56) depends on the input (1x120x64).
        # Adjust if your data or pooling changes.
        self.fc1 = nn.Linear(in_features=4 * 2 * 7, out_features=n_classes)

    def forward(self, x):
        # x shape: (batch, 1, 120, 64)

        # Layer 1
        x = F.elu(self.conv1(x))       # (batch,16,120,1)
        x = self.batchnorm1(x)
        x = F.dropout(x, 0.25)
        x = x.permute(0, 3, 1, 2)      # (batch,1,16,120)

        # Layer 2
        x = self.padding1(x)          # shape => (batch,1,17,153)
        x = F.elu(self.conv2(x))      # => (batch,4,16,122) approx
        x = self.batchnorm2(x)
        x = F.dropout(x, 0.25)
        x = self.pooling2(x)          # => (batch,4,8,30) approx

        # Layer 3
        x = self.padding2(x)          # => (batch,4,13,35) approx
        x = F.elu(self.conv3(x))      # => (batch,4,6,32) approx
        x = self.batchnorm3(x)
        x = F.dropout(x, 0.25)
        x = self.pooling3(x)          # => (batch,4,2,7)

        # Flatten
        x = x.view(-1, 4 * 2 * 7)      # => (batch,56)
        # Multi-class => raw logits
        x = self.fc1(x)               # => (batch,n_classes)
        return x

# ─────────────────────────────────────────────────────────────────────
# B) Simple Accuracy Evaluation for Multi-Class
#   (Could add precision/recall if needed.)
# ─────────────────────────────────────────────────────────────────────
def evaluate(model, X, y, device):
    model.eval()
    with torch.no_grad():
        inputs = torch.from_numpy(X).to(device).float()
        logits = model(inputs)              # shape (N, n_classes)
        preds = torch.argmax(logits, dim=1) # shape (N,)
    preds_np = preds.cpu().numpy()
    return accuracy_score(y, preds_np)

# ─────────────────────────────────────────────────────────────────────
# C) Data Loading
#   Example loading each CSV from a “processed/” folder:
#   - The first letter determines the label: d=0,i=1,r=2,u=3,l=4
#   - We assume each CSV is shaped (120,64) or you must adapt.
# ─────────────────────────────────────────────────────────────────────
def load_csv_data(processed_root="processed"):
    label_map = {'d': 0, 'i': 1, 'r': 2, 'u': 3, 'l': 4}
    X_list, y_list = [], []

    # Walk through every subdirectory and file under “processed_root”
    for root, dirs, files in os.walk(processed_root):
        for fname in files:
            if not fname.endswith(".csv"):
                continue

            class_code = fname[0].lower()
            if class_code not in label_map:
                continue
            label = label_map[class_code]

            csv_path = os.path.join(root, fname)
            df = pd.read_csv(csv_path)

            # Expect shape (120, 64)? Adjust to your real shape
            if df.shape != (120, 64):
                print(f"[WARNING] {csv_path} has shape {df.shape}, skipping.")
                continue

            arr = df.values.astype("float32")  
            arr = np.expand_dims(arr, axis=0)  # => (1, 120, 64)
            X_list.append(arr)
            y_list.append(label)

    if len(X_list) == 0:
        raise RuntimeError("No valid CSV files found in subfolders.")

    X = np.stack(X_list, axis=0)  
    y = np.array(y_list, dtype=np.int64)

    return X, y


# ─────────────────────────────────────────────────────────────────────
# D) Main Training Script
# ─────────────────────────────────────────────────────────────────────
def main():
    # 1) Choose device (MPS on Apple Silicon or CPU fallback)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("Using device:", device)

    # 2) Load data from CSV
    X, y = load_csv_data(processed_root="processed")
    print("Data loaded:")
    print("X shape =", X.shape, "y shape =", y.shape)  # (N, 1, 120,64), (N,)

    # 3) Split into train/val/test (you can change ratios or do cross-val)
    X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=42, stratify=y_tmp)

    print("Train size =", len(X_train), "Val size =", len(X_val), "Test size =", len(X_test))

    # 4) Instantiate the multi-class EEGNet
    net = EEGNet(n_classes=5).to(device)

    # 5) Loss & Optimizer
    criterion = nn.CrossEntropyLoss()  # multi-class
    optimizer = optim.Adam(net.parameters(), lr=1e-3)

    # 6) Training Loop
    num_epochs = 10
    batch_size = 16

    for epoch in range(num_epochs):
        net.train()
        # Shuffle training data
        perm = np.random.permutation(len(X_train))
        X_train, y_train = X_train[perm], y_train[perm]

        running_loss = 0.0
        num_batches = len(X_train) // batch_size

        for i in range(num_batches):
            start = i * batch_size
            end   = start + batch_size

            inputs = torch.from_numpy(X_train[start:end]).to(device).float()
            labels = torch.from_numpy(y_train[start:end]).to(device).long()

            optimizer.zero_grad()
            outputs = net(inputs)            # shape (batch, 5)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Evaluate at end of epoch
        train_acc = evaluate(net, X_train, y_train, device)
        val_acc   = evaluate(net, X_val,   y_val,   device)
        test_acc  = evaluate(net, X_test,  y_test,  device)

        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(f"Loss: {running_loss:.4f}")
        print(f"Train Accuracy: {train_acc:.4f}")
        print(f"Val Accuracy:   {val_acc:.4f}")
        print(f"Test Accuracy:  {test_acc:.4f}")

    print("\nTraining complete.")

if __name__ == "__main__":
    main()