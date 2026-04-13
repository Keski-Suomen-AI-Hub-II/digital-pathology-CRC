import os
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, average_precision_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from xgboost import XGBClassifier
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# 1. DATASET: Load tile-level feature vectors for each WSI
# ============================================================

class WsiTileDataset(Dataset):
    def __init__(self, wsi_list, label_dict, npy_dir, multi):
        self.wsi_list = wsi_list
        self.label_dict = label_dict
        self.npy_dir = npy_dir
        self.multi = multi

    def __len__(self):
        return len(self.wsi_list)

    def __getitem__(self, idx):
        print(self.wsi_list[idx])
        wsi = self.wsi_list[idx]
        if self.multi:
            arr5x = np.load(os.path.join(self.npy_dir, wsi + "_outputs.npy"))  # (tiles, feat_dim)
            arr20x = np.load(os.path.join(self.npy_dir[:-3]+"_20x", wsi + "_outputs.npy"))  # (tiles, feat_dim)
            arr5x = torch.tensor(arr5x, dtype=torch.float32)
            arr20x = torch.tensor(arr20x, dtype=torch.float32)
        
            # arr = np.hstack([arr5x, arr20x])
        
        else:
            arr = np.load(os.path.join(self.npy_dir, wsi + "_outputs.npy"))  # (tiles, feat_dim)
            arr = torch.tensor(arr, dtype=torch.float32)
        
        label = torch.tensor(self.label_dict[wsi], dtype=torch.long)
        
        if self.multi:
            print(wsi)
            return arr5x, arr20x, label, wsi
        else:
            return arr, label, wsi


# ============================================================
# 2. MIL Models
# ============================================================
class AttentionMIL_end(nn.Module):
    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()

        self.fc = nn.Linear(input_dim, hidden_dim)
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        self.classifier = nn.Linear(hidden_dim, 2)  # binary classification

    def forward(self, x):
        h = torch.relu(self.fc(x))       # (tiles, hidden_dim)
        a = self.attention(h)            # (tiles, 1)
        a = torch.softmax(a, dim=0)      # attention weights
        z = torch.sum(a * h, dim=0)      # aggregated WSI vector (hidden_dim)
        x = self.classifier(z.unsqueeze(0))   # pass through classifier
        return x                          # return tensor for end-to-end training   

class DualScaleAttentionMIL(nn.Module):
    def __init__(
        self,
        input_dim_5x,
        input_dim_20x,
        hidden_dim=256,
        n_classes=2
    ):
        super().__init__()

        self.mil_5x  = AttentionMIL(input_dim_5x, hidden_dim)
        self.mil_20x = AttentionMIL(input_dim_20x, hidden_dim)

        # fuse slide embeddings
        self.classifier = nn.Linear(2 * hidden_dim, n_classes)

    def forward(self, x_5x, x_20x):

        z_5x = self.mil_5x(x_5x)
        z_20x = self.mil_20x(x_20x)

        z = torch.cat([z_5x, z_20x], dim=0)
        logits = self.classifier(z.unsqueeze(0))

        return logits

class CLAM_end(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, dropout=0.25):
        """
        Returns:
            M: aggregated bag feature vector (hidden_dim,)
            a: attention weights (tiles,)
        """
        super().__init__()

        # projection layer
        self.project = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # gated attention
        self.att_V = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )
        self.att_U = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )

        # attention weight layer
        self.att_weights = nn.Linear(hidden_dim, 1)

        # classifier
        self.classifier = nn.Linear(hidden_dim, 2)  # binary classification

    def forward(self, x):
        """
        x: (tiles, feat_dim)
        """
        H = self.project(x)               # (tiles, hidden_dim)

        V = self.att_V(H)                 # (tiles, hidden_dim)
        U = self.att_U(H)                 # (tiles, hidden_dim)
        gated = V * U                     # CLAM gated attention

        a = self.att_weights(gated)       # (tiles, 1)
        a = torch.softmax(a, dim=0)       # normalize over tiles

        # aggregated WSI feature vector
        M = torch.sum(a * H, dim=0)       # (hidden_dim,)

        x = self.classifier(M.unsqueeze(0))   # pass through classifier
        return x                          # return tensor for end-to-end training


class DualScaleCLAM(nn.Module):
    def __init__(
        self,
        input_dim_5x,
        input_dim_20x,
        hidden_dim=512,
        n_classes=2,
        dropout=0.25
    ):
        super().__init__()

        self.clam_5x = CLAMEncoder(input_dim_5x, hidden_dim, dropout)
        self.clam_20x = CLAMEncoder(input_dim_20x, hidden_dim, dropout)

        # fuse slide-level embeddings
        self.classifier = nn.Linear(2 * hidden_dim, n_classes)

    def forward(self, x_5x, x_20x):
        """
        x_5x:  (tiles_5x,  input_dim_5x)
        x_20x: (tiles_20x, input_dim_20x)
        """
        M_5x, a_5x   = self.clam_5x(x_5x)
        M_20x, a_20x = self.clam_20x(x_20x)

        M = torch.cat([M_5x, M_20x], dim=0)
        logits = self.classifier(M.unsqueeze(0))

        return logits, a_5x, a_20x

def extract_features(dataloader, model, device):
    
    X, y = [], []
    for tiles, label, wsi in dataloader:
        tiles = tiles.squeeze(0).to(device)       # (tiles, feat_dim)
        agg = model(tiles)                    # (hidden_dim)
        X.append(agg)
        y.append(label.item())
    return np.array(X), np.array(y)

# ============================================================
# 3. MAIN TRAINING SCRIPT
# ============================================================    

def train_end_to_end_mil(npy_dir, labels, model_type, epochs, device, trainval, multi=False):

    # ----------------------------------------
    # Split WSIs 50 / 25 / 25
    # ----------------------------------------
    all_wsis = list(labels.keys())
    
    if trainval==True:
        train_wsis, val_wsis = train_test_split(
            all_wsis,
            test_size=0.30, 
            random_state=30,
            stratify=[labels[w] for w in all_wsis]
        )
        
        print(f"Train: {len(train_wsis)}, Val: {len(val_wsis)}")#, Test: {len(test_wsis)}")

    else:
        train_wsis, temp_wsis = train_test_split(
            all_wsis,
            test_size=0.50, 
            random_state=40, 
            stratify=[labels[w] for w in all_wsis]
        )

        val_wsis, test_wsis = train_test_split(
            temp_wsis,
            test_size=0.50,
            random_state=40,
            stratify=[labels[w] for w in temp_wsis]
        )

        print(f"Train: {len(train_wsis)}, Val: {len(val_wsis)}, Test: {len(test_wsis)}")

    # ----------------------------------------
    # Load example to get feature dimension
    # ----------------------------------------
        
    example_arr = np.load(os.path.join(npy_dir, train_wsis[0] + "_outputs.npy"))
    feat_dim = example_arr.shape[1]
    
    if model_type in ('attention', 'abmil'):
        mil_model = AttentionMIL_end(input_dim=feat_dim).to(device)
    if model_type == 'dual_attention':
        mil_model = DualScaleAttentionMIL(
            input_dim_5x=feat_dim,
            input_dim_20x=feat_dim
        ).to(device)
    if model_type == 'dual_clam':
        mil_model = DualScaleCLAM(
            input_dim_5x=feat_dim,
            input_dim_20x=feat_dim
        ).to(device)
    elif model_type == 'clam':
        mil_model = CLAM_end(input_dim=feat_dim).to(device)

    train_ds = WsiTileDataset(train_wsis, labels, npy_dir, multi)
    val_ds   = WsiTileDataset(val_wsis,   labels, npy_dir, multi)
    
    if trainval==False:
        test_ds  = WsiTileDataset(test_wsis,  labels, npy_dir, multi)

    train_loader = DataLoader(train_ds, batch_size=1, shuffle=False)
    val_loader   = DataLoader(val_ds,   batch_size=1, shuffle=False)
    
    if trainval==False:
        test_loader  = DataLoader(test_ds,  batch_size=1, shuffle=False)

    mil_model = mil_model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(mil_model.parameters(), lr=1e-4)

    epochs = 100
    patience = 8            # early stopping patience
    best_val_loss = np.inf
    epochs_no_improve = 0

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    # ----------------------------------------
    # Training loop
    # ----------------------------------------
    for epoch in range(epochs):
        mil_model.train()

        train_loss_sum = torch.zeros(1, device=device)
        train_correct  = torch.zeros(1, device=device)
        train_total    = torch.zeros(1, device=device)

        for batch in train_loader:
            if multi:
                tiles_5x, tiles_20x, label, _ = batch
                tiles_5x  = tiles_5x[0].to(device, non_blocking=True)
                tiles_20x = tiles_20x[0].to(device, non_blocking=True)
            else:
                tiles, label, _ = batch
                tiles = tiles[0].to(device, non_blocking=True)

            label = label.view(-1).long().to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            logits = mil_model(tiles_5x, tiles_20x)[0] if multi else mil_model(tiles)
            loss   = criterion(logits, label)

            loss.backward()
            optimizer.step()

            train_loss_sum += loss.detach()
            train_correct  += (torch.argmax(logits, dim=1) == label).sum()
            train_total    += 1

        avg_train_loss = (train_loss_sum / train_total).item()
        train_acc      = (train_correct  / train_total).item()
        train_losses.append(avg_train_loss)
        train_accs.append(train_acc)

        # -------- VALIDATION --------
        mil_model.eval()
        val_loss = 0.0
        correct  = 0
        total    = 0
        val_probs_gpu, val_preds_gpu, val_labels_gpu = [], [], []

        with torch.no_grad():
            for batch in val_loader:
                if multi:
                    tiles_5x, tiles_20x, label, _ = batch
                    tiles_5x  = tiles_5x[0].to(device, non_blocking=True)
                    tiles_20x = tiles_20x[0].to(device, non_blocking=True)
                else:
                    tiles, label, _ = batch
                    tiles = tiles[0].to(device, non_blocking=True)

                label  = label.view(-1).long().to(device, non_blocking=True)
                logits = mil_model(tiles_5x, tiles_20x)[0] if multi else mil_model(tiles)
                loss   = criterion(logits, label)

                val_loss += loss.item()
                probs     = torch.softmax(logits, dim=1)
                pred      = torch.argmax(probs, dim=1)

                val_probs_gpu.append(probs.detach())
                val_preds_gpu.append(pred.detach())
                val_labels_gpu.append(label.detach())

                correct += (pred == label).sum().item()
                total   += 1

        val_probs  = torch.cat(val_probs_gpu,  dim=0).cpu().numpy()
        val_preds  = torch.cat(val_preds_gpu,  dim=0).cpu().numpy()
        val_labels = torch.cat(val_labels_gpu, dim=0).cpu().numpy()

        avg_val_loss = val_loss / len(val_loader)
        val_acc      = correct / total
        val_losses.append(avg_val_loss)
        val_accs.append(val_acc)

        print(f"Epoch [{epoch+1}/{epochs}] "
            f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.4f} "
            f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f}")

        # -------- EARLY STOPPING --------
        save_name = "model_name.pt"
        if avg_val_loss < best_val_loss:
            best_val_loss  = avg_val_loss
            epochs_no_improve = 0
            torch.save(mil_model.state_dict(), save_name)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    # ----------------------------------------
    # Plot curves
    # ----------------------------------------

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label="Train Acc")
    plt.plot(val_accs, label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.show()
        
    return mil_model, val_preds, val_probs, val_labels

def test_model(npy_dir, labels, mil_model, device, save_dir, multi=False):
    
    all_wsis = list(labels.keys())
    test_ds  = WsiTileDataset(all_wsis, labels, npy_dir, multi)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)
    batch = next(iter(test_loader))
    
    print(len(all_wsis))
    example_arr = np.load(os.path.join(npy_dir, all_wsis[0] + "_outputs.npy"))

    feat_dim = example_arr.shape[1]

    # -------- TEST --------
    mil_model.to(device)
    mil_model.eval()

    test_correct  = torch.zeros(1, device=device)
    test_total    = torch.zeros(1, device=device)

    test_logits_gpu = []
    test_probs_gpu  = []
    test_preds_gpu  = []
    test_labels_gpu = []
    
    if multi:
        with torch.no_grad():
            for tiles_5x, tiles_20x, label, wsi_id in test_loader:
                tiles_5x = tiles_5x[0].to(device, non_blocking=True)
                tiles_20x = tiles_20x[0].to(device, non_blocking=True)
                label = label.view(-1).long().to(device, non_blocking=True)
                logits = mil_model(tiles_5x, tiles_20x)
                probs = torch.softmax(logits, dim=1)
                pred = torch.argmax(probs, dim=1)

                # keep on GPU
                test_probs_gpu.append(probs.detach())
                test_preds_gpu.append(pred.detach())
                test_labels_gpu.append(label.detach())
                test_total += 1

    else:

        with torch.no_grad():
            for tiles, label, wsi_id in test_loader:
                tiles = tiles[0].to(device, non_blocking=True)      # (num_tiles, feat_dim)
                label = label.view(-1).long().to(device, non_blocking=True)
                outputs = mil_model(tiles)                          # (1, C)

                probs = torch.softmax(outputs, dim=1)
                pred  = torch.argmax(probs, dim=1)

              
                test_probs_gpu.append(probs.detach())
                test_preds_gpu.append(pred.detach())
                test_labels_gpu.append(label.detach())

                test_total += 1

    # ---- move to CPU ONCE ----
    test_acc = (test_correct / test_total).item()


    # test_logits = torch.cat(test_logits_gpu, dim=0).cpu().numpy()
    test_probs = torch.cat(test_probs_gpu, dim=0)[:, 1].cpu().numpy()
    test_preds  = torch.cat(test_preds_gpu, dim=0).cpu().numpy()
    test_labels = torch.cat(test_labels_gpu, dim=0).cpu().numpy()

    print(test_probs.shape)
    print(test_preds.shape)
    print(test_labels.shape)

    print("\n===== TEST RESULTS =====")
    print("Test ACC:", accuracy_score(test_labels, test_preds))
    # print("Test AUC:", roc_auc_score(test_labels, test_probs))
    print("Test F1:", f1_score(test_labels, test_preds, average='weighted'))
    
    # return test_labels, test_logits, test_probs, test_preds
    return test_labels, test_probs, test_preds