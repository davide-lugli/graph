import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import os

# -----------------------
# CONFIG
# -----------------------
CITY = 'T'
DATA_PATH = 'dataset/preproc/test'
SEQ_LEN = 60 * 48  # 60 days * 48 half-hour slots per day
PRED_LEN = 15 * 48
BATCH_SIZE = 64
EPOCHS = 5
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# -----------------------
# DATASET
# -----------------------
class MovementDataset(Dataset):
    def __init__(self, csv_path, seq_len=SEQ_LEN, mode='train', chunk_size=10_000):
        self.csv_path = csv_path
        self.seq_len = seq_len
        self.mode = mode
        self.chunk_size = chunk_size
        self.uid_offsets = self._index_uids()

    def _index_uids(self):
        """Build an index mapping each user to file offsets for lazy loading."""
        uid_offsets = {}
        offset = 0
        for chunk in pd.read_csv(self.csv_path, chunksize=self.chunk_size):
            for uid in chunk['uid'].unique():
                if uid not in uid_offsets:
                    uid_offsets[uid] = offset
            offset += self.chunk_size
        return list(uid_offsets.keys())

    def __len__(self):
        return len(self.uid_offsets)

    def __getitem__(self, idx):
        uid = self.uid_offsets[idx]
        chunks = pd.read_csv(self.csv_path, chunksize=self.chunk_size)
        user_df = None
        for chunk in chunks:
            if uid in chunk['uid'].values:
                user_df = chunk[chunk['uid'] == uid]
                break
        if user_df is None:
            raise IndexError(f"User ID {uid} not found in dataset.")
        coords = user_df[['x', 'y']].values
        coords = coords.astype(np.float32)

        if self.mode == 'train':
            x_seq = coords[:self.seq_len]
            y_seq = coords[1:self.seq_len+1]
        else:
            x_seq = coords[:self.seq_len]
            y_seq = np.zeros((PRED_LEN, 2), dtype=np.float32)

        return torch.tensor(x_seq), torch.tensor(y_seq)

# -----------------------
# MODEL
# -----------------------
class GRUPredictor(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64, num_layers=1):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 2)

    def forward(self, x):
        out, _ = self.gru(x)
        out = self.fc(out)
        return out

# -----------------------
# TRAINING LOOP
# -----------------------
def train_model(model, dataloader, epochs=EPOCHS):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        total_loss = 0
        for x_seq, y_seq in dataloader:
            x_seq, y_seq = x_seq.to(DEVICE), y_seq.to(DEVICE)
            optimizer.zero_grad()
            output = model(x_seq)
            loss = loss_fn(output[:, :-1, :], y_seq)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")

# -----------------------
# INFERENCE
# -----------------------
@torch.no_grad()
def predict_user(model, user_seq, pred_len=PRED_LEN):
    model.eval()
    seq = torch.tensor(user_seq, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    preds = []

    for _ in range(pred_len):
        out = model(seq)
        next_xy = out[:, -1, :]  # last predicted coordinate
        preds.append(next_xy.cpu().numpy().flatten())
        seq = torch.cat([seq, next_xy.unsqueeze(1)], dim=1)  # feed prediction as input

    return np.array(preds)

def run_inference(model, test_csv_path, output_csv_path):
    preds_list = []
    for chunk in pd.read_csv(test_csv_path, chunksize=10_000):
        for uid in chunk['uid'].unique():
            user_df = chunk[chunk['uid'] == uid]
            coords = user_df[['x', 'y']].values.astype(np.float32)

            # last known 60 days
            context_seq = coords[:SEQ_LEN]

            # if the tail is 999s, predict
            if (coords[-PRED_LEN:, 0] == 999).all():
                pred_coords = predict_user(model, context_seq)
                user_df.loc[user_df['x'] == 999, ['x', 'y']] = pred_coords
                preds_list.append(user_df)
    result = pd.concat(preds_list)
    result.to_csv(output_csv_path, index=False)
    print(f"Predictions saved to {output_csv_path}")

# -----------------------
# MAIN EXECUTION
# -----------------------
if __name__ == "__main__":
    train_path = os.path.join(DATA_PATH, f"city_{CITY}_trainmerged.csv")

    dataset = MovementDataset(train_path, mode='train')
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = GRUPredictor().to(DEVICE)
    train_model(model, dataloader)

    run_inference(model, os.path.join(DATA_PATH, f"city_{CITY}_testmerged.csv"), os.path.join(DATA_PATH, f"city_{CITY}_testpredictions.csv"))
