import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional
import json

def gaussian_nll(mu, sigma, target):
    eps = 1e-6
    variance = sigma ** 2 + eps
    nll = 0.5 * (torch.log(variance) + (target - mu) ** 2 / variance)
    return nll.mean()

class RiskPredictor(nn.Module):

    def __init__(
        self,
        input_dim: int = 12,
        hidden_dim: int = 64,
        dropout: float = 0.1
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.dropout2 = nn.Dropout(dropout)

        self.mu_head = nn.Linear(hidden_dim // 2, 2)
        self.sigma_head = nn.Linear(hidden_dim // 2, 2)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        h = F.relu(self.fc1(x))
        h = self.dropout1(h)
        h = F.relu(self.fc2(h))
        h = self.dropout2(h)

        mu = torch.sigmoid(self.mu_head(h))
        sigma = F.softplus(self.sigma_head(h))

        return mu, sigma

    def predict(
        self,
        features: np.ndarray,
        device: str = 'cpu'
    ) -> Tuple[np.ndarray, np.ndarray]:
        self.eval()
        with torch.no_grad():
            x = torch.from_numpy(features).float().to(device)
            mu, sigma = self.forward(x)
            return mu.cpu().numpy(), sigma.cpu().numpy()

    def save(self, path: str, metadata: Optional[Dict] = None):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        save_dict = {
            'model_state_dict': self.state_dict(),
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'metadata': metadata or {}
        }

        torch.save(save_dict, path)
        print(f"✅ 模型已保存到 {path}")

    @classmethod
    def load(cls, path: str, device: str = 'cpu') -> 'RiskPredictor':
        checkpoint = torch.load(path, map_location=device)

        model = cls(
            input_dim=checkpoint['input_dim'],
            hidden_dim=checkpoint['hidden_dim']
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()

        model._loaded_from = str(path)

        print(f"✅ 模型已加载：{path}")
        if checkpoint['metadata']:
            print(f"   元数据：{checkpoint['metadata']}")

        return model

class RiskPredictorTrainer:

    def __init__(
        self,
        model: RiskPredictor,
        learning_rate: float = 1e-3,
        device: str = 'cpu'
    ):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')

    def train_epoch(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        batch_size: int = 256
    ) -> float:
        self.model.train()

        n_samples = X_train.shape[0]
        indices = np.random.permutation(n_samples)
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            batch_idx = indices[start:end]

            X_batch = torch.from_numpy(X_train[batch_idx]).float().to(self.device)
            y_batch = torch.from_numpy(y_train[batch_idx]).float().to(self.device)

            mu, sigma = self.model(X_batch)

            loss_sla = self._gaussian_nll(mu[:, 0], sigma[:, 0], y_batch[:, 0])
            loss_failure = self._gaussian_nll(mu[:, 1], sigma[:, 1], y_batch[:, 1])
            loss = loss_sla + loss_failure

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        return epoch_loss / n_batches

    def validate(
        self,
        X_val: np.ndarray,
        y_val: np.ndarray
    ) -> Tuple[float, Dict[str, float]]:
        self.model.eval()

        with torch.no_grad():
            X_tensor = torch.from_numpy(X_val).float().to(self.device)
            y_tensor = torch.from_numpy(y_val).float().to(self.device)

            mu, sigma = self.model(X_tensor)

            loss_sla = self._gaussian_nll(mu[:, 0], sigma[:, 0], y_tensor[:, 0])
            loss_failure = self._gaussian_nll(mu[:, 1], sigma[:, 1], y_tensor[:, 1])
            val_loss = (loss_sla + loss_failure).item()

            mse_sla = F.mse_loss(mu[:, 0], y_tensor[:, 0]).item()
            mse_failure = F.mse_loss(mu[:, 1], y_tensor[:, 1]).item()

            mae_sla = F.l1_loss(mu[:, 0], y_tensor[:, 0]).item()
            mae_failure = F.l1_loss(mu[:, 1], y_tensor[:, 1]).item()

        metrics = {
            'mse_sla': mse_sla,
            'mse_failure': mse_failure,
            'mae_sla': mae_sla,
            'mae_failure': mae_failure,
            'mean_sigma_sla': sigma[:, 0].mean().item(),
            'mean_sigma_failure': sigma[:, 1].mean().item()
        }

        return val_loss, metrics

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 100,
        batch_size: int = 256,
        patience: int = 10,
        verbose: bool = True
    ) -> Dict:
        best_model_state = None
        patience_counter = 0

        for epoch in range(epochs):

            train_loss = self.train_epoch(X_train, y_train, batch_size)
            self.train_losses.append(train_loss)

            val_loss, metrics = self.validate(X_val, y_val)
            self.val_losses.append(val_loss)

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                best_model_state = self.model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1

            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}")
                print(f"  Train Loss: {train_loss:.4f}")
                print(f"  Val Loss: {val_loss:.4f}")
                print(f"  MSE (SLA/Failure): {metrics['mse_sla']:.4f} / {metrics['mse_failure']:.4f}")
                print(f"  Mean σ: {metrics['mean_sigma_sla']:.4f} / {metrics['mean_sigma_failure']:.4f}")

            if patience_counter >= patience:
                print(f"✅ 早停触发（epoch {epoch+1}），最佳 val_loss: {self.best_val_loss:.4f}")
                break

        if best_model_state:
            self.model.load_state_dict(best_model_state)

        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss,
            'final_metrics': metrics
        }

    @staticmethod
    def _gaussian_nll(mu, sigma, target):
        return gaussian_nll(mu, sigma, target)
