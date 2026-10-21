import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional
import json

class RiskPredictorBCE(nn.Module):

    def __init__(
        self,
        input_dim: int = 13,
        hidden_dim: int = 64,
        dropout: float = 0.1
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2),
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.net(x)
        return torch.sigmoid(logits)

    def predict(
        self,
        features: np.ndarray,
        device: str = 'cpu'
    ) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            x = torch.from_numpy(features).float().to(device)
            p = self.forward(x)
            return p.cpu().numpy()

    def save(self, path: str, metadata: Optional[Dict] = None):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        save_dict = {
            'model_state_dict': self.state_dict(),
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'predictor_type': 'bce',
            'metadata': metadata or {}
        }

        torch.save(save_dict, path)
        print(f"[INFO] BCE 预测器已保存到: {path}")

    @classmethod
    def load(cls, path: str, device: str = 'cpu') -> 'RiskPredictorBCE':
        checkpoint = torch.load(path, map_location=device)

        input_dim = checkpoint.get('input_dim', 13)
        hidden_dim = checkpoint.get('hidden_dim', 64)

        model = cls(
            input_dim=input_dim,
            hidden_dim=hidden_dim
        )

        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()

        model._loaded_from = Path(path).resolve()
        model._predictor_type = checkpoint.get('predictor_type', 'bce')

        print(f"[INFO] BCE 预测器已加载: {path}")
        print(f"      - input_dim: {input_dim}, hidden_dim: {hidden_dim}")

        return model
