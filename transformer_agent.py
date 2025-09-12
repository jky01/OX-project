"""Transformer-based agent for learning Tic-Tac-Toe."""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
from torch.distributions import Categorical

from tictactoe import TicTacToe


class TransformerNet(nn.Module):
    """A small Transformer network mapping board states to move logits."""

    def __init__(self, emb_dim: int = 32, nhead: int = 4, num_layers: int = 2):
        super().__init__()
        self.embedding = nn.Embedding(3, emb_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim, nhead=nhead, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(emb_dim, 1)

    def forward(self, board: torch.Tensor) -> torch.Tensor:
        # board: (batch, 9) of token ids {0,1,2}
        x = self.embedding(board)
        x = self.encoder(x)
        logits = self.head(x).squeeze(-1)
        return logits  # (batch, 9)


class TransformerAgent:
    """Agent that selects moves according to a Transformer policy."""

    def __init__(self, model: TransformerNet | None = None, device: torch.device | None = None):
        self.device = device or torch.device("cpu")
        self.model = model or TransformerNet()
        self.model.to(self.device)

    @staticmethod
    def encode_board(board: List[int]) -> List[int]:
        """Encode board values into token ids for the model."""
        mapping = {0: 0, 1: 1, -1: 2}
        return [mapping[v] for v in board]

    def _policy(self, game: TicTacToe) -> Categorical:
        board = torch.tensor([self.encode_board(game.board)], dtype=torch.long, device=self.device)
        logits = self.model(board)[0]
        mask = torch.full((9,), float("-inf"), device=self.device)
        mask[game.available_moves()] = 0
        probs = torch.softmax(logits + mask, dim=-1)
        return Categorical(probs)

    def select_move(self, game: TicTacToe) -> int:
        """Sample a move without tracking gradients (for evaluation)."""
        with torch.no_grad():
            m = self._policy(game)
            return int(m.sample().item())

    def select_move_training(self, game: TicTacToe):
        """Sample a move and return the log probability for training."""
        m = self._policy(game)
        action = m.sample()
        return int(action.item()), m.log_prob(action)

    def save(self, path: str) -> None:
        torch.save(self.model.state_dict(), path)

    def load(self, path: str) -> None:
        state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state)

