"""Simple Tic-Tac-Toe environment used for training agents.

The board is represented as a list of nine integers:
    0: empty
    1: player X
   -1: player O

The game alternates moves between players and provides utility
functions to check for winners and available moves.
"""

from typing import List, Optional


class TicTacToe:
    """Tic-Tac-Toe game state."""

    def __init__(self) -> None:
        self.board: List[int] = [0] * 9
        self.player: int = 1  # 1 for X, -1 for O

    def reset(self) -> None:
        """Reset the board to the initial empty state."""
        self.board = [0] * 9
        self.player = 1

    def available_moves(self) -> List[int]:
        """Return a list of indices for empty squares."""
        return [i for i, b in enumerate(self.board) if b == 0]

    def make_move(self, idx: int) -> None:
        """Place the current player's mark on the board and switch turns."""
        self.board[idx] = self.player
        self.player *= -1

    def check_winner(self) -> Optional[int]:
        """Check the board for a winner.

        Returns:
            1 if player X wins,
           -1 if player O wins,
            0 if the game is a draw,
            None if the game is still ongoing.
        """
        wins = [
            (0, 1, 2),
            (3, 4, 5),
            (6, 7, 8),
            (0, 3, 6),
            (1, 4, 7),
            (2, 5, 8),
            (0, 4, 8),
            (2, 4, 6),
        ]
        for a, b, c in wins:
            s = self.board[a] + self.board[b] + self.board[c]
            if s == 3:
                return 1
            if s == -3:
                return -1
        if all(self.board):
            return 0
        return None

