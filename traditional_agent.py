"""Traditional baseline agent for Tic-Tac-Toe.

This agent selects moves randomly from the available positions.
It serves as an opponent for training the Transformer-based agent.
"""

import random
from tictactoe import TicTacToe


class RandomAgent:
    """Selects a random move from the available options."""

    def select_move(self, game: TicTacToe) -> int:
        return random.choice(game.available_moves())

