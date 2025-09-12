"""Train the Transformer agent by playing against the traditional agent."""

from __future__ import annotations

import argparse
import os
from typing import List

import torch
import torch.optim as optim

from tictactoe import TicTacToe
from traditional_agent import RandomAgent
from transformer_agent import TransformerAgent


def play_episode(agent: TransformerAgent, opponent: RandomAgent, optimizer: optim.Optimizer | None = None) -> int:
    """Play a single game and update the agent if an optimizer is provided.

    Returns the game outcome: 1 for win, -1 for loss, 0 for draw.
    """
    game = TicTacToe()
    log_probs: List[torch.Tensor] = []

    while True:
        move, log_prob = agent.select_move_training(game)
        log_probs.append(log_prob)
        game.make_move(move)
        result = game.check_winner()
        if result is not None:
            reward = 1 if result == 1 else -1 if result == -1 else 0
            if optimizer is not None:
                loss = -torch.stack(log_probs).sum() * reward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            return result

        # Opponent move
        move = opponent.select_move(game)
        game.make_move(move)
        result = game.check_winner()
        if result is not None:
            reward = 1 if result == 1 else -1 if result == -1 else 0
            if optimizer is not None:
                loss = -torch.stack(log_probs).sum() * reward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=1000, help="Number of games to play")
    parser.add_argument("--checkpoint-interval", type=int, default=100, help="Save model every N episodes")
    parser.add_argument("--checkpoint-dir", default="checkpoints", help="Directory to store checkpoints")
    parser.add_argument("--resume", type=str, default=None, help="Path to a checkpoint to resume from")
    args = parser.parse_args()

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    agent = TransformerAgent()
    if args.resume:
        agent.load(args.resume)
    opponent = RandomAgent()
    optimizer = optim.Adam(agent.model.parameters(), lr=1e-3)

    for episode in range(1, args.episodes + 1):
        play_episode(agent, opponent, optimizer)
        if episode % args.checkpoint_interval == 0:
            path = os.path.join(args.checkpoint_dir, f"ckpt_{episode}.pth")
            agent.save(path)
            print(f"Saved checkpoint to {path}")


if __name__ == "__main__":
    main()

