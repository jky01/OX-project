# OX-project

This repository explores training a Transformer-based agent to play
Tic-Tac-Toe by competing against a traditional baseline agent. The
traditional agent picks random moves, while the Transformer learns via
reinforcement learning.

## Project Goals
- Two agents can play Tic-Tac-Toe against each other.
- The Transformer agent improves through repeated games.
- Checkpoints are saved periodically so training can be resumed later.

## Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Train the Transformer agent against the random agent:
   ```bash
   python train.py --episodes 1000 --checkpoint-interval 100
   ```
3. Resume training from a checkpoint if needed:
   ```bash
   python train.py --resume checkpoints/ckpt_100.pth
   ```

Checkpoints are stored in the `checkpoints/` directory.

