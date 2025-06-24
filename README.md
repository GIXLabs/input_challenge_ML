# Tetris Deep Reinforcement Learning

**Credit**: RL section is adapted from [tetris-ai](https://github.com/nuno-faria/tetris-ai.git).

## Project Overview

- Train a bot to play [Tetris](https://en.wikipedia.org/wiki/Tetris) using deep reinforcement learning (RL) or imitation learning (IL).
- Play Tetris using trained AI or as a human.
- Collect demonstration data from both human and RL agents to improve imitation learning.


## Project Structure

```
./
├── assets/         # Images, gifs, diagrams
├── models/         # Saved Keras models
├── data/           # Collected demonstration data (human and RL)
├── src/            # Scripts for training and inference
│   ├── run.py              # Main training script for DQN agent
│   ├── run_model.py        # Script to run inference, collect RL demos, or test imitation policy
│   ├── behav_clone.py      # Train a policy network from demonstration data
│   ├── play_human.py       # Play Tetris as a human and collect data
│   ├── play_human_vs_ai.py # Play Tetris: human vs AI
│   ├── tetris.py           # Tetris game logic (used by scripts)
│   ├── dqn_agent.py        # DQN agent implementation
│   └── logs.py             # Custom logging utilities
├── tetris-ai/      # Original code from tetris-ai
├── logs/           # Training and evaluation logs
├── environment.yml # Conda environment file
├── requirements.txt
├── LICENSE
└── README.md
```

## Setup

If you are using M1 Macbooks, you may find information contained in this [repo useful to setup your virtual environment](https://github.com/mrdbourke/m1-machine-learning-test).
After, please follow the steps below to setup your virtual environment. 

1. **Clone the repository**
2. **Create the conda environment:**
   ```sh
   conda env create -f environment.yml
   ```

## Demo

First 10000 points, after some training.

![Demo - First 10000 points](./assets/demo.gif)

## Usage

### 1. **Train the RL agent:**
```sh
python src/run.py
```
(Hyperparameters can be changed in `src/run.py`.)

### 2. **Collect Human Demonstration Data:**
```sh
python src/play_human.py
```
- Each session is saved as a separate file in the `data/` directory (e.g., `human_demo_YYYYMMDD_HHMMSS.npy`).
- Play multiple games to collect more data.

### 3. **Collect RL Demonstration Data:**
```sh
python src/run_model.py models/best.keras
```
- In RL mode, the script will save RL agent demonstrations in `data/` (e.g., `rl_demo_YYYYMMDD_HHMMSS.npy`).
- You can interrupt with Ctrl+C to save partial data.

### 4. **Train an Imitation Policy (Behavioral Cloning):**
```sh
python src/behav_clone.py
```
- This script loads all `human_demo_*.npy` and `rl_demo_*.npy` files from `data/` and trains a policy network.
- The trained policy is saved in `models/policy_bc.keras`.

### 5. **Test a Trained Policy (Imitation or RL):**
```sh
python src/run_model.py models/policy_bc.keras
```
- The script auto-detects the model type and runs in the appropriate mode.

### 6. **View logs with TensorBoard:**
```sh
tensorboard --logdir ./logs
```
Then click on the url of your local host to view the board.

## Environment

Main dependencies (see `environment.yml` for full list):
- python 3.10
- tensorflow
- keras
- numpy
- tqdm
- matplotlib
- scikit-learn
- opencv-python

## State and Action Format

- **State:** `[lines_cleared, holes, total_bumpiness, sum_height]` (4 features from the board)
- **Action:** Integer encoding:
    - 0 = left
    - 1 = right
    - 2 = down
    - 3 = rotate
- RL data is automatically converted to this format for imitation learning.

## How does it work

#### Reinforcement Learning

At first, the agent will play random moves, saving the states and the given reward in a limited queue (replay memory). At the end of each episode (game), the agent will train itself (using a neural network) with a random sample of the replay memory. As more and more games are played, the agent becomes smarter, achieving higher and higher scores.

Since in reinforcement learning once an agent discovers a good 'path' it will stick with it, it was also considered an exploration variable (that decreases over time), so that the agent picks sometimes a random action instead of the one it considers the best. This way, it can discover new 'paths' to achieve higher scores.

#### Imitation Learning (Behavioral Cloning)

Trains a policy network to mimic expert actions (from human or RL agent demonstrations) by learning directly from (state, action) pairs, without using reward signals during training.

- Collect (state, action) pairs from human or RL agent play.
- Train a policy network to predict the action given the state (supervised learning).
- The policy can then be used to play Tetris by itself.

#### Training

The training is based on the [Q Learning algorithm](https://en.wikipedia.org/wiki/Q-learning) for RL, and on supervised learning for imitation.

## Results

For 2000 episodes, with epsilon ending at 1500, the agent kept going for too long around episode 1460, so it had to be terminated. Here is a chart with the maximum score every 50 episodes, until episode 1450:

![results](./assets/results.svg)

Note: Decreasing the `epsilon_end_episode` could make the agent achieve better results in a smaller number of episodes.

## Tips and Troubleshooting

- **Data Quality:** The more diverse and skillful your demonstrations, the better your imitation policy will be.
- **Ctrl+C:** You can safely interrupt data collection or RL runs with Ctrl+C; data will be saved.
- **ESC in RL mode:** Not supported due to OpenCV limitations; use Ctrl+C instead.
- **Action Format:** All actions are converted to integers for training, even if RL data originally used tuples.

## Useful Links

#### Deep Q Learning
- PythonProgramming - https://pythonprogramming.net/q-learning-reinforcement-learning-python-tutorial/
- Keon - https://keon.io/deep-q-learning/
- Towards Data Science - https://towardsdatascience.com/self-learning-ai-agents-part-ii-deep-q-learning-b5ac60c3f47

#### Tetris
- Code My Road - https://codemyroad.wordpress.com/2013/04/14/tetris-ai-the-near-perfect-player/ (uses evolutionary strategies)
