import sys
import os
import numpy as np
import time
import datetime
import cv2

if len(sys.argv) < 2:
    exit("Missing model file")

from dqn_agent import DQNAgent
from tetris import Tetris
import tensorflow as tf

model_path = sys.argv[1]

# Try to auto-detect model type (RL or imitation)
def is_policy_model(model):
    # Load model and check output shape
    model = tf.keras.models.load_model(model_path)
    # If output shape is (None, 4), assume policy (4 actions)
    return model.output_shape[-1] == 4

if is_policy_model(model_path):
    print("Detected imitation (policy) model. Running in imitation mode.")
    model = tf.keras.models.load_model(model_path)
    env = Tetris()
    done = False
    env.render()
    GRAVITY_DELAY = 0.5  # seconds
    last_gravity_time = time.time()
    while not done:
        state = np.array(env._get_board_props(env.board)).reshape(1, -1)
        action_probs = model.predict(state, verbose=0)
        action = np.argmax(action_probs)
        # Map action index to move: 0=left, 1=right, 2=down, 3=rotate
        x, rotation = env.current_pos[0], env.current_rotation
        now = time.time()
        gravity_applied = False
        if (now - last_gravity_time) >= GRAVITY_DELAY and action != 2:
            action = 2  # Force down
            gravity_applied = True
        # Compute intended move
        if action == 0:  # left
            x = max(0, x - 1)
        elif action == 1:  # right
            x = min(Tetris.BOARD_WIDTH - 1, x + 1)
        elif action == 3:  # rotate
            rotation = (rotation + 90) % 360
        # Always use play() to execute the move
        reward, done = env.play(x, rotation, render=True)
        if gravity_applied or action == 2:
            last_gravity_time = now
        env.render()
    print(f'Imitation policy final score: {env.get_game_score()}')
else:
    print("Detected RL (Q-network) model. Running in RL mode.")
    env = Tetris()
    agent = DQNAgent(env.get_state_size(), modelFile=model_path)
    done = False
    rl_data = []
    try:
        while not done:
            next_states = {tuple(v): k for k, v in env.get_next_states().items()}
            best_state = agent.best_state(next_states.keys())
            best_action = next_states[best_state]
            state = env._get_board_props(env.board)
            rl_data.append((state, best_action))
            reward, done = env.play(best_action[0], best_action[1], render=True)
    except KeyboardInterrupt:
        print('Ctrl+C pressed. Exiting RL mode and saving data...')
    finally:
        if rl_data:
            DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
            os.makedirs(DATA_DIR, exist_ok=True)
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'rl_demo_{timestamp}.npy'
            np.save(os.path.join(DATA_DIR, filename), rl_data)
