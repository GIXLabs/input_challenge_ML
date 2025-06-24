import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split

# Load all demonstration data from the data directory
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
all_data = []
for fname in os.listdir(DATA_DIR):
    if (fname.startswith('human_demo') or fname.startswith('rl_demo')) and fname.endswith('.npy'):
        all_data.extend(np.load(os.path.join(DATA_DIR, fname), allow_pickle=True))
if not all_data:
    raise RuntimeError('No demonstration data found in data directory.')
data = np.array(all_data)
states = np.array([x[0] for x in data])
raw_actions = [x[1] for x in data]

def tuple_to_action(prev_x, prev_rot, x, rot):
    if x < prev_x:
        return 0  # left
    elif x > prev_x:
        return 1  # right
    elif rot != prev_rot:
        return 3  # rotate
    else:
        return 2  # down

actions = []
prev_x, prev_rot = None, None
for i, act in enumerate(raw_actions):
    if isinstance(act, tuple):
        if prev_x is not None and prev_rot is not None:
            actions.append(tuple_to_action(prev_x, prev_rot, act[0], act[1]))
        else:
            actions.append(2)  # Default to down for the first move
        prev_x, prev_rot = act[0], act[1]
    else:
        actions.append(act)
        prev_x, prev_rot = None, None  # Reset for human data

actions = np.array(actions)

# Convert to numpy arrays
states = np.array(states)
actions = np.array(actions)

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(states, actions, test_size=0.2, random_state=42)

# Build a simple policy network
model = keras.Sequential([
    keras.layers.Input(shape=(states.shape[1],)),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(4, activation='softmax')  # 4 possible actions
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# Save the trained policy
model_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
os.makedirs(model_dir, exist_ok=True)
model.save(os.path.join(model_dir, 'policy_bc.keras'))