import numpy as np
import pickle
import gymnasium as gym

H = 200
D = 80 * 80

def preprocess(img):
    img = img[35:195]
    img = img[::2, ::2, 0]
    img[img == 144] = 0
    img[img == 109] = 0
    img[img != 0] = 1
    return img.astype(float).ravel()

class Model(object):
    """Deine trainierte Architektur (muss identisch sein)."""
    def __init__(self):
        self.weights = {}
        self.weights["W1"] = np.random.randn(H, D) / np.sqrt(D)
        self.weights["W2"] = np.random.randn(H) / np.sqrt(H)

    def policy_forward(self, x):
        h = np.dot(self.weights["W1"], x)
        h[h < 0] = 0
        logp = np.dot(self.weights["W2"], h)
        p = 1.0 / (1.0 + np.exp(-logp))
        return p, h

checkpoint_path = "checkpoint.pkl"
with open(checkpoint_path, "rb") as f:
    checkpoint_data = pickle.load(f)

model = Model()
model.weights = checkpoint_data["model_weights"]
print("Geladene Gewichte aus Checkpoint:", checkpoint_path)

env = gym.make("ale_py:ALE/Pong-v5", render_mode="human")

observation, info = env.reset()
prev_x = None
episode_reward = 0

while True:
    cur_x = preprocess(observation)
    x = cur_x - prev_x if prev_x is not None else np.zeros(D)
    prev_x = cur_x

    aprob, _ = model.policy_forward(x)
    action = 2 if np.random.uniform() < aprob else 3

    observation, reward, terminated, truncated, info = env.step(action)
    episode_reward += reward
    
    env.render()

    if terminated or truncated:
        print("Episode beendet mit Reward:", episode_reward)
        break

env.close()