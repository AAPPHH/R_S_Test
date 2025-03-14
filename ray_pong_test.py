import numpy as np
import os
import ray
import time
import pickle
import gymnasium as gym

log_file = open("/home/john/R_S_Test/logs.txt", "a")
def log_message(message):
    print(message)  # Optional: Ausgabe auch in die Konsole, falls gewünscht
    log_file.write(message + "\n")
    log_file.flush()  # Sicherstellen, dass die Ausgabe sofort geschrieben wird

log_message("Starte Ray Pong Test")
print("Starte Ray Pong Test")

H = 200  # The number of hidden layer neurons.
gamma = 0.99  # The discount factor for reward.
decay_rate = 0.99  # The decay factor for RMSProp leaky sum of grad^2.
D = 80 * 80  # The input dimensionality: 80x80 grid.
learning_rate = 1e-4  # Magnitude of the update.

def preprocess(img):
    # Crop the image.
    img = img[35:195]
    # Downsample by factor of 2.
    img = img[::2, ::2, 0]
    # Erase background (background type 1).
    img[img == 144] = 0
    # Erase background (background type 2).
    img[img == 109] = 0
    # Set everything else (paddles, ball) to 1.
    img[img != 0] = 1
    return img.astype(float).ravel()


def process_rewards(r):
    """Compute discounted reward from a vector of rewards."""
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        # Reset the sum, since this was a game boundary (pong specific!).
        if r[t] != 0:
            running_add = 0
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


def rollout(model, env):
    """Evaluates  env and model until the env returns "Terminated" or "Truncated".

    Returns:
        xs: A list of observations
        hs: A list of model hidden states per observation
        dlogps: A list of gradients
        drs: A list of rewards.

    """
    # Reset the game.
    observation, info = env.reset()
    # Note that prev_x is used in computing the difference frame.
    prev_x = None
    xs, hs, dlogps, drs = [], [], [], []
    terminated = truncated = False
    while not terminated and not truncated:
        cur_x = preprocess(observation)
        x = cur_x - prev_x if prev_x is not None else np.zeros(D)
        prev_x = cur_x

        aprob, h = model.policy_forward(x)
        # Sample an action.
        action = 2 if np.random.uniform() < aprob else 3

        # The observation.
        xs.append(x)
        # The hidden state.
        hs.append(h)
        y = 1 if action == 2 else 0  # A "fake label".
        # The gradient that encourages the action that was taken to be
        # taken (see http://cs231n.github.io/neural-networks-2/#losses if
        # confused).
        dlogps.append(y - aprob)

        observation, reward, terminated, truncated, info = env.step(action)

        # Record reward (has to be done after we call step() to get reward
        # for previous action).
        drs.append(reward)
    return xs, hs, dlogps, drs

class Model(object):
    """This class holds the neural network weights."""

    def __init__(self):
        self.weights = {}
        self.weights["W1"] = np.random.randn(H, D) / np.sqrt(D)
        self.weights["W2"] = np.random.randn(H) / np.sqrt(H)

    def policy_forward(self, x):
        h = np.dot(self.weights["W1"], x)
        h[h < 0] = 0  # ReLU nonlinearity.
        logp = np.dot(self.weights["W2"], h)
        # Softmax
        p = 1.0 / (1.0 + np.exp(-logp))
        # Return probability of taking action 2, and hidden state.
        return p, h

    def policy_backward(self, eph, epx, epdlogp):
        """Backward pass to calculate gradients.

        Arguments:
            eph: Array of intermediate hidden states.
            epx: Array of experiences (observations).
            epdlogp: Array of logps (output of last layer before softmax).

        """
        dW2 = np.dot(eph.T, epdlogp).ravel()
        dh = np.outer(epdlogp, self.weights["W2"])
        # Backprop relu.
        dh[eph <= 0] = 0
        dW1 = np.dot(dh.T, epx)
        return {"W1": dW1, "W2": dW2}

    def update(self, grad_buffer, rmsprop_cache, lr, decay):
        """Applies the gradients to the model parameters with RMSProp."""
        for k, v in self.weights.items():
            g = grad_buffer[k]
            rmsprop_cache[k] = decay * rmsprop_cache[k] + (1 - decay) * g ** 2
            self.weights[k] += lr * g / (np.sqrt(rmsprop_cache[k]) + 1e-5)


def zero_grads(grad_buffer):
    """Reset the batch gradient buffer."""
    for k, v in grad_buffer.items():
        grad_buffer[k] = np.zeros_like(v)

# This forces OpenMP to use 1 single thread, which is needed to 
# prevent contention between multiple actors. 
# See https://docs.ray.io/en/latest/ray-core/configure.html for 
# more details. 
os.environ["OMP_NUM_THREADS"] = "1"
# Tell numpy to only use one core. If we don't do this, each actor may
# try to use all of the cores and the resulting contention may result
# in no speedup over the serial version. Note that if numpy is using
# OpenBLAS, then you need to set OPENBLAS_NUM_THREADS=1, and you
# probably need to do it from the command line (so it happens before
# numpy is imported).
os.environ["MKL_NUM_THREADS"] = "1"

try:
    ray_address = os.environ.get("RAY_ADDRESS", "auto")
    print(f"Connecting to Ray at address: {ray_address}")

    # Initialize Ray with the explicit address
    ray.init(address=ray_address, ignore_reinit_error=True)
    print("Cluster resources:", ray.cluster_resources())
except ConnectionError:
    ray.init()



@ray.remote
class RolloutWorker(object):
    def __init__(self):
        self.env = gym.make("ale_py:ALE/Pong-v5")

    def compute_gradient(self, model):
        # Compute a simulation episode.
        xs, hs, dlogps, drs = rollout(model, self.env)
        reward_sum = sum(drs)
        # Vectorize the arrays.
        epx = np.vstack(xs)
        eph = np.vstack(hs)
        epdlogp = np.vstack(dlogps)
        epr = np.vstack(drs)

        # Compute the discounted reward backward through time.
        discounted_epr = process_rewards(epr)
        # Standardize the rewards to be unit normal (helps control the gradient
        # estimator variance).
        discounted_epr -= np.mean(discounted_epr)
        discounted_epr /= np.std(discounted_epr)
        # Modulate the gradient with advantage (the policy gradient magic
        # happens right here).
        epdlogp *= discounted_epr
        return model.policy_backward(eph, epx, epdlogp), reward_sum

model = Model()

checkpoint_path = "checkpoint.pkl"

if os.path.exists(checkpoint_path):
    with open(checkpoint_path, "rb") as f:
        checkpoint_data = pickle.load(f)
    start_iter = checkpoint_data["iteration"] + 1
    model.weights = checkpoint_data["model_weights"]
    rmsprop_cache = checkpoint_data["rmsprop_cache"]
    running_reward = checkpoint_data["running_reward"]
    log_message(f"Checkpoint loaded. Resuming training at iteration {start_iter}.")
    print(f"Checkpoint geladen. Training startet wieder bei Iteration {start_iter}.")
else:
    start_iter = 1
    running_reward = None
    rmsprop_cache = {k: np.zeros_like(v) for k, v in model.weights.items()}
    log_message("No checkpoint found, starting training at iteration 1.")
    print("Kein Checkpoint vorhanden, starte Training bei Iteration 1.")

grad_buffer = {k: np.zeros_like(v) for k, v in model.weights.items()}

iterations_to_run = 1000
batch_size = 128
actors = [RolloutWorker.remote() for _ in range(batch_size)]

for i in range(start_iter, start_iter + iterations_to_run):
    model_id = ray.put(model)
    gradient_ids = [actor.compute_gradient.remote(model_id) for actor in actors]

    start_time = time.time()
    for _ in range(batch_size):
        [grad_id], gradient_ids = ray.wait(gradient_ids)
        grad, reward_sum = ray.get(grad_id)
        for k in model.weights:
            grad_buffer[k] += grad[k]
        running_reward = (
            reward_sum
            if running_reward is None
            else running_reward * 0.99 + reward_sum * 0.01
        )

    end_time = time.time()
    log_message(
        f"Batch {i} computed {batch_size} rollouts in {end_time - start_time:.2f}s, "
        f"running mean is {running_reward}"
    )
    print(
        f"Batch {i} computed {batch_size} rollouts in {end_time - start_time:.2f}s, "
        f"running mean is {running_reward}"
    )

    model.update(grad_buffer, rmsprop_cache, learning_rate, decay_rate)
    zero_grads(grad_buffer)

    if i % 100 == 0:
        checkpoint_data = {
            "iteration": i,
            "model_weights": model.weights,
            "rmsprop_cache": rmsprop_cache,
            "running_reward": running_reward
        }
        with open(checkpoint_path, "wb") as f:
            pickle.dump(checkpoint_data, f)
        log_message(f"Checkpoint saved at iteration {i}.")
        print(f"Checkpoint bei Iteration {i} gespeichert.")