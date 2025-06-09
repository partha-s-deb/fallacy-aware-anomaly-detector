import os
import numpy as np

# Set path to your ECG5000 dataset
path = "C:/Users/KIIT0001/Documents/Anomaly_ditection/ECG5000"
train = np.loadtxt(os.path.join(path, "ECG5000_TRAIN.txt"))
test = np.loadtxt(os.path.join(path, "ECG5000_TEST.txt"))
data = np.vstack([train, test])
X = data[:, 1:]
y = (data[:, 0] != 1).astype(int)  # 0 = normal, 1 = anomaly

# Fallacy-Aware Model
class FallacyAwareAnomalyDetector:
    def __init__(self, input_dim, Fmax=5, alpha=0.01, gamma=0.9, epsilon=0.1):
        self.weights = np.random.randn(input_dim)
        self.F = 0
        self.Fmax = Fmax
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.consecutive_fp = 0
        self.q_table = {}

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def predict(self, x):
        prob = self.sigmoid(np.dot(x, self.weights))
        Pfailure = min(self.F * 0.1, 0.9)
        rand = np.random.rand()
        alpha_fp = 0.2 * (1 - self.F / self.Fmax)

        if rand < Pfailure:
            return 0  # safe default
        elif rand < self.epsilon:
            prelu_out = np.where(prob > 0, prob, alpha_fp * prob)
            return int(prelu_out > 0.5)
        else:
            return int(prob > 0.5)

    def update(self, x, y_true):
        prob = self.sigmoid(np.dot(x, self.weights))
        y_pred = int(prob > 0.5)
        reward = 1 if y_pred == y_true else -1

        # Update Q-table
        key = (y_true, y_pred)
        self.q_table[key] = self.q_table.get(key, 0) + reward

        # Update weights
        self.weights += self.alpha * (y_true - prob) * x

        # Fallacy logic
        if reward > 0:
            self.F = min(self.F + 1, self.Fmax)
            self.consecutive_fp += 1
            if self.consecutive_fp >= 3:
                self.F = max(0, self.F - 2)
                self.consecutive_fp = 0
        else:
            self.F = max(0, self.F - 1)
            self.consecutive_fp = 0

        return reward, y_pred

# Initialize model
model = FallacyAwareAnomalyDetector(input_dim=X.shape[1])

# Train
rewards = []
for epoch in range(20):
    indices = np.random.permutation(len(X))
    for i in indices:
        reward, pred = model.update(X[i], y[i])
        rewards.append(reward)
    avg_reward = np.mean(rewards[-len(X):])
    print(f"Epoch {epoch + 1}: Avg reward = {avg_reward:.4f}, FP = {model.F}")

# Evaluate
final_preds = [model.predict(x) for x in X]
final_preds = np.array(final_preds)

# Metrics
accuracy = np.mean(final_preds == y)
total_anomalies = np.sum(y)
detected_anomalies = np.sum((final_preds == 1) & (y == 1))

print("\n--- Final Evaluation ---")
print(f"Accuracy: {accuracy:.4f}")
print(f"Total Anomalies Present: {total_anomalies}")
print(f"Anomalies Detected Correctly: {detected_anomalies}")
print(f"Q-table Entries: {len(model.q_table)}")
