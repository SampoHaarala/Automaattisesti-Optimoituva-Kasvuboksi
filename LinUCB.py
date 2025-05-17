import numpy as np

class LinUCB:
    def __init__(self, n_arms, n_features, alpha=1.0):
        self.n_arms = n_arms
        self.alpha = alpha
        self.A = [np.identity(n_features) for _ in range(n_arms)]
        self.b = [np.zeros((n_features, 1)) for _ in range(n_arms)]

    def select_arm(self, context):
        p = []
        for i in range(self.n_arms):
            A_inv = np.linalg.inv(self.A[i])
            theta = A_inv @ self.b[i]
            context_vec = context.reshape(-1, 1)
            score = (theta.T @ context_vec)[0, 0] + self.alpha * np.sqrt((context_vec.T @ A_inv @ context_vec)[0, 0])
            p.append(score)
        return int(np.argmax(p))

    def update(self, arm, context, reward):
        x = context.reshape(-1, 1)
        self.A[arm] += x @ x.T
        self.b[arm] += reward * x

    def update(self, arm, context, reward):
        x = context.reshape(-1, 1)
        self.A[arm] += x @ x.T
        self.b[arm] += reward * x
