import numpy as np
import matplotlib.pyplot as plt

class Multi_Armed_Bandit:
    def __init__(self, n: int, mean_range=(-1, 1), std_dev=1.0):
        self.n = n
        self.means = np.random.uniform(mean_range[0], mean_range[1], n)  # Hidden reward means
        self.std_dev = std_dev  # Fixed standard deviation
    
    def pull(self, arm: int) -> float:
        if 0 <= arm < self.n:
            return np.random.normal(self.means[arm], self.std_dev)
        else:
            raise ValueError("Invalid bandit arm index.")
    
    def best_arm(self) -> int:
        return np.argmax(self.means)
    
    def get_means(self):
        return self.means
    
class Agent_greedy:
    def __init__(self, n: int, epsilon=0.1):
        self.n = n
        self.epsilon = epsilon
        self.values = np.zeros(n)  
        self.action_counts = np.zeros(n) 
    
    def select_action(self) -> int:
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, self.n)
        else:
            return np.argmax(self.values)
    
    def update(self, action: int, reward: float):
        self.action_counts[action] += 1
        n = self.action_counts[action]
        value = self.values[action]
        self.values[action] = ((n - 1) / n) * value + (1 / n) * reward
        
    def get_value(self):
        return np.argmax(self.values)
    
    def reset(self):
        self.values = np.zeros(self.n) 
        self.action_counts = np.zeros(self.n)  

class Agent_UCB:
    def __init__(self, n: int, c = 0.7):
        self.n = n
        self.c = c
        self.values = np.zeros(n) 
        self.action_counts = np.zeros(n) 
        self.total_count = 0
    
    def select_action(self) -> int:
        UCB_value = self.values + self.c * np.sqrt(np.log(self.total_count + 1) / (self.action_counts + 1e-5)) # 1e-5 for not zero division
        return np.argmax(UCB_value)
    
    def update(self, action: int, reward: float):
        self.action_counts[action] += 1
        self.total_count += 1
        n = self.action_counts[action]
        value = self.values[action]
        self.values[action] = ((n - 1) / n) * value + (1 / n) * reward
        
    def get_value(self):
        return np.argmax(self.values)
    
    def reset(self):
        self.values = np.zeros(self.n) 
        self.action_counts = np.zeros(self.n) 
        self.total_count = 0

def plot(data,title):
    for i in range(len(data)):
        x = data[i][0]
        y = data[i][2]
        plt.plot(x, y, marker='o', linestyle='-', color=(i/10, 0.5, 1-i/10), label=i)
        plt.xlabel("timestep")
        plt.ylabel("Reward")
        plt.title(title)
        plt.legend()
    plt.show()


