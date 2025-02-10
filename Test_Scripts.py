from Multi_Armed_bandit import *

bandit_amount = 5
epsilon = [0.1,0.01,1.0]
n_time = 100000
c_value = [0.7,0.2,1.0]
Set1 = Multi_Armed_Bandit(bandit_amount)
greedy_Agent = Agent_greedy(bandit_amount)
UCB_Agent = Agent_UCB(bandit_amount)


# total_reward = np.zeros(bandit_amount)
total_reward = []
bandit_rewards = np.zeros((bandit_amount, n_time))
data = np.empty(n_time)
data_action = np.empty(n_time)
cumulative_avg = []
cumulative_reward_avg = []

for i in range(len(epsilon)):
    greedy_Agent.epsilon = epsilon[i]
    for t in range(n_time):
        action = greedy_Agent.select_action()
        reward = Set1.pull(action)
        greedy_Agent.update(action,reward)
        # total_reward[action] += reward
        bandit_rewards[action, t] = reward
        data[t] = reward
        data_action[t] = action
    total_reward.append(bandit_rewards)
    cumulative_avg.append(np.cumsum(data_action)/ (np.arange(n_time) + 1))
    cumulative_reward_avg.append(np.cumsum(data)/ (np.arange(n_time) + 1))
    greedy_Agent.reset()
    data = np.empty(n_time)
    data_action = np.empty(n_time)
    bandit_rewards = np.zeros((bandit_amount, n_time))

for i in range(len(c_value)):
    UCB_Agent.c = c_value[i]
    for t in range(n_time):
        action = UCB_Agent.select_action()
        reward = Set1.pull(action)
        UCB_Agent.update(action,reward)
        # total_reward[action] += reward
        bandit_rewards[action, t] = reward
        data[t] = reward
        data_action[t] = action
    total_reward.append(bandit_rewards)
    cumulative_avg.append(np.cumsum(data_action)/ (np.arange(n_time) + 1))
    cumulative_reward_avg.append(np.cumsum(data)/ (np.arange(n_time) + 1))
    UCB_Agent.reset()
    data = np.empty(n_time)
    data_action = np.empty(n_time)
    bandit_rewards = np.zeros((bandit_amount, n_time))
    
print(total_reward)
# print(f"Value : {greedy_Agent.get_value()}")
print(f"Value : {UCB_Agent.get_value()}")
for i in range(len(Set1.get_means())):
    print(f"Prop{i} = {Set1.get_means()[i]}")
    
final_rewards = np.zeros((len(total_reward), bandit_amount))

for i in range(len(total_reward)): 
    final_rewards[i] = np.sum(total_reward[i], axis=1) 
    

# axes[0].bar(np.arange(bandit_amount),total_reward[0])
plt.figure(figsize=(15, 7))
for i in range(len(total_reward)):
    if i < len(epsilon):
        label = f"Epsilon {epsilon[i]}"
    else:
        label = f"UCB c {c_value[i - len(epsilon)]}"
    plt.bar(np.arange(bandit_amount) + i * 0.15, final_rewards[i], width=0.15,label=label)
plt.xlabel('Bandit_number')
plt.ylabel('Total Reward')
plt.legend()
plt.show()

fig, axes = plt.subplots(1, 2,figsize=(15, 6)) 
for i in range(len(epsilon)):
    axes[0].plot(cumulative_avg[i], label=f'epsilon {epsilon[i]}')
for i in range(len(c_value)):
    axes[0].plot(cumulative_avg[len(epsilon)-1+i], label=f'c_value {c_value[i]}')
axes[0].set_xlabel('Times')
axes[0].set_ylabel('Action')
axes[0].set_xscale('log')
axes[0].legend()
for i in range(len(epsilon)):
    axes[1].plot(cumulative_reward_avg[i], label=f'epsilon {epsilon[i]}')
for i in range(len(c_value)):
    axes[1].plot(cumulative_reward_avg[len(epsilon)-1+i], label=f'c_value {c_value[i]}')
axes[1].set_xlabel('Times')
axes[1].set_ylabel('Reward')
axes[1].set_xscale('log')
axes[1].legend()
plt.show()


for i in range(len(epsilon)):
    plt.figure(figsize=(15, 7))
    for bandit in range(bandit_amount):
        # plt.plot(np.arange(n_time), np.cumsum(total_reward[i][bandit]) / (np.arange(n_time) + 1), label=f'epsilon {epsilon[i]} Bandit {bandit}')
        plt.plot(np.arange(n_time), np.cumsum(total_reward[i][bandit]) , label=f'epsilon {epsilon[i]} Bandit {bandit}')
    plt.xlabel('Times')
    plt.ylabel('Reward')
    plt.xscale('log')
    plt.legend()
    plt.show()

# plt.figure(figsize=(15, 7))
for i in range(len(c_value)):
    plt.figure(figsize=(15, 7))
    for bandit in range(bandit_amount):
        # plt.plot(np.arange(n_time), np.cumsum(total_reward[len(epsilon)-1+i][bandit]) / (np.arange(n_time) + 1), label=f'c_value {c_value[i]} Bandit {bandit}')
        plt.plot(np.arange(n_time), np.cumsum(total_reward[len(epsilon)-1+i][bandit]) , label=f'c_value {c_value[i]} Bandit {bandit}')
    plt.xlabel('Times')
    plt.ylabel('Reward')
    plt.xscale('log')
    plt.legend()
    plt.show()

# for ax in axes.flatten():  # Flatten to iterate over all axes
#     ax.set_xlabel('Reward')
#     ax.set_ylabel('Time')
#     ax.set_xscale('log')
#     ax.legend()
# plt.tight_layout()

# plot(data,"Try")