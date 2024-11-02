import matplotlib.pyplot as plt
from config import number_of_episodes, number_of_steps, root_path
import pandas as pd
import numpy as np
import os

def plot_epsilon(epsilons, steps, x_lable, y_lable, title, label):
    plt.figure(figsize=(10, 6))
    plt.plot(steps, epsilons, label=label, color='b', marker='o')
    plt.xlabel(x_lable)
    plt.ylabel(y_lable)
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(rl_path, f'{title}.jpg'), format='jpg')  # Save as JPG
    plt.savefig(os.path.join(rl_path, f'{title}.svg'), format='svg')
    plt.show()

def update_epsilon(epsilon, epsilon_decay=0.00009, epsilon_min=0.12):
    if epsilon >= epsilon_min:
       epsilon -= epsilon*epsilon_decay
    return epsilon
rl_path = os.path.join(root_path, 'RL_Results')
# if not os.path.exists(rl_path):os.makedirs(rl_path)
# steps, epsilons = list(range(number_of_steps*number_of_episodes)), [0.99]
# for i in steps[:-1]:
#     epsilons.append(update_epsilon(epsilons[-1]))
# # plot_epsilon(epsilons, steps, 'Steps', 'Epsilon', 'Epsilon Value Over Steps', 'Epsilon Value')
reward_per_steps = pd.read_csv(os.path.join(rl_path, 'RL_Resutls.csv'))['reward'].tolist()
reward = []
for i in range(0, len(reward_per_steps), 300):
    reward.append(sum(reward_per_steps[i:i+300]))
episodes = list(range(len(reward)))

plot_epsilon(reward, episodes, 'Episodes', 'Reward', 'Reward Value Over Episodes', 'Reward Value')

measures_names = ['num_dropped_packets_no_connection', 'num_dropped_packets_buffer_overflow','num_dropped_packets_expiration', 'num_dropped_packets_distance', 'num_arrived', 'e2e_delay', 'pdr']

measures_per_steps = pd.read_csv(os.path.join(rl_path, 'measures.csv'))

for measure_name in measures_names:
    measure = []
    measure_per_steps = measures_per_steps[measure_name].tolist()
    for i in range(0, len(measure_per_steps), 300):
        measure.append(sum(measure_per_steps[i:i+300]))
    episodes = list(range(len(measure)))
    plot_epsilon(reward, episodes, 'Episodes', 'measure_name', measure_name+' Value Over Episodes', measure_name+' Value')
