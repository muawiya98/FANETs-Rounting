from config import root_path, numbper_of_nodes, save_object, load_object
import numpy as np
import pandas as pd
import os
import sys

class QLearningRouting:
    def __init__(self, num_actions, id, lr=0.01, epsilon=0.99, epsilon_decay=0.00009, epsilon_min=0.12, gamma=0.9):
        self.lr, self.epsilon, self.epsilon_decay, self.epsilon_min, self.gamma = lr, epsilon,  epsilon_decay, epsilon_min, gamma
        self.state, self.action = None, None
        self.num_actions = num_actions
        self.learned = False
        self.q_table = {}
        self.id = id
        self.data_frame = pd.DataFrame()
        self.data_frame_test_phase = pd.DataFrame()

        self.save_period = 0

    def make_as_learned(self):
        self.learned = True

    def get_state(self, destination):
        return destination
    
    def policy(self, state, available_action):
        available_action = [available_action[key].id for key, value in available_action.items() if value is not None]
        if available_action==[]:return self.id
        if np.random.rand() < self.epsilon:return np.random.choice(available_action)
        else:
            self.prepar_q_table(state=state)
            available_q_value = self.q_table[state][available_action] #
            if sum(available_q_value)==0:return np.random.choice(available_action)
            best_q_value_index = [i for i, x in enumerate(self.q_table[state]) if x == max(available_q_value)]
            best_q_value_index = [x for x in best_q_value_index if x in available_action][-1]
            temp_list = np.zeros(numbper_of_nodes)
            temp_list[best_q_value_index] = self.q_table[state][best_q_value_index]
            return np.argmax(temp_list)
    
    def best_action(self, destination, neighbors):
        action = []
        if destination == self.id: return action.append(self.id)
        action = [node_id for node_id in neighbors if node_id==destination]
        return action[-1] if len(action)>=1 else None
    
    def choose_action(self, destination, neighbors, available_action, reward=None):
        next_state = self.get_state(destination)
        action = self.best_action(destination, neighbors)
        if action is None:
            action = self.policy(next_state, available_action)
        if not self.action is None:self.update_q_table(self.state, self.action, reward, next_state)
        self.update_epsilon()
        self.action = action
        self.state = next_state
        return self.action

    def choose_action_test_phase(self, destination, neighbors, available_action, reward, id_for_load):
        q_table = load_object(path=os.path.join(root_path, 'RL_Results'), filename=f'agent_{id_for_load}')
        next_state = self.get_state(destination)
        action = self.best_action(destination, neighbors)
        if action is None:
            try:
                action = np.argmax(q_table[next_state])
            except:action = self.id
            # available_action = [available_action[key].id for key, value in available_action.items() if value is not None]
            # if available_action==[]:action=self.id
            # available_q_value = q_table[next_state][available_action] #
            # if sum(available_q_value)==0:action = np.random.choice(available_action)
            # best_q_value_index = [i for i, x in enumerate(q_table[next_state]) if x == max(available_q_value)]
            # best_q_value_index = [x for x in best_q_value_index if x in available_action][-1]
            # temp_list = np.zeros(numbper_of_nodes)
            # temp_list[best_q_value_index] = q_table[next_state][best_q_value_index]
            # action = np.argmax(temp_list)   
        
        df = pd.DataFrame([[action, reward, self.epsilon, next_state]], columns=['action', 'reward', 'epsilon','state'])
        self.data_frame_test_phase = self.data_frame_test_phase.append(df, ignore_index=True) # type: ignore
        save_path = os.path.join(root_path, 'RL_Results')
        if not os.path.exists(save_path):os.makedirs(save_path)
        self.data_frame_test_phase.to_csv(os.path.join(save_path, f'RL_Resutls_Test_{self.id}.csv'), index=False)
        return action

    def prepar_q_table(self, state=None, next_state=None):
        if (not state is None) and (not state in self.q_table.keys()):
            self.q_table[state] = np.zeros(numbper_of_nodes)
        if (not next_state is None) and (not next_state in self.q_table.keys()):
            self.q_table[next_state] = np.zeros(numbper_of_nodes)
    
    def update_q_table(self, state, action, reward, next_state):
        self.prepar_q_table(state=state, next_state=next_state)
        self.q_table[state][action] = ((1 - self.lr) * self.q_table[state][action]) + (self.lr * (reward + self.gamma * max(self.q_table[next_state])))
        
        df = pd.DataFrame([[action, reward, self.epsilon, state, self.q_table[state][action], self.q_table]], columns=['action', 'reward', 'epsilon','state', 'q_value', 'q_table'])
        self.data_frame = self.data_frame.append(df, ignore_index=True) # type: ignore
        self.save_period+=1
        if self.save_period%100==0:
            save_path = os.path.join(root_path, 'RL_Results')
            if not os.path.exists(save_path):os.makedirs(save_path)
            self.data_frame.to_csv(os.path.join(save_path, f'RL_Resutls_{self.id}.csv'), index=False)
            save_object(path=save_path, obj=self.q_table, filename=f'agent_{self.id}')
    def update_epsilon(self):
        if self.epsilon >= self.epsilon_min:
            self.epsilon -= self.epsilon*self.epsilon_decay