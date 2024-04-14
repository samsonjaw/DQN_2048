import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input
from tensorflow.keras.models import load_model
from collections import deque
import random
import datetime
from tb import logger
from SumTree import SumTree
state_size = 16
action_size = 4

class DQN:
    
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.total_episode = 10000
        self.tree = SumTree(10000)
        self.a = 0.6
        self.e = 0.01
        self.priority_max = 0.1

        self.beta = 0.4
        self.beta_increase = 0.001

        self.epochs = 10

        self.gamma = 0.99 #reward decay
        self.init_epsilon = 0.3 #if epsilon greedy epsilon= 1
        self.epsilon = 0.3
        #self.epsilon_min = 0.01
        #self.epsilon_decay_rate = 0.995# if epsilon greedy epsilon_decay < 1
        self.learning_rate = 0.001 #alpha
        self.evaluation_net = self.build_net()
        self.target_net = self.build_net()
        self.target_update_freq = 500
        self.update_target() #copy evaluation to target
        self.step = 0
        

    def build_net(self):
        net = Sequential()
        net.add(Input(shape=(4,4,1)))
        net.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'))
        net.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
        net.add(Flatten())
        net.add(Dense(512, activation='relu'))
        net.add(Dense(128, activation='relu'))
        net.add(Dense(action_size,activation='linear'))
        net.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return net

    def update_target(self):
        self.target_net.set_weights(self.evaluation_net.get_weights())

    def preprocess_state(self, state): 
        return np.log(state + 1) / 16

    def choose_action(self, state):
        
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        state = state[np.newaxis, :, :, np.newaxis]
        state = self.preprocess_state(state)
        actions = self.evaluation_net.predict(state)
        return np.argmax(actions[0]) #return the most valuable action
    
    def store(self, state, action, reward, next_state, gameover):
        data = (state, action, reward, next_state, gameover)
        priority = (np.abs(self.priority_max) + self.e) ** self.a #  proportional priority
        self.tree.add_priority(priority, data)

    def sample(self, batch_size):
        states, actions, rewards, next_states, gameovers, idxs, is_weights = [], [], [], [], [], [], []
        total_priority = self.tree.total()
        segment = total_priority / batch_size
        self.beta = min(1., self.beta + self.beta_increase)

        min_prob = np.min(self.tree.tree[-self.tree.capacity:]) / total_priority
        min_prob = max(min_prob, 1e-10)
        max_weight = (min_prob * batch_size) ** (-self.beta)

        for i in range(batch_size):
            a = segment * i
            b = segment * (i+1)
            value = random.uniform(a,b)
            idx, p, data = self.tree.get(value)

            sampling_prob = p / total_priority
            is_weight = (sampling_prob * batch_size) ** (-self.beta) / max_weight

            #data = (state, action, reward, next_state, gameover)
            states.append(data[0])
            actions.append(data[1])
            rewards.append(data[2])
            next_states.append(data[3])
            gameovers.append(data[4])
            idxs.append(idx)       
            is_weights.append(is_weight)

        is_weights = np.array(is_weights).reshape(-1, 1)
        return idxs, np.stack(states, axis=0), actions, rewards, np.stack(next_states, axis=0), gameovers, is_weights

    def update_batch(self, idxs, errors):
        self.priority_max = max(self.priority_max, np.max(np.abs(errors)))
        for i, idx in enumerate(idxs):
            error = np.abs(errors[i])
            priority = (np.max(error) + self.e) ** self.a
            self.tree.update_priority(idx, priority) 

    def replay(self, batch_size):#learn from the memory
        current_batch_size = min(self.tree.n_entries, batch_size)
        idxs, states, actions, rewards, next_states, gameovers, is_weights = self.sample(current_batch_size) #randomly sample form memory

        #states = states[:, :, :, np.newaxis]
        #next_states = next_states[:, :, :, np.newaxis]

        states = np.expand_dims(states, axis=-1)
        next_states = np.expand_dims(next_states, axis=-1)

        q_values = self.evaluation_net.predict(states)
        q_next = self.target_net.predict(next_states)

        targets = q_values.copy()

        for i in range(current_batch_size):
            target = rewards[i]

            if not gameovers[i]:
                action = np.argmax(q_values[i])
                target = self.gamma * q_next[i][action]
            
            targets[i][actions[i]] = target

        with tf.GradientTape() as tape:
            predictions = self.evaluation_net(states, training = True)
            errors = targets - predictions
            weighted_errors = is_weights * errors**2
            loss = tf.reduce_mean(weighted_errors)
        gradients = tape.gradient(loss, self.evaluation_net.trainable_variables)
        self.evaluation_net.optimizer.apply_gradients(zip(gradients, self.evaluation_net.trainable_variables))
        
        errors = np.abs(errors) + self.e
        self.update_batch(idxs, errors)

        
        self.step += 1
        loss_value = loss.numpy()
        
        #print(f"Step {self.step}, Loss: {loss_value}")
        logger.log_scalar("loss", loss_value, self.step)
    

    def change_epsilon(self, value):
        self.epsilon = value

    def epsilon_decay(self, episode, total_episode):
        self.epsilon = self.init_epsilon * (1 - episode / total_episode)

    def save_model(self):
        self.evaluation_net.save('net/' + logger.time + '_evaluation_net_weights.h5')

        self.target_net.save('net/' + logger.time + '_target_net_weights.h5')

        #self.evaluation_net.load_weights('evaluation_net_weights.h5')

        #self.target_net.load_weights('target_net_weights.h5')

RL = DQN(state_size=16, action_size=4)
#tensorboard --logdir=logs