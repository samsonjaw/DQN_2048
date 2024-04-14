import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from collections import deque
import random
import datetime
from tb import logger

state_size = 16
action_size = 4

class DQN:
    
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.total_episode = 500
        self.memory = deque(maxlen=5000)
        self.gamma = 0.95 #reward decay
        self.init_epsilon = 0.3 #if epsilon greedy epsilon= 1
        self.epsilon = 0.3
        self.epsilon_min = 0.01
        self.epsilon_decay_rate = 0.995# if epsilon greedy epsilon_decay < 1
        self.learning_rate = 0.01 #alpha
        self.evaluation_net = self.build_net()
        self.target_net = self.build_net()
        self.target_update_freq = 500
        self.update_target() #copy evaluation to target
        self.step = 0

    def build_net(self):
        net = Sequential()
        net.add(Input(shape=(self.state_size))) #input
        net.add(Dense(512, activation='relu'))
        net.add(Dense(256, activation='relu'))
        net.add(Dense(self.action_size, activation = 'linear'))#output
        net.compile(loss = 'mse', optimizer = Adam(learning_rate = self.learning_rate))
        return net

    def update_target(self):
        self.target_net.set_weights(self.evaluation_net.get_weights())

    def preprocess_state(self, state): 
        return np.log(state + 1) / 16

    def store(self, state, action, reward, next_state, gameover):
        self.memory.append((self.preprocess_state(state), action, reward, self.preprocess_state(next_state), gameover))
    
    def choose_action(self, state):
        
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        state = state.reshape(1, -1) 
        state = self.preprocess_state(state)
        actions = self.evaluation_net.predict(state)
        return np.argmax(actions[0]) #return the most valuable action
    
    def replay(self, batch_size):#learn from the memory
        current_batch_size = min(len(self.memory), batch_size)
        minibatch = random.sample(self.memory, current_batch_size) #randomly sample form memory
    
        #memory inclue state action, reward, next_state, done
        states = np.array([s[0] for s in minibatch])
        next_states = np.array([s[3] for s in minibatch])

        states = states.reshape(current_batch_size, -1)
        next_states = next_states.reshape(current_batch_size, -1)

        q_values = self.evaluation_net.predict(states)
        q_next = self.target_net.predict(next_states)

        for i, (state, action, reward, next_state, done) in enumerate(minibatch):
            target = reward

            next_state = next_state[np.newaxis, :, :, np.newaxis]
            
            if not done:
                target = self.gamma * np.amax(q_next[i])
            
            q_values[i][action] = target
                      
        history = self.evaluation_net.fit(states, q_values, epochs = 1, verbose = 0)
        self.step += 1
        loss_value = history.history['loss'][0]
        logger.log_scalar("loss", loss_value, self.step)
        
        #if self.epsilon > self.epsilon_min:
        #    self.epsilon *= self.epsilon_decay_rate
        #    print(self.epsilon)

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


