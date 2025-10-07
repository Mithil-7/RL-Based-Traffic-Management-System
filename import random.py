import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import models, layers, optimizers
import matplotlib.pyplot as plt

class DynamicTrafficEnvironment:
    def __init__(self, num_roads, num_vehicle_types, waiting_penalty=0.03, closed_road_reward_factor=0.2):
        self.num_roads = num_roads
        self.num_vehicle_types = num_vehicle_types
        self.action_size = num_roads
        self.state_size = num_roads * num_vehicle_types
        self.vehicle_rewards = np.zeros((num_roads, num_vehicle_types))
        self.waiting_penalty = waiting_penalty
        self.closed_road_reward_factor = closed_road_reward_factor
        self.road_openings = np.zeros(num_roads)
        self.time_saved = 0
        self.total_time_saved = 0      
        self.episode_time_saved = 0    
        self.road_clear_times = []  
        print(f"Initialized environment with {num_roads} roads and {num_vehicle_types} vehicle types.")
        self.current_state = self.reset()

    def reset(self):
        self.current_state = []
        self.episode_time_saved = 0   
        for i in range(self.num_roads):
            important_vehicle_count = random.randint(0, 4)
            non_important_vehicle_counts = [random.randint(important_vehicle_count + 1, 10) for _ in range(self.num_vehicle_types - 1)]
            road_state = [important_vehicle_count] + non_important_vehicle_counts
            self.current_state.extend(road_state)
            print(f"Road {i} initialized with state: {road_state}")
        state_array = np.array(self.current_state)
        print(f"Environment reset. Initial state: {state_array}")
        return state_array

    def calculate_road_clear_time(self, road_index):
        """
        Calculate the total time required to clear all traffic on a specific road.
        Formula: (number of vehicles in road * 18) / 5
        
        Args:
            road_index (int): Index of the road (0, 1, 2, etc.)
            
        Returns:
            float: Time in seconds to clear the road
        """
        start_index = road_index * self.num_vehicle_types
        end_index = start_index + self.num_vehicle_types
        vehicles_on_road = self.current_state[start_index:end_index]
        total_vehicles = sum(vehicles_on_road)
        
        # Formula: (no. of vehicles * 18) / 5
        clear_time = (total_vehicles * 18) / 5
        
        print(f"Road {road_index} - Vehicles: {vehicles_on_road}, Total: {total_vehicles}")
        print(f"Road {road_index} - Estimated clear time: {clear_time:.2f} seconds")
        
        return clear_time

    def calculate_all_roads_clear_time(self):
        """
        Calculate clear time for all roads and return as a list.
        
        Returns:
            list: Clear times for all roads
        """
        clear_times = []
        total_time = 0
        
        print("\n=== Road Clear Time Analysis ===")
        for i in range(self.num_roads):
            road_time = self.calculate_road_clear_time(i)
            clear_times.append(road_time)
            total_time += road_time
        
        print(f"Total time to clear ALL roads: {total_time:.2f} seconds")
        print(f"Average time per road: {total_time/self.num_roads:.2f} seconds")
        
        return clear_times

    def get_priority_road(self):
        """
        Get the road with the longest clear time (highest priority).
        
        Returns:
            tuple: (road_index, clear_time)
        """
        clear_times = []
        for i in range(self.num_roads):
            start_index = i * self.num_vehicle_types
            end_index = start_index + self.num_vehicle_types
            total_vehicles = sum(self.current_state[start_index:end_index])
            clear_time = (total_vehicles * 18) / 5
            clear_times.append((i, clear_time))
        
        # Sort by clear time (descending) - highest priority first
        clear_times.sort(key=lambda x: x[1], reverse=True)
        priority_road, max_time = clear_times[0]
        
        print(f"Priority road: {priority_road} (clear time: {max_time:.2f} seconds)")
        return priority_road, max_time

    def print_road_clear_times(self):
        """
        Print clear times for all roads in a formatted way.
        """
        print("\n" + "="*50)
        print("ROAD CLEAR TIME ANALYSIS")
        print("="*50)
        
        for i in range(self.num_roads):
            start_index = i * self.num_vehicle_types
            end_index = start_index + self.num_vehicle_types
            vehicles = self.current_state[start_index:end_index]
            total_vehicles = sum(vehicles)
            clear_time = (total_vehicles * 18) / 5
            
            print(f"Road {i}:")
            print(f"  Vehicles by type: {vehicles}")
            print(f"  Total vehicles: {total_vehicles}")
            print(f"  Clear time: {clear_time:.2f} seconds")
            print(f"  Clear time: {clear_time/60:.2f} minutes")
            print("-" * 30)

  
    def step(self, action):
        print(f"Action taken: {action}")
        road_index = action
        start_index = road_index * self.num_vehicle_types
        end_index = start_index + self.num_vehicle_types
        vehicles = self.current_state[start_index:end_index]
        
        # Calculate clear time BEFORE taking action
        pre_action_clear_time = self.calculate_road_clear_time(road_index)
        
        print(f"Current state for road {road_index}: {vehicles}")

        important_vehicle_reward = 1.25 
        non_important_vehicle_reward = 1.0  
        vehicles_to_go = [random.randint(0, v) for v in vehicles]
        vehicle_rewards = [important_vehicle_reward if i == 0 else non_important_vehicle_reward for i in range(self.num_vehicle_types)]
        rewards = sum(v_to_go * vehicle_rewards[i] for i, v_to_go in enumerate(vehicles_to_go))
        self.current_state[start_index:end_index] = [v - g for v, g in zip(vehicles, vehicles_to_go)]
        leftover_vehicles = self.current_state[start_index:end_index]
        waiting_penalty = self.waiting_penalty * sum(leftover_vehicles)
        total_reward = rewards - waiting_penalty
        vehicles_passed = sum(vehicles_to_go)
        self.total_time_saved += vehicles_passed     
        self.episode_time_saved += vehicles_passed    
        self.road_openings[road_index] += 1 

        post_action_clear_time = self.calculate_road_clear_time(road_index)
        time_reduction = pre_action_clear_time - post_action_clear_time
        
        print(f"Time reduction on road {road_index}: {time_reduction:.2f} seconds")

    
        if sum(leftover_vehicles) == 0:
            print(f"🎉 Road {road_index} is completely cleared!")
            self.road_clear_times.append((road_index, pre_action_clear_time))

        for i in range(self.num_roads):
            if i != road_index:
                closed_road_start_index = i * self.num_vehicle_types
                closed_road_end_index = closed_road_start_index + self.num_vehicle_types
                closed_road_vehicles = self.current_state[closed_road_start_index:closed_road_end_index]
                closed_road_reward = self.closed_road_reward_factor * sum(closed_road_vehicles)
                total_reward -= closed_road_reward
                print(f"Closed road {i} reward: {closed_road_reward}")

        done = all(v == 0 for v in self.current_state)
        print(f"Updated state: {self.current_state}, Done: {done}")
        return np.array(self.current_state), total_reward, done

    def print_state(self):
        print("Current state of each road:")
        for i in range(self.num_roads):
            start_index = i * self.num_vehicle_types
            end_index = start_index + self.num_vehicle_types
            vehicles = self.current_state[start_index:end_index]
            print(f"Road {i} State: {vehicles}")

    def print_vehicle_rewards(self):
        print("Vehicle Rewards:")
        for i in range(self.num_roads):
            print(f"Road {i}: {self.vehicle_rewards[i]}")

    def print_road_openings(self):
        print("Road Openings:")
        for i in range(self.num_roads):
            print(f"Road {i} opened {self.road_openings[i]} times")

    def print_time_saved(self):
        print(f"Episode time saved: {self.episode_time_saved}")
        print(f"Total time saved: {self.total_time_saved}")



def build_model(state_size, action_size):
    model = models.Sequential()
    model.add(layers.Dense(32, activation='relu', input_shape=(state_size,)))
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dense(8, activation='relu'))
    model.add(layers.Dense(action_size, activation='linear'))
    model.compile(loss='mse', optimizer=optimizers.Adam(learning_rate=0.001))
    print("Model built and compiled.")
    return model

class DQNAgent:
    def __init__(self, state_size, action_size, learning_rate=0.001, gamma=0.95, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01, batch_size=32, memory_size=2000):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = []
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.model = self._build_model()
        print(f"Initialized DQNAgent with state size {state_size} and action size {action_size}.")

    def _build_model(self):
        model = models.Sequential()
        model.add(layers.Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(layers.Dense(24, activation='relu'))
        model.add(layers.Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=optimizers.Adam(learning_rate=self.learning_rate))
        print("Neural network model built.")
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > 2000:
            self.memory.pop(0)
        print(f"Memory size: {len(self.memory)}")

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            action = random.randrange(self.action_size)
            print(f"Random action: {action}")
            return action
        act_values = self.model.predict(state[np.newaxis])
        action = np.argmax(act_values[0])
        print(f"Predicted action: {action}")
        return action

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        minibatch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target += self.gamma * np.amax(self.model.predict(next_state[np.newaxis])[0])
            target_f = self.model.predict(state[np.newaxis])
            target_f[0][action] = target
            self.model.fit(state[np.newaxis], target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


env = DynamicTrafficEnvironment(3, 3)  
agent = DQNAgent(state_size=env.state_size, action_size=env.action_size)  
episode_rewards = [] 
road_states = []  

for episode in range(50):
    print(f"Episode {episode} started.")
    state = env.reset()
    total_reward = 0
    done = False
    
    
    env.print_road_clear_times()
    
    while not done:
        env.print_state()
        
        # Get priority road suggestion
        priority_road, priority_time = env.get_priority_road()
        
        action = agent.act(state)
        next_state, reward, done = env.step(action)
        
        # Show remaining clear times after action
        remaining_times = env.calculate_all_roads_clear_time()
        
        total_reward += reward
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        
        if done:
            agent.replay()
            print(f"Episode {episode} finished. Total Reward: {total_reward}")
            break
