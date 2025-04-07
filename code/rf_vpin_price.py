import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import random
from collections import deque
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Helper Functions
def formatPrice(n):
    return ("-$" if n < 0 else "$") + "{0:.2f}".format(abs(n))

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def getState(data, t, n):
    """
    Returns an n-day state representation ending at time t
    """
    d = t - n + 1
    block = []
    
    # Pad with t0 if d < 0
    if d < 0:
        block = np.array([-d * [data[0]] + list(data[0:t + 1])])
    else:
        block = data[d:t + 1]
    
    res = []
    # Calculate price differences and apply sigmoid
    for i in range(len(block) - 1):
        res.append(sigmoid(block[i + 1] - block[i]))
    
    # Ensure we have n-1 elements in our state
    while len(res) < n - 1:
        res.append(0)
    
    # Return as a 1 x (n-1) array
    return np.array([res])

def plot_behavior(data_input, states_buy, states_sell, profit, save_plot=False, filename=None):
    # Create plots directory if it doesn't exist
    if save_plot and not os.path.exists('plots'):
        os.makedirs('plots')
        
    fig = plt.figure(figsize = (15,5))
    plt.plot(data_input, color='r', lw=2.)
    plt.plot(data_input, '^', markersize=10, color='m', label = 'Buying signal', markevery = states_buy)
    plt.plot(data_input, 'v', markersize=10, color='k', label = 'Selling signal', markevery = states_sell)
    plt.title('Total gains: %f'%(profit))
    plt.legend()
    
    if save_plot:
        if filename is None:
            filename = f'plots/trading_behavior_profit_{profit:.2f}.png'
        else:
            filename = f'plots/{filename}'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {filename}")
    
    plt.show()

# Define smaller DQN Network using PyTorch
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        # Simplified architecture with fewer parameters
        self.fc1 = nn.Linear(state_size, 32)  # Reduced from 64 to 32 neurons
        self.fc2 = nn.Linear(32, 16)          # Reduced from 32 to 16 neurons
        self.fc3 = nn.Linear(16, action_size) # Removed one layer, connecting directly to output
        
        # Print model parameter count
        params = sum(p.numel() for p in self.parameters())
        print(f"Model created with {params} parameters")
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)  # Linear activation for Q-values

# Agent Class
class Agent:
    def __init__(self, state_size, is_eval=False, model_name=""):
        self.state_size = state_size # normalized previous days
        self.action_size = 3 # sit, buy, sell
        self.memory = deque(maxlen=1000)
        self.inventory = []
        self.model_name = model_name
        self.is_eval = is_eval

        # Hyperparameters
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        
        # Device configuration - using MPS for Mac with Apple Silicon, fallback to CPU
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            print("Using MPS (Metal Performance Shaders) for acceleration")
        else:
            self.device = torch.device("cpu")
            print("MPS not available, using CPU")
        
        # Create the neural network
        self.model = DQN(state_size, self.action_size).to(self.device)
        
        # Load model if evaluating
        if is_eval and os.path.exists(model_name):
            self.model.load_state_dict(torch.load(model_name, map_location=self.device))
            self.model.eval()
            print(f"Model loaded from {model_name}")
        
        # Loss and optimizer
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
    
    def act(self, state): 
        # Random action for exploration
        if not self.is_eval and random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        
        # Convert numpy array to PyTorch tensor
        state_tensor = torch.FloatTensor(state).to(self.device)
        
        # Get action from model
        with torch.no_grad():
            q_values = self.model(state_tensor)
        
        return torch.argmax(q_values).item()

    def expReplay(self, batch_size):
        # Skip if we don't have enough experiences
        if len(self.memory) < batch_size:
            return
            
        mini_batch = []
        l = len(self.memory)
        for i in range(l - batch_size + 1, l):
            mini_batch.append(self.memory[i])
        
        for state, action, reward, next_state, done in mini_batch:
            # Convert numpy arrays to PyTorch tensors
            state_tensor = torch.FloatTensor(state).to(self.device)
            next_state_tensor = torch.FloatTensor(next_state).to(self.device)
            
            # Calculate target Q value
            target = reward
            if not done:
                with torch.no_grad():
                    target = reward + self.gamma * torch.max(self.model(next_state_tensor)).item()
            
            # Get current Q values
            self.model.train()
            q_values = self.model(state_tensor)
            target_f = q_values.clone().detach()
            target_f[0, action] = target
            
            # Perform one step of optimization
            self.optimizer.zero_grad()
            loss = self.criterion(q_values, target_f)
            loss.backward()
            self.optimizer.step()
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# Main function
def main():
    # Load data
    try:
        dataset = pd.read_csv('data/SP500.csv', index_col=0)
    except FileNotFoundError:
        print("Data file not found. Please ensure the SP500.csv file is in the 'data' directory.")
        return
    
    # Check and create data directory if it doesn't exist
    if not os.path.exists('data'):
        os.makedirs('data')
    
    # Check and create models directory if it doesn't exist
    if not os.path.exists('models'):
        os.makedirs('models')
    
    # Prepare data
    close_prices = dataset['Close'].values
    
    # Split data into training and testing sets
    train_size = int(len(close_prices) * 0.8)
    train_data = close_prices[:train_size]
    test_data = close_prices[train_size:]
    
    print(f"Data loaded: {len(close_prices)} total points")
    print(f"Training data: {len(train_data)} points")
    print(f"Testing data: {len(test_data)} points")
    
    # Parameters
    window_size = 10  # Increased to 10 as requested
    batch_size = 32
    episodes = 5
    model_name = "models/model_sp500.pth"
    
    # Training function (inline definition to fix bugs)
    def train_model(stock_data, window_size=10, batch_size=32, episodes=5, model_name="model_sp500.pth"):
        data = stock_data
        l = len(data) - 1
        agent = Agent(window_size)
        
        print(f"Training on {l} data points for {episodes} episodes")
        
        for e in range(episodes):
            print(f"Running episode {e+1}/{episodes}")
            state = getState(data, 0, window_size + 1)
            total_profit = 0
            agent.inventory = []
            losses = []
            
            for t in range(l):
                action = agent.act(state)
                
                # Make sure we don't go out of bounds
                if t + 1 < len(data):
                    next_state = getState(data, t + 1, window_size + 1)
                else:
                    # If we're at the last element, just use the current state
                    next_state = state
                
                reward = 0

                if action == 1: # buy
                    agent.inventory.append(data[t])
                    print(f"Step {t}: Buy: {formatPrice(data[t])}")
                
                elif action == 2 and len(agent.inventory) > 0: # sell
                    bought_price = agent.inventory.pop(0)
                    reward = max(data[t] - bought_price, 0)
                    total_profit += data[t] - bought_price
                    print(f"Step {t}: Sell: {formatPrice(data[t])} | Profit: {formatPrice(data[t] - bought_price)}")
                
                done = True if t == l - 1 else False
                agent.memory.append((state, action, reward, next_state, done))
                state = next_state
                
                if done:
                    print("##############")
                    print(f"Episode {e+1} completed")
                    print(f"Total Profit: {formatPrice(total_profit)}")
                    print(f"Final epsilon: {agent.epsilon:.4f}")
                    print("##############")
                
                if len(agent.memory) > batch_size:
                    agent.expReplay(batch_size)
                    
                # Print progress every 500 steps
                if t % 500 == 0 and t > 0:
                    print(f"Progress: {t}/{l} steps completed in episode {e+1}")
            
            # Save model after every epoch (episode)
            # Create models directory if it doesn't exist
            if not os.path.exists(os.path.dirname(model_name)):
                os.makedirs(os.path.dirname(model_name))
                
            # Save with episode number in filename to keep track of progress
            epoch_model_name = model_name.replace('.pth', f'_epoch_{e+1}.pth')
            torch.save(agent.model.state_dict(), epoch_model_name)
            print(f"Model saved to {epoch_model_name} after episode {e+1}")
            
            # Also save as the main model file (overwrite)
            torch.save(agent.model.state_dict(), model_name)
            print(f"Model also saved as {model_name}")
    
    # Evaluation function (inline definition to fix bugs)
    def evaluate_model(stock_data, window_size=10, model_name="model_sp500.pth", save_plot=False):
        data = stock_data
        l = len(data) - 1
        agent = Agent(window_size, is_eval=True, model_name=model_name)
        
        print(f"Evaluating on {l} data points")
        
        state = getState(data, 0, window_size + 1)
        total_profit = 0
        agent.inventory = []
        
        states_buy = []
        states_sell = []
        
        for t in range(l):
            action = agent.act(state)
            
            # Make sure we don't go out of bounds
            if t + 1 < len(data):
                next_state = getState(data, t + 1, window_size + 1)
            else:
                # If we're at the last element, just use the current state
                next_state = state
            
            if action == 1: # buy
                agent.inventory.append(data[t])
                states_buy.append(t)
                print(f"Step {t}: Buy: {formatPrice(data[t])}")
            
            elif action == 2 and len(agent.inventory) > 0: # sell
                bought_price = agent.inventory.pop(0)
                reward = max(data[t] - bought_price, 0)
                total_profit += data[t] - bought_price
                states_sell.append(t)
                print(f"Step {t}: Sell: {formatPrice(data[t])} | Profit: {formatPrice(data[t] - bought_price)}")
            
            state = next_state
            
            # Print progress every 500 steps
            if t % 500 == 0 and t > 0:
                print(f"Progress: {t}/{l} steps completed in evaluation")
        
        print("------------------------------------------")
        print(f"Total Profit: {formatPrice(total_profit)}")
        print(f"Total Buy actions: {len(states_buy)}")
        print(f"Total Sell actions: {len(states_sell)}")
        print("------------------------------------------")
        
        # Extract model name for plot filename
        model_basename = os.path.basename(model_name).replace('.pth', '')
        plot_filename = f"trading_behavior_{model_basename}.png"
        
        plot_behavior(data, states_buy, states_sell, total_profit, save_plot=save_plot, filename=plot_filename)
        return total_profit, states_buy, states_sell
    
    # Ask user whether to train or evaluate
    choice = input("Do you want to train (t) or evaluate (e) the model? [t/e]: ").lower()
    
    if choice == 't':
        # Train the model
        train_model(train_data, window_size, batch_size, episodes, model_name)
        
        # Ask if user wants to evaluate after training
        eval_choice = input("Do you want to evaluate the trained model? [y/n]: ").lower()
        if eval_choice == 'y':
            save_plot_choice = input("Do you want to save the evaluation plot? [y/n]: ").lower()
            save_plot = (save_plot_choice == 'y')
            profit, states_buy, states_sell = evaluate_model(test_data, window_size, model_name, save_plot)
            print(f"Model trained and evaluated successfully!")
            print(f"Total profit on test data: {formatPrice(profit)}")
    
    elif choice == 'e':
        # Allow selecting a specific epoch model
        model_files = [f for f in os.listdir('models') if f.startswith('model_sp500') and f.endswith('.pth')]
        
        if len(model_files) == 0:
            print("No model files found. Please train the model first.")
            return
            
        print("Available model files:")
        for i, model_file in enumerate(model_files):
            print(f"{i+1}. {model_file}")
            
        model_choice = input(f"Enter model number to evaluate (1-{len(model_files)}) or press enter for default: ")
        
        if model_choice.strip() and model_choice.isdigit() and 1 <= int(model_choice) <= len(model_files):
            selected_model = os.path.join('models', model_files[int(model_choice)-1])
            print(f"Selected model: {selected_model}")
        else:
            selected_model = model_name
            print(f"Using default model: {selected_model}")
            
        # Check if model exists
        if not os.path.exists(selected_model):
            print(f"Model file {selected_model} not found. Please train the model first.")
            return
            
        # Evaluate the model
        save_plot_choice = input("Do you want to save the evaluation plot? [y/n]: ").lower()
        save_plot = (save_plot_choice == 'y')
        profit, states_buy, states_sell = evaluate_model(test_data, window_size, selected_model, save_plot)
        print(f"Model evaluated successfully!")
        print(f"Total profit on test data: {formatPrice(profit)}")
    
    else:
        print("Invalid choice. Please run the script again and enter 't' for training or 'e' for evaluation.")

if __name__ == "__main__":
    main() 