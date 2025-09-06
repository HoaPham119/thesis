import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import random
from collections import deque
import os
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Helper Functions
def formatPrice(n, scaler):
    price = np.array([[n,0]])
    price = scaler.inverse_transform(price)[0][0]
    return ("-$" if price < 0 else "$") + "{0:.2f}".format(abs(price))

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
        block = np.array([[-d * [data[0]] + list(data[0:t + 1])]])
    else:
        block = data[d:t + 1]
    
    res = []
    # Calculate differences and apply sigmoid to each feature (Price, VPIN)
    for i in range(len(block) - 1):
        # Price difference (block[i+1][0] is the price, block[i][0] is the previous price)
        price_diff = block[i + 1][0] - block[i][0]
        # VPIN difference (block[i+1][1] is the VPIN, block[i][1] is the previous VPIN)
        vpin_diff = block[i + 1][1] - block[i][1]
        res.append(sigmoid(price_diff))  # Apply sigmoid to price difference
        res.append(sigmoid(vpin_diff))  # Apply sigmoid to VPIN difference
    
    # Ensure we have n*2 - 2 elements in our state (since we have two features: Price, VPIN)
    while len(res) < n * 2 - 2:
        res.append(0)
    
    # Return as a 1 x (n*2 - 2) array
    return np.array([res])

def plot_behavior(data_input, states_buy, states_sell, profit, save_plot=False, filename=None):
    # Create plots directory if it doesn't exist
    if save_plot and not os.path.exists('plots'):
        os.makedirs('plots')
        
    fig = plt.figure(figsize=(15, 5))

    # Vẽ đường Price và VPIN
    plt.plot(data_input[:, 0], color='r', lw=2., label='Price')
    plt.plot(data_input[:, 1], color='b', lw=2., label='VPIN')

    # Vẽ các tín hiệu mua/bán trên giá
    plt.plot(states_buy, data_input[states_buy, 0], '^', markersize=10, color='m', label='Buying signal')
    plt.plot(states_sell, data_input[states_sell, 0], 'v', markersize=10, color='k', label='Selling signal')

    plt.title('Total gains: %f' % profit)
    plt.legend()
    plt.show()

    
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
        self.fc1 = nn.Linear(state_size, 64)  # Adjusted to reflect input size
        self.fc2 = nn.Linear(64, 32)  # Adjusted to reflect input size
        self.fc3 = nn.Linear(32, action_size) # Connecting directly to output
        
        params = sum(p.numel() for p in self.parameters())
        print(f"Model created with {params} parameters")
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)  # Linear activation for Q-values

# Agent Class
class Agent:
    def __init__(self, state_size, is_eval=False, model_name=""):
        self.state_size = state_size  # Adjusted for 2 features per day: price and vpin
        self.action_size = 3  # sit, buy, sell
        self.memory = deque(maxlen=1000)
        self.inventory = []
        self.model_name = model_name
        self.is_eval = is_eval

        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            print("Using MPS (Metal Performance Shaders) for acceleration")
        else:
            self.device = torch.device("cpu")
            print("MPS not available, using CPU")
        
        self.model = DQN(state_size, self.action_size).to(self.device)
        
        if is_eval and os.path.exists(model_name):
            self.model.load_state_dict(torch.load(model_name, map_location=self.device))
            self.model.eval()
            print(f"Model loaded from {model_name}")
        
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
    
    def act(self, state): 
        if not self.is_eval and random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        
        state_tensor = torch.FloatTensor(state).to(self.device)
        
        with torch.no_grad():
            q_values = self.model(state_tensor)
        
        return torch.argmax(q_values).item()

    def expReplay(self, batch_size):
        if len(self.memory) < batch_size:
            return
            
        mini_batch = []
        l = len(self.memory)
        for i in range(l - batch_size + 1, l):
            mini_batch.append(self.memory[i])
        
        for state, action, reward, next_state, done in mini_batch:
            state_tensor = torch.FloatTensor(state).to(self.device)
            next_state_tensor = torch.FloatTensor(next_state).to(self.device)
            
            target = reward
            if not done:
                with torch.no_grad():
                    target = reward + self.gamma * torch.max(self.model(next_state_tensor)).item()
            
            self.model.train()
            q_values = self.model(state_tensor)
            target_f = q_values.clone().detach()
            target_f[0, action] = target
            
            self.optimizer.zero_grad()
            loss = self.criterion(q_values, target_f)
            loss.backward()
            self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# Main function
def main():
    # Load data
    try:
        dataset = pd.read_csv('RL_data/VCBVPIN.csv', index_col=0)
        dataset = dataset.dropna()
    except FileNotFoundError:
        print("Data file not found. Please ensure the AAPL.csv file is in the 'data' directory.")
        return
    
    if not os.path.exists('data'):
        os.makedirs('data')
    
    if not os.path.exists('models'):
        os.makedirs('models')
    
    # Prepare data
    dataset = dataset.tail(200)
    price_vpin_data = dataset[["Price", "VPIN"]].values
    price_vpin_data = price_vpin_data[~np.isnan(price_vpin_data).any(axis=1)]  # Remove rows with NaN values

    train_size = int(len(price_vpin_data) * 0.8)
    train_data = price_vpin_data[:train_size]
    test_data = price_vpin_data[train_size:]
    
    # Fit scale data
    scaler = MinMaxScaler()
    train_data = scaler.fit_transform(train_data)
    test_data  = scaler.transform(test_data)
    
    print(f"Data loaded: {len(price_vpin_data)} total points")
    print(f"Training data: {len(train_data)} points")
    print(f"Testing data: {len(test_data)} points")
    
    window_size = 10
    batch_size = 32
    episodes = 5
    model_name = "models/model_sp500.pth"
    
    def train_model(stock_data, window_size=10, batch_size=32, episodes=5, model_name="model_sp500.pth"):
        data = stock_data
        l = len(data) - 1
        agent = Agent(window_size * 2)
        
        print(f"Training on {l} data points for {episodes} episodes")
        
        for e in range(episodes):
            print(f"Running episode {e+1}/{episodes}")
            state = getState(data, 0, window_size + 1)
            total_profit = 0
            agent.inventory = []
            
            for t in range(l):
                action = agent.act(state)
                
                if t + 1 < len(data):
                    next_state = getState(data, t + 1, window_size + 1)
                else:
                    next_state = state
                
                reward = 0

                if action == 1:
                    agent.inventory.append(data[t])
                    print(f"Step {t}: Buy: {formatPrice(data[t][0], scaler)}")
                elif action == 2 and len(agent.inventory) > 0:
                    bought_price = agent.inventory.pop(0)
                    reward = max(data[t][0] - bought_price[0], 0)
                    total_profit += data[t][0] - bought_price[0]
                    print(f"Step {t}: Sell: {formatPrice(data[t][0], scaler)} | Profit: {formatPrice(data[t][0] - bought_price[0], scaler)}")
                
                done = True if t == l - 1 else False
                agent.memory.append((state, action, reward, next_state, done))
                state = next_state
                
                if done:
                    print("##############")
                    print(f"Episode {e+1} completed")
                    print(f"Total Profit: {formatPrice(total_profit, scaler)}")
                    print(f"Final epsilon: {agent.epsilon:.4f}")
                    print("##############")
                
                if len(agent.memory) > batch_size:
                    agent.expReplay(batch_size)
            
            epoch_model_name = model_name.replace('.pth', f'_epoch_{e+1}.pth')
            torch.save(agent.model.state_dict(), epoch_model_name)
            print(f"Model saved to {epoch_model_name} after episode {e+1}")
            
            torch.save(agent.model.state_dict(), model_name)
            print(f"Model also saved as {model_name}")
    
    def evaluate_model(stock_data, window_size=10, model_name="model_sp500.pth", save_plot=False):
        data = stock_data
        l = len(data) - 1
        agent = Agent(window_size * 2, is_eval=True, model_name=model_name)
        
        print(f"Evaluating on {l} data points")
        
        state = getState(data, 0, window_size + 1)
        total_profit = 0
        agent.inventory = []
        
        states_buy = []
        states_sell = []
        
        for t in range(l):
            action = agent.act(state)
            
            if t + 1 < len(data):
                next_state = getState(data, t + 1, window_size + 1)
            else:
                next_state = state
            
            if action == 1:
                agent.inventory.append(data[t])
                states_buy.append(t)
                print(f"Step {t}: Buy: {formatPrice(data[t][0], scaler)}")
            
            elif action == 2 and len(agent.inventory) > 0:
                bought_price = agent.inventory.pop(0)
                reward = max(data[t][0] - bought_price[0], 0)
                total_profit += data[t][0] - bought_price[0]
                states_sell.append(t)
                print(f"Step {t}: Sell: {formatPrice(data[t][0], scaler)} | Profit: {formatPrice(data[t][0] - bought_price[0], scaler)}")
            
            state = next_state
        
        print("------------------------------------------")
        print(f"Total Profit: {formatPrice(total_profit, scaler)}")
        print(f"Total Buy actions: {len(states_buy)}")
        print(f"Total Sell actions: {len(states_sell)}")
        print("------------------------------------------")
        
        model_basename = os.path.basename(model_name).replace('.pth', '')
        plot_filename = f"trading_behavior_{model_basename}.png"
        
        plot_behavior(data, states_buy, states_sell, total_profit, save_plot=save_plot, filename=plot_filename)
        return total_profit, states_buy, states_sell
    
    choice = input("Do you want to train (t) or evaluate (e) the model? [t/e]: ").lower()
    
    if choice == 't':
        train_model(train_data, window_size, batch_size, episodes, model_name)
        
        eval_choice = input("Do you want to evaluate the trained model? [y/n]: ").lower()
        if eval_choice == 'y':
            save_plot_choice = input("Do you want to save the evaluation plot? [y/n]: ").lower()
            save_plot = (save_plot_choice == 'y')
            profit, states_buy, states_sell = evaluate_model(test_data, window_size, model_name, save_plot)
            print(f"Model trained and evaluated successfully!")
            print(f"Total profit on test data: {formatPrice(profit, scaler)}")
    
    elif choice == 'e':
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
            
        if not os.path.exists(selected_model):
            print(f"Model file {selected_model} not found. Please train the model first.")
            return
            
        save_plot_choice = input("Do you want to save the evaluation plot? [y/n]: ").lower()
        save_plot = (save_plot_choice == 'y')
        profit, states_buy, states_sell = evaluate_model(test_data, window_size, selected_model, save_plot)
        print(f"Model evaluated successfully!")
        print(f"Total profit on test data: {formatPrice(profit, scaler)}")
    
    else:
        print("Invalid choice. Please run the script again and enter 't' for training or 'e' for evaluation.")

if __name__ == "__main__":
    main()
