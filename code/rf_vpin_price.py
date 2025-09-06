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
    price = scaler.inverse_transform([[n, 0]])[0][0]
    return ("-$" if price < 0 else "$" ) + "{0:.2f}".format(abs(price))

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def getState(data, t, n):
    d = t - n + 1
    if d < 0:
        block = np.array([data[0]] * (-d) + data[0:t + 1].tolist())
    else:
        block = data[d:t + 1]

    res = []
    for i in range(len(block) - 1):
        price_diff = block[i + 1][0] - block[i][0]
        vpin_diff = block[i + 1][1] - block[i][1]
        res.extend([sigmoid(price_diff), sigmoid(vpin_diff)])

    while len(res) < (n - 1) * 2:
        res.append(0)

    return np.array([res])

def plot_behavior(data_input, states_buy, states_sell, profit, save_plot=False, filename=None):
    if save_plot and not os.path.exists('plots'):
        os.makedirs('plots')

    plt.figure(figsize=(15, 5))

    plt.plot(data_input[:, 0], color='r', lw=2., label='Price')
    plt.plot(data_input[:, 1], color='b', lw=2., label='VPIN')

    plt.plot(states_buy, data_input[states_buy, 0], '^', markersize=10, color='m', label='Buying signal')
    plt.plot(states_sell, data_input[states_sell, 0], 'v', markersize=10, color='k', label='Selling signal')

    plt.title(f'Total gains: {profit:.2f}')
    plt.legend()

    if save_plot:
        if filename is None:
            filename = f'plots/trading_behavior_profit_{profit:.2f}.png'
        else:
            filename = f'plots/{filename}'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {filename}")

    plt.show()

# DQN Network
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 64) # ánh xạ từ đầu vào state_size sang 64 neuron.
        self.fc2 = nn.Linear(64, 32) # từ 64 neuron sang 32 neuron.
        self.fc3 = nn.Linear(32, action_size) #từ 32 neuron ra số lượng hành động (action_size), là các Q-values cho từng hành động.

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# Agent Class
class Agent:
    def __init__(self, state_size, is_eval=False, model_name=""):
        self.state_size = state_size
        self.action_size = 3
        self.memory = deque(maxlen=1000)
        self.inventory = []
        self.model_name = model_name
        self.is_eval = is_eval
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.model = DQN(state_size, self.action_size).to(self.device)

        if is_eval and os.path.exists(model_name):
            self.model.load_state_dict(torch.load(model_name, map_location=self.device))
            self.model.eval()

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
        mini_batch = random.sample(self.memory, batch_size)

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
        dataset = pd.read_csv('RL_data/VCBVPIN.csv', index_col=0).dropna()
    except FileNotFoundError:
        print("Data file not found. Please ensure the VCBVPIN.csv file is in the 'data' directory.")
        return

    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)

    # Prepare data
    dataset = dataset.tail(200)
    price_vpin_data = dataset[["Price", "VPIN"]].dropna().values

    train_size = int(len(price_vpin_data) * 0.8)
    train_data = price_vpin_data[:train_size]
    test_data = price_vpin_data[train_size:]

    # Scale data
    scaler = MinMaxScaler()
    train_data = scaler.fit_transform(train_data)
    test_data = scaler.transform(test_data)

    print(f"Data loaded: {len(price_vpin_data)} total points")
    print(f"Training data: {len(train_data)} points")
    print(f"Testing data: {len(test_data)} points")

    window_size, batch_size, episodes = 10, 32, 5
    model_name = "models/model_sp500.pth"

    def train_model(stock_data, window_size, batch_size, episodes, model_name):
        agent = Agent(window_size * 2)
        data_length = len(stock_data) - 1

        print(f"Training on {data_length} data points for {episodes} episodes")

        for episode in range(episodes):
            state = getState(stock_data, 0, window_size + 1)
            total_profit, agent.inventory = 0, []

            print(f"Running episode {episode + 1}/{episodes}")

            for t in range(data_length):
                action = agent.act(state)
                next_state = getState(stock_data, t + 1, window_size + 1) if t + 1 < len(stock_data) else state

                reward = 0
                if action == 1:
                    agent.inventory.append(stock_data[t])
                    print(f"Step {t}: Buy: {formatPrice(stock_data[t][0], scaler)}")
                elif action == 2 and agent.inventory:
                    bought_price = agent.inventory.pop(0)
                    reward = max(stock_data[t][0] - bought_price[0], 0)
                    total_profit += stock_data[t][0] - bought_price[0]
                    price_diff_display = float(formatPrice(stock_data[t][0], scaler).replace("$", "").replace("-", "")) - float(formatPrice(bought_price[0], scaler).replace("$", "").replace("-", ""))
                    print(f"Step {t}: Sell: {formatPrice(stock_data[t][0], scaler)} | Profit: {formatPrice(stock_data[t][0], scaler)} - {formatPrice(bought_price[0], scaler)} = {price_diff_display:.2f}")

                done = t == data_length - 1
                agent.memory.append((state, action, reward, next_state, done))
                state = next_state

                if done:
                    print("##############")
                    print(f"Episode {episode + 1} completed")
                    print(f"Total Profit: {formatPrice(total_profit, scaler)}")
                    print(f"Final epsilon: {agent.epsilon:.4f}")
                    print("##############")

                if len(agent.memory) > batch_size:
                    agent.expReplay(batch_size)

            epoch_model_name = model_name.replace('.pth', f'_epoch_{episode + 1}.pth')
            torch.save(agent.model.state_dict(), epoch_model_name)
            print(f"Model saved to {epoch_model_name} after episode {episode + 1}")

            torch.save(agent.model.state_dict(), model_name)
            print(f"Model also saved as {model_name}")

    def evaluate_model(stock_data, window_size, model_name, save_plot):
        agent = Agent(window_size * 2, is_eval=True, model_name=model_name)
        data_length = len(stock_data) - 1
        state = getState(stock_data, 0, window_size + 1)
        total_profit, agent.inventory = 0, []
        
        states_buy, states_sell = [], []

        print(f"Evaluating on {data_length} data points")

        for t in range(data_length):
            action = agent.act(state)
            next_state = getState(stock_data, t + 1, window_size + 1) if t + 1 < len(stock_data) else state

            if action == 1:
                agent.inventory.append(stock_data[t])
                states_buy.append(t)
                print(f"Step {t}: Buy: {formatPrice(stock_data[t][0], scaler)}")
            elif action == 2 and agent.inventory:
                bought_price = agent.inventory.pop(0)
                total_profit += stock_data[t][0] - bought_price[0]
                states_sell.append(t)
                price_diff_display = float(formatPrice(stock_data[t][0], scaler).replace("$", "").replace("-", "")) - float(formatPrice(bought_price[0], scaler).replace("$", "").replace("-", ""))
                print(f"Step {t}: Sell: {formatPrice(stock_data[t][0], scaler)} | Profit: {formatPrice(stock_data[t][0], scaler)} - {formatPrice(bought_price[0], scaler)} = {price_diff_display:.2f}")

            state = next_state

        print("------------------------------------------")
        print(f"Total Profit: {formatPrice(total_profit, scaler)}")
        print(f"Total Buy actions: {len(states_buy)}")
        print(f"Total Sell actions: {len(states_sell)}")
        print("------------------------------------------")

        plot_filename = f"trading_behavior_{os.path.basename(model_name).replace('.pth', '')}.png"
        plot_behavior(stock_data, states_buy, states_sell, total_profit, save_plot=save_plot, filename=plot_filename)

    choice = input("Do you want to train (t) or evaluate (e) the model? [t/e]: ").lower()

    if choice == 't':
        train_model(train_data, window_size, batch_size, episodes, model_name)
        if input("Do you want to evaluate the trained model? [y/n]: ").lower() == 'y':
            save_plot = input("Do you want to save the evaluation plot? [y/n]: ").lower() == 'y'
            evaluate_model(test_data, window_size, model_name, save_plot)
    elif choice == 'e':
        evaluate_model(test_data, window_size, model_name, input("Do you want to save the evaluation plot? [y/n]: ").lower() == 'y')
    else:
        print("Invalid choice. Please run again and choose 't' or 'e'.")

if __name__ == "__main__":
    main()
