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

filename_inputs = [
 'VCB',
 ]

def formatPrice(n, scaler):
    price = scaler.inverse_transform([[n, 0]])[0][0]
    return ("-$" if price < 0 else "$") + "{0:.2f}".format(abs(price))

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
        res.extend([price_diff, vpin_diff])

    while len(res) < (n - 1) * 2:
        res.append(0)

    return np.array([res])


def plot_behavior(data_input, states_buy, states_sell, profit, scaler, save_plot=False, filename=None):
    if save_plot and not os.path.exists(f'plots/deulindqn/{filename_input}/'):
        os.makedirs(f'plots/deulindqn/{filename_input}')

    price_vpin_inverse = scaler.inverse_transform(data_input)
    fig, ax1 = plt.subplots(figsize=(15, 5))

    color_price = 'tab:red'
    color_vpin = 'tab:blue'

    ax1.set_xlabel('Time')
    ax1.set_ylabel('Price', color=color_price)
    ax1.plot(price_vpin_inverse[:, 0], color=color_price, lw=2., label='Price')
    ax1.tick_params(axis='y', labelcolor=color_price)
    ax1.plot(states_buy, price_vpin_inverse[states_buy, 0], '^', markersize=10, color='m', label='Buying signal')
    ax1.plot(states_sell, price_vpin_inverse[states_sell, 0], 'v', markersize=10, color='k', label='Selling signal')

    ax2 = ax1.twinx()
    ax2.set_ylabel('VPIN', color=color_vpin)
    ax2.plot(price_vpin_inverse[:, 1], color=color_vpin, lw=2., label='VPIN')
    ax2.tick_params(axis='y', labelcolor=color_vpin)

    # Thêm text vào biểu đồ
    ax1.text(0.01, 0.95, f"Total Buy actions: {len(states_buy)}", transform=ax1.transAxes, fontsize=10, verticalalignment='top')
    ax1.text(0.01, 0.90, f"Total Sell actions: {len(states_sell)}", transform=ax1.transAxes, fontsize=10, verticalalignment='top')

    fig.suptitle(f'Total gains: ${profit:.2f}')
    fig.tight_layout()

    if save_plot:
        if filename is None:
            filename = f'plots/deulindqn/{filename_input}/trading_behavior_profit_{profit:.2f}.png'
        else:
            filename = f'plots/deulindqn/{filename_input}/{filename}'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {filename}")

    plt.legend()
    plt.show(block=False)
    return

class DuelingDQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DuelingDQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.value_stream = nn.Linear(32, 1)
        self.advantage_stream = nn.Linear(32, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        return value + (advantage - advantage.mean(dim=1, keepdim=True))

class Agent:
    def __init__(self, state_size, is_eval=False, model_name=""):
        self.state_size = state_size
        self.action_size = 3
        self.memory = deque(maxlen=50000)
        self.inventory = []
        self.model_name = model_name
        self.is_eval = is_eval
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.999
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.model = DuelingDQN(state_size, self.action_size).to(self.device)
        self.target_model = DuelingDQN(state_size, self.action_size).to(self.device)
        self.update_target_model()
        if is_eval and os.path.exists(model_name):
            self.model.load_state_dict(torch.load(model_name, map_location=self.device))
            self.model.eval()
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001)

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def act(self, state):
        state_tensor = torch.FloatTensor(state).to(self.device)
        if state_tensor.dim() == 1:
            state_tensor = state_tensor.unsqueeze(0)
        if not self.is_eval and random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        with torch.no_grad():
            q_values = self.model(state_tensor)
        return torch.argmax(q_values).item()

    def expReplay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        mini_batch = random.sample(self.memory, batch_size)

        for state, action, reward, next_state, done in mini_batch:
            state_tensor = torch.FloatTensor(state).to(self.device)
            next_state_tensor = torch.FloatTensor(next_state).to(self.device)
            if state_tensor.dim() == 1:
                state_tensor = state_tensor.unsqueeze(0)
            if next_state_tensor.dim() == 1:
                next_state_tensor = next_state_tensor.unsqueeze(0)

            target = reward
            if not done:
                with torch.no_grad():
                    q_next_online = self.model(next_state_tensor)
                    best_action = torch.argmax(q_next_online).item()
                    q_next_target = self.target_model(next_state_tensor)
                    target = reward + self.gamma * q_next_target[0, best_action].item()

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

def main(filename_input):
    
    try:
        dataset = pd.read_csv(f'RL_data/{filename_input}VPIN.csv', index_col=0).dropna()
    except FileNotFoundError:
        print(f"Data file not found. Please ensure the {filename_input}VPIN.csv file is in the 'RL_data' directory.")
        return

    os.makedirs('data', exist_ok=True)
    os.makedirs(f'models/deulindqn/{filename_input}', exist_ok=True)

    price_vpin_data = dataset[["Price", "VPIN"]].dropna().values
    train_size = int(len(price_vpin_data) * 0.8)
    train_data = price_vpin_data[:train_size]
    test_data = price_vpin_data[train_size:]
    scaler = MinMaxScaler()
    train_data = scaler.fit_transform(train_data)
    test_data = scaler.transform(test_data)

    window_size, batch_size, episodes = 10, 32, 5
    model_name = f"models/deulindqn/{filename_input}/model_sp500.pth"

    def train_model(stock_data):
        agent = Agent(window_size * 2)
        data_length = len(stock_data) - 1
        print(f"Training on {data_length} data points for {episodes} episodes")
        transaction_fee = 0.001  # phí giao dịch 0.1%
        min_trade_interval = 3   # không giao dịch quá sát nhau
        penalty_close_interval = 0.5 

        for episode in range(episodes):
            state = getState(stock_data, 0, window_size + 1)
            total_profit, agent.inventory = 0, []
            last_trade_step = -10 
            print(f"Running episode {episode + 1}/{episodes}")

            for t in range(data_length):
                action = agent.act(state)
                next_state = getState(stock_data, t + 1, window_size + 1) if t + 1 < len(stock_data) else state

                reward = 0
                is_trade = False
                if action == 1:
                    agent.inventory.append(stock_data[t])
                    print(f"Step {t}: Buy: {formatPrice(stock_data[t][0], scaler)}")
                elif action == 2 and agent.inventory:
                    bought_price = agent.inventory.pop(0)
                    sell_price = formatPrice(stock_data[t][0], scaler)
                    buy_price = formatPrice(bought_price[0], scaler)
                    profit = float(sell_price.replace("$", "").replace("-", "")) - float(buy_price.replace("$", "").replace("-", ""))
                    # reward = profit
                    reward = profit - transaction_fee
                    total_profit += profit
                    is_trade = True
                    print(f"Step {t}: Sell: {sell_price} | Profit: {profit:.2f}")
                if is_trade and (t - last_trade_step < min_trade_interval):
                    reward -= penalty_close_interval
                    print(f"Step {t}: Penalized for trading too soon (-{penalty_close_interval})")
                if is_trade:
                    last_trade_step = t
                done = t == data_length - 1
                agent.memory.append((state, action, reward, next_state, done))
                state = next_state

                if len(agent.memory) > batch_size:
                    agent.expReplay(batch_size)

                if done:
                    final_price = stock_data[-1][0]
                    while agent.inventory:
                        bought_price = agent.inventory.pop(0)
                        sell_price = formatPrice(final_price, scaler)
                        buy_price = formatPrice(bought_price[0], scaler)
                        profit = float(sell_price.replace("$", "").replace("-", "")) - float(buy_price.replace("$", "").replace("-", ""))
                        reward = profit - transaction_fee
                        total_profit += profit
                        print(f"End of episode {episode + 1}: Sell remaining position: {sell_price} | Profit: {profit:.2f}")
                    print("##############")
                    print(f"Episode {episode + 1} completed")
                    print(f"Total Profit: ${total_profit:.2f}")
                    print(f"Final epsilon: {agent.epsilon:.4f}")
                    print("##############")

            agent.update_target_model()
            epoch_model_name = model_name.replace('.pth', f'_epoch_{episode + 1}.pth')
            torch.save(agent.model.state_dict(), epoch_model_name)
            print(f"Model saved to {epoch_model_name} after episode {episode + 1}")
            torch.save(agent.model.state_dict(), model_name)
            print(f"Model also saved as {model_name}")

    def evaluate_model(stock_data, save_plot):
        model_dir = f'models/deulindqn/{filename_input}'
        model_files = [f for f in os.listdir(model_dir) if f.endswith('.pth')]
        if not model_files:
            print("No model files found in 'models/' directory.")
            return

        print("Available model files:")
        for i, f in enumerate(model_files):
            print(f"{i + 1}. {f}")

        model_choices=[ "1","2", "3","4","5","6", "7", "8", "9", "10"]
        for model_choice in model_choices:
            if not model_choice.isdigit() or not (1 <= int(model_choice) <= len(model_files)):
                print("Invalid choice.")
                return

            chosen_model = os.path.join(model_dir, model_files[int(model_choice) - 1])
            agent = Agent(window_size * 2, is_eval=True, model_name=chosen_model)
            transaction_fee = 0.001
            min_trade_interval = 3
            penalty_close_interval = 0.5
            last_trade_step = -10
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
                    if t - last_trade_step < min_trade_interval:
                        print(f"Step {t}: Penalized for trading too soon (-{penalty_close_interval})")
                    last_trade_step = t

                    print(f"Step {t}: Buy: {formatPrice(stock_data[t][0], scaler)}")
                elif action == 2 and agent.inventory:
                    bought_price = agent.inventory.pop(0)
                    sell_price = formatPrice(stock_data[t][0], scaler)
                    buy_price = formatPrice(bought_price[0], scaler)
                    profit = float(sell_price.replace("$", "").replace("-", "")) - float(buy_price.replace("$", "").replace("-", ""))
                    reward = profit - transaction_fee

                    # Phạt nếu trade quá gần
                    if t - last_trade_step < min_trade_interval:
                        reward -= penalty_close_interval
                        print(f"Step {t}: Penalized for trading too soon (-{penalty_close_interval})")

                    total_profit += profit
                    last_trade_step = t

                    states_sell.append(t)
                    print(f"Step {t}: Sell: {sell_price} | Profit: {profit:.2f}")

                state = next_state
            while agent.inventory:
                final_price = stock_data[-1][0]
                bought_price = agent.inventory.pop(0)
                sell_price = formatPrice(final_price, scaler)
                buy_price = formatPrice(bought_price[0], scaler)
                profit = float(sell_price.replace("$", "").replace("-", "")) - float(buy_price.replace("$", "").replace("-", ""))
                total_profit += profit
                states_sell.append(data_length - 1)
                print(f"Final Sell: {sell_price} | Profit: {profit:.2f}")

            print("------------------------------------------")
            print(f"Total Profit: ${total_profit:.2f}")
            print(f"Total Buy actions: {len(states_buy)}")
            print(f"Total Sell actions: {len(states_sell)}")
            print("------------------------------------------")

            plot_filename = f"trading_behavior_{os.path.basename(chosen_model).replace('.pth', '')}.png"
            plot_behavior(stock_data, states_buy, states_sell, total_profit, scaler, save_plot=save_plot, filename=plot_filename)

    # choice = input("Do you want to train (t) or evaluate (e) the model? [t/e]: ").lower()
    choice = "t"
    if choice == 't':
        train_model(train_data)
        save_plot=True
        evaluate_model(test_data, save_plot)
    else:
        print("Invalid choice. Please run again and choose 't' or 'e'.")

if __name__ == "__main__":
    for filename_input in filename_inputs:
        try:
            print(filename_input)
            main(filename_input)
        except:
            pass
