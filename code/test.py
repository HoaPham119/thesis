import pickle
import pandas as pd
import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

with open('data/orderbook/ACB.pkl', 'rb') as f:
    data = pickle.load(f)
print(data)