import pickle
import os
from pathlib import Path

BASE_DIR = (os.path.join(os.path.dirname(os.path.abspath(__file__)),"../")) 

def load_data(folder = "tick"):
    file_path = os.path.join(BASE_DIR, f"data/{folder}")
    folder_path = Path(file_path)
    files = [f.name for f in folder_path.iterdir() if f.is_file()]
    data_dict = {}
    for filename in files:
        name, ext = os.path.splitext(filename)
        with open(f"{folder_path}/{filename}", 'rb') as f:
            data_dict[name] = pickle.load(f)
    return data_dict

if __name__ == "__main__":  
    load_data()