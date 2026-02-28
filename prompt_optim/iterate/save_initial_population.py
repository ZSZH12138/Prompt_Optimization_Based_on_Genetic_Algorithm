import json
import os

def save_initial_population(data):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    des_path=os.path.join(current_dir, "..", "data",f"selected_prompts1.json")
    with open(des_path,'w') as f:
        json.dump(data,f)