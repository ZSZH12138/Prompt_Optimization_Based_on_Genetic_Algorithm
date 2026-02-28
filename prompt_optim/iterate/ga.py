import os
import json
from prompt_optim.iterate.tournament import select_parents_limited
from prompt_optim.iterate.get_children import generate_children
from prompt_optim.iterate.select_prompt import select_next_population

def ga():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path=os.path.join(current_dir, "..", "data", "selected_prompts1.json")
    for i in range(1,11):
        with open(data_path,'r',encoding="utf-8")as f:
            data=json.load(f)
        parents=select_parents_limited(data,5)
        children=generate_children(parents,7)
        select_next_population(data,children,i)
        print(f"繁殖完成")
        data_path=os.path.join(current_dir, "..", "data", f"selected_prompts{i+1}.json")