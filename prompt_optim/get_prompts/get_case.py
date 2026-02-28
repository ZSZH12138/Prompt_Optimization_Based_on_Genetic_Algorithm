import os
import json

def get_case(text,title,author):
    word_count=len(text)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    des_path=os.path.join(current_dir, "..", "data","eval_cases.json")
    data={
        "case_id": "case1",
        "title": title,
        "author": author,
        "text":text,
        "word_count":word_count
    }
    data=[data]
    with open(des_path,'w',encoding="utf-8") as f:
        json.dump(data,f)