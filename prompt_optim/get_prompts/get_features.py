import requests
import json
from tqdm import tqdm
import os

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"

def prompt_to_features(prompt: str) -> dict:
    system_prompt = """ You are a prompt analysis engine.
                        
                        Your task is to convert a given prompt into a structured feature representation.
                        
                        Rules:
                        - Extract intent, not surface wording
                        - Be concise but precise
                        - Do NOT invent information
                        - Output MUST be valid JSON
                        - Follow the schema exactly
                        - **Do NOT wrap the output in markdown or code blocks**
                        - **Output JSON only, starting with { and ending with }**
                        
                        Schema:
                        {
                          "role": string,
                          "task": string,
                          "input_assumption": string,
                          "constraints": array of strings,
                          "reasoning_style": string,
                          "output_format": string,
                          "self_check": string
                        }
                        """

    user_prompt = f"""
                    Now analyze the following prompt:
                    \"\"\"
                    {prompt}
                    \"\"\"
                    """

    payload = {
        "model": "deepseek-chat",
        "temperature": 0.0,  # ⚠️ 必须低温，保证结构稳定
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    }

    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }

    response = requests.post(
        DEEPSEEK_API_URL,
        headers=headers,
        data=json.dumps(payload),
        timeout=30
    )

    response.raise_for_status()

    content = response.json()["choices"][0]["message"]["content"]

    try:
        features = json.loads(content)
    except json.JSONDecodeError:
        raise ValueError("Model output is not valid JSON:\n" + content)

    return features

def get_features(prompts):
    cnt=0
    structured_population=[]
    print("--------------正在提取提示词特征----------------")
    while True:
        try:
            for i,p in tqdm(enumerate(prompts), total=len(prompts)):  # prompts: List[str]
                if i<cnt:
                    continue
                features = prompt_to_features(p)
                structured_population.append({
                    "raw_prompt": p,
                    "features": features,
                    "score": None  # 之后再填
                })
                cnt+=1
            return structured_population

        except Exception:
            continue