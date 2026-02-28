import os
import requests
from tqdm import tqdm

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"
MODEL_NAME = "deepseek-chat"   # 常用模型名

def generate_summary_prompts(user_requirement: str, n: int = 7):
    """
    根据用户需求，每次调用一次 API 生成 1 个总结类 Prompt 模板
    共生成 n 个，返回 List[str]
    """

    print("-----------开始生成提示词模板------------")

    STYLE_HINTS = [
        "concise and minimal",
        "structured with headings",
        "bullet-point based",
        "analytical and critical",
        "academic and formal",
        "executive summary style",
        "step-by-step explanation",
    ]

    system_prompt = (
        "You are a professional Prompt Engineer.\n"
        "Your task is to design high-quality prompt template for text summarization.\n"
        "The prompts should be general, reusable, and suitable for long-form texts."
    )

    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }

    prompts = []

    for i in tqdm(range(n)):
        style_hint = STYLE_HINTS[i % len(STYLE_HINTS)]

        user_prompt = f"""
        User requirement:
        {user_requirement}
        
        Style preference:
        {style_hint}
        
        Rules:
        - Generate ONE summarization prompt template.
        - The output should be a TEMPLATE, not a specific summary.
        - Focus on a single summarization style.
        - Do NOT summarize any text.
        - Output the prompt template directly, without numbering or explanation.
        """

        payload = {
            "model": MODEL_NAME,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.8,
        }

        response = requests.post(
            DEEPSEEK_API_URL,
            json=payload,
            headers=headers
        )
        response.raise_for_status()

        content = response.json()["choices"][0]["message"]["content"].strip()
        prompts.append(content)

    return prompts