import os
import random
import copy
from openai import OpenAI
from prompt_optim.score.score_structured_features import score_structured_features

random.seed(19)

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/v1"
)

def crossover_features(fea_a, fea_b):
    """
    对两个 parent 的 features 做基因级杂交
    """
    child_features = {}

    for key in fea_a.keys():
        val_a = fea_a[key]
        val_b = fea_b.get(key)

        # 如果其中一个没有这个基因，直接用现有的
        if val_b is None:
            child_features[key] = copy.deepcopy(val_a)
            continue

        # list 类型（比如 constraints）：合并
        if isinstance(val_a, list) and isinstance(val_b, list):
            child_features[key]=crossover_constraints(val_a,val_b)

        # 其他类型（字符串等）：随机继承
        else:
            child_features[key] = copy.deepcopy(
                random.choice([val_a, val_b])
            )

    for key in fea_b.keys():
        if key in child_features.keys():
            continue
        val_b=fea_b[key]
        child_features[key] = copy.deepcopy(val_b)

    return child_features

def crossover_constraints(cons_a,cons_b,p_keep=0.5):
    max_constraints=max(len(cons_a),len(cons_b))
    # 统一基因池（但不等于 merge）
    gene_pool = list(set(cons_a + cons_b))

    child_constraints = []

    for c in gene_pool:
        if random.random() < p_keep:
            child_constraints.append(copy.deepcopy(c))

    # 防止空约束（这是工程兜底）
    if not child_constraints and gene_pool:
        child_constraints.append(random.choice(gene_pool))

    # 长度裁剪（防爆）
    if max_constraints is not None and len(child_constraints) > max_constraints:
        child_constraints = random.sample(child_constraints, max_constraints)

    return child_constraints

def render_raw_prompt(features):
    """
    将 features 渲染回 raw_prompt
    你之后可以不断升级这个函数
    """
    prompt = f"""You are a {features['role']}.

    Your task:
    {features['task']}
    
    Assumptions:
    {features['input_assumption']}
    
    Constraints:
    """

    for c in features.get("constraints", []):
        prompt += f"- {c}\n"

    prompt += f"""
    Reasoning style:
    {features['reasoning_style']}
    
    Output format:
    {features['output_format']}
    
    Self check:
    {features['self_check']}
    """
    return prompt.strip()

def breed(parent_a, parent_b):
    """
    两个父代 → 一个子代
    """
    child_features = crossover_features(
        parent_a["features"],
        parent_b["features"]
    )

    child_features=mutate_features(child_features)

    child_raw_prompt = render_raw_prompt(child_features)

    child = {
        "raw_prompt": child_raw_prompt,
        "features": child_features,
        # 子代还没评估
        "score": None,
        "scores": None
    }

    return child

def generate_children(parents, num_children):
    children = []

    for _ in range(num_children):
        p1, p2 = random.sample(parents, 2)
        child = breed(p1, p2)
        children.append(child)

    score_structured_features(children)

    return children

def mutate_features(features,mutation_rate=0.1):
    """
    对一个 child 的 features 进行变异
    mutation_rate: 单个基因发生变异的概率
    """
    mutated = copy.deepcopy(features)

    for key, val in features.items():
        if random.random() > mutation_rate:
            continue  # 不发生变异

        # ===== 约束：list 类型 =====
        if key == "constraints" and isinstance(val, list):
            mutated[key] = mutate_constraints(val,features)

        # ===== 文本型 =====
        elif isinstance(val, str):
            mutated[key] = mutate_text_field(key, val,features)

    return mutated

def mutate_constraints(constraints,features,p_add=0.3,p_remove=0.3,p_shuffle=0.4):
    cons = copy.deepcopy(constraints)

    # 删除一个
    if cons and random.random() < p_remove:
        cons.pop(random.randrange(len(cons)))

    # 打乱顺序（低风险但有用）
    if len(cons) > 1 and random.random() < p_shuffle:
        random.shuffle(cons)

    if random.random() < p_add:
        try:
            new_c = llm_generate_constraint(features)
            if new_c and new_c not in cons:
                cons.append(new_c)
        except Exception as e:
            # LLM 失败 = 本次不加，不兜底
            pass

    return cons

def llm_generate_constraint(features,temperature=0.7):
    """
    使用 DeepSeek 生成一个新的 constraint
    """
    system_prompt = (
        "You are an expert prompt engineer. "
        "Your task is to generate ONE concise, actionable constraint "
        "that improves prompt quality without changing the task."
    )

    user_prompt = f"""
    Task:
    {features.get("task")}
    
    Existing constraints:
    {features.get("constraints", [])}
    
    Generate ONE new constraint.
    Requirements:
    - Do NOT repeat existing constraints
    - Do NOT change the task
    - Be concise (one sentence)
    - Return ONLY the constraint text
    """

    resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=temperature
    )

    return resp.choices[0].message.content.strip()

def mutate_text_field(key,original_text,features,temperature=0.7):
    system_prompt = (
        "You are an expert in prompt engineering and instruction tuning. "
        "You perform controlled mutations on prompt components."
    )

    user_prompt = f"""
        We are performing a genetic mutation on ONE field of a prompt.
        
        Field name:
        {key}
        
        Original content:
        {original_text}
        
        Task (for context, do NOT change):
        {features.get("task")}
        
        Requirements:
        - Keep the SAME semantic intent
        - Only change STYLE or phrasing
        - Do NOT introduce new requirements
        - Do NOT change the task
        - Output ONE concise sentence
        - Return ONLY the mutated text
        """

    resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=temperature
    )

    return resp.choices[0].message.content.strip()