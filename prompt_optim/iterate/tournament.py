import random

random.seed(19)

def tournament_select(population, k=3):
    """
    从 population 中用锦标赛选择 1 个个体
    :param population: List[Dict]，每个 dict 含有 "score"
    :param k: 锦标赛规模（每次抽几个）
    :return: 选中的个体（dict）
    """
    # 随机抽 k 个（不放回）
    candidates = random.sample(population, k)

    # 按 score 选最优
    winner = max(candidates, key=lambda x: x["score"])
    return winner

def select_parents_limited(population, num_parents, k=3, max_repeat=2):
    parents = []
    d={}

    while len(parents) < num_parents:
        p = tournament_select(population, k)
        raw_prompt=p["raw_prompt"]
        if raw_prompt in d.keys() and d[raw_prompt]>=max_repeat:
            continue
        elif raw_prompt in d.keys():
            d[raw_prompt]+=1
        else:
            d[raw_prompt]=1
        parents.append(p)
    return parents