import os
import json

def select_next_population(orig_population,children,cnt,population_size=7):
    # 合并池
    pool = orig_population + children

    # 按 score 排序（高 → 低）
    pool_sorted = sorted(
        pool,
        key=lambda x: x["score"],
        reverse=True
    )

    next_population = []

    # 全局精英
    elites = pool_sorted[:4]
    next_population.extend(elites)

    # 子代保留
    children_sorted = sorted(
        children,
        key=lambda x: x["score"],
        reverse=True
    )

    for c in children_sorted:
        if len(next_population) >= 6:
            break
        if c not in next_population:
            next_population.append(c)

    # 多样性位
    remaining = [
        x for x in pool
        if x not in next_population
    ]

    if remaining and len(next_population) < population_size:
        diverse = select_most_diverse(
            remaining,
            next_population
        )
        next_population.append(diverse)

    # ===== 兜底（理论上不会触发）=====
    if len(next_population) < population_size:
        for x in pool_sorted:
            if x not in next_population:
                next_population.append(x)
            if len(next_population) == population_size:
                break

    current_dir = os.path.dirname(os.path.abspath(__file__))
    des_path= os.path.join(current_dir, "..", "data", f"selected_prompts{cnt+1}.json")
    with open(des_path,'w')as f:
        json.dump(next_population,f)

def select_most_diverse(candidates, selected):
    """
    从 candidates 中选一个
    与 selected 集合差异最大的个体
    """
    best = None
    best_dist = -1

    for cand in candidates:
        d = sum(
            feature_distance(cand["features"], s["features"])
            for s in selected
        )
        if d > best_dist:
            best_dist = d
            best = cand

    return best

def feature_distance(fea_a, fea_b):
    """
    计算两个 feature dict 的粗粒度差异
    """
    dist = 0

    for k in fea_a.keys():
        if k not in fea_b:
            dist += 1
            continue

        va, vb = fea_a[k], fea_b[k]

        if isinstance(va, list) and isinstance(vb, list):
            dist += len(set(va) ^ set(vb))
        else:
            dist += int(va != vb)

    return dist