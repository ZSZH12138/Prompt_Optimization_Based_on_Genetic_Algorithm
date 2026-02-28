import json
import os
import shutil
from prompt_optim.get_prompts.get_prompts import generate_summary_prompts
from prompt_optim.get_prompts.get_case import get_case
from prompt_optim.get_prompts.get_features import get_features
from prompt_optim.iterate.get_children import generate_children
from prompt_optim.iterate.save_initial_population import save_initial_population
from prompt_optim.iterate.select_prompt import select_next_population
from prompt_optim.iterate.tournament import select_parents_limited
from prompt_optim.score.score_structured_features import score_structured_features


class PromptOptimizerPipeline:
    def __init__(self, requirement, text, author, title):
        self.requirement = requirement
        self.text = text
        self.author=author
        self.title=title

    def run(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(current_dir,  "data")
        if os.path.exists(data_dir):
            shutil.rmtree(data_dir)
        os.mkdir(data_dir)
        yield "开始进行优化步骤"
        prompts=self._get_prompts_and_cases()
        yield "已经生成相关提示词以及文本字典"
        features=self._get_features(prompts)
        yield "提示词提取完成"
        structured=self._score(features)
        yield "已经完成初始评分 形成种群"
        save_initial_population(structured)
        data_path = os.path.join(current_dir, "data", "selected_prompts1.json")
        for i in range(1, 11):
            print(f"----------------正在进行第{i}代种群的繁殖------------------------------")
            with open(data_path, 'r', encoding="utf-8") as f:
                data = json.load(f)
            parents = select_parents_limited(data, 5)
            children = generate_children(parents, 7)
            select_next_population(data, children, i)
            yield f"已经完成第{i}代繁殖"
            data_path = os.path.join(current_dir, "data", f"selected_prompts{i + 1}.json")


    def _get_prompts_and_cases(self):
        prompts=generate_summary_prompts(self.requirement)
        get_case(self.text,self.title,self.author)
        return prompts

    def _get_features(self,prompts):
        features=get_features(prompts)
        return features

    def _score(self,features):
        structured=score_structured_features(features)
        return structured
