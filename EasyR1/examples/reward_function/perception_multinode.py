import re
from typing import Any
import random
import os
import time
from openai import OpenAI
import math
from typing import Optional
from mathruler.grader import grade_answer
from concurrent.futures import ThreadPoolExecutor, as_completed # 导入并行处理模块

openai_api_key = "empty"
# You need to update the URL of your judge model.
api_base_list = ["http://172.23.117.84:18901/v1", "http://172.23.114.7:18901/v1", "http://172.23.100.32:18901/v1", "http://172.23.9.136:18901/v1"]
model = "judge"
client_list = []
for api_base in api_base_list:
    client = OpenAI(
        api_key=openai_api_key,
        base_url=api_base,
    )
    client_list.append(client)

system_prompt = """
# 角色
你是一个判断专家，专注于判断输入的2个答案是否一致。

# 任务
你的任务是判断[模型回答]与[参考答案]是否一致，不需要思考[问题]真正的正确答案，以下是详细的判断步骤：
1. 问题理解：仔细阅读[问题]，并按照[判断标准]对问题进行分类，找出问题中包含多少个提问。
    - 问题中可能包含占位符“<image_*>”（其中“ * ”为数字），代表问题中有图片输入。注意：此类问题不用进行问题理解。
2. 答案对照：按照问题中的提问顺序，将[模型回答]与[参考答案]一一进行判断，对比是否一致。若存在一处不一致，则视为不一致。

# 判断标准

## 简答类
答案不唯一或不具体，需要根据材料、条件，自行组织语言回答问题或进行解答题目、证明结论。

### 简答（描述）
简答（描述）类问题，如材料题、写作题、图片描述等，[参考答案]与[模型回答]不需要完全一致，[模型答案]中包含[参考答案]中的要点，且表意一致（例如：参考答案为神态描写，则模型答案也必须为神态描写，否则判断为不一致），即判断为一致。

## 客观类
存在明确、客观的答案，在多个答案中选择正确答案或通过常识、计算推理直接给出答案，如科学知识、数学、物理等。
- 可以忽略答案组织形式（排版、分隔方式、是否使用Latex、大小写等）。例如：计算题只需要最终结果数值一致即可（例：“6棵”、“6”、“six”等视为一致）。

### 选择题
给出答案选项，答案选项可能用字母（A、B、C、D、...）、罗马数字（I、II、III、IV、...）或阿拉伯数字（1、2、3、4、...）标记，选择其中一个或多个选项。[模型回答]中的答案只需要与[参考答案]中对应的标记一致，即判断为一致。

### 分类（判断）题
判断[问题]中指定内容是否正确，或对[问题]中给出的元素根据指定类型进行分类。[模型回答]必须给出明确的判断（或分类），且必须与[参考答案]对应，否则判为不一致。

## 图片输入选择类
仅判断[模型回答]与[参考答案]是否一致。禁止分析问题中的图片序号。
- 可以忽略答案组织形式（如排版、分隔方式、是否使用Latex等）。

# 输出
结论输出：用一个词（是或否）在最后得出结论，格式为 \boxed{Yes} 或 \boxed{No}。

## 输出示例
<最终结果>
\boxed{Yes/No}
<\最终结果>


以下是输入内容：
"""

def get_prompt(_question, _answer, _pred):
    # 假设这个路径是正确的，并且文件存在
    judge_user_prompt = """
    [问题]:{question}
    [参考答案]:{answer}
    [模型回答]:{prediction}
    """

    full_prompt = judge_user_prompt.format(
            question=_question,
            answer=_answer,
            prediction=_pred
        )
    return full_prompt

# Metadata
REWARD_NAME = "perceptual"
REWARD_TYPE = "batch"

SUBSTITUTIONS = [
    ("an ", ""),
    ("a ", ""),
    (".$", "$"),
    ("\\$", ""),
    (r"\ ", ""),
    (" ", ""),
    ("mbox", "text"),
    (",\\text{and}", ","),
    ("\\text{and}", ","),
    ("\\text{m}", "\\text{}"),
]

REMOVED_EXPRESSIONS = [
    "square",
    "ways",
    "integers",
    "dollars",
    "mph",
    "inches",
    "hours",
    "km",
    "units",
    "\\ldots",
    "sue",
    "points",
    "feet",
    "minutes",
    "digits",
    "cents",
    "degrees",
    "cm",
    "gm",
    "pounds",
    "meters",
    "meals",
    "edges",
    "students",
    "childrentickets",
    "multiples",
    "\\text{s}",
    "\\text{.}",
    "\\text{\ns}",
    "\\text{}^2",
    "\\text{}^3",
    "\\text{\n}",
    "\\text{}",
    r"\mathrm{th}",
    r"^\circ",
    r"^{\circ}",
    r"\;",
    r",\!",
    "{,}",
    '"',
    "\\dots",
]

def extract_mcq_option(answer, question):
    """
    判断答案是否为多选题格式 (例如: A, A., (A), A xxx)
    同时排除 Any, Apple, Area 等普通单词。
    """
    if not check_question_is_mcq(question):
        return False

    if not isinstance(answer, str) or not answer:
        return False
    
    # 去除首尾空格
    text = answer.strip()
    
    pattern = r'^[ (\[]*([A-F])(?:(?=$)|[\.\)\]]|(?:[\:\-]\s+))'
    match = re.match(pattern, text)

    if match:
        return match.group(1)  # 返回捕获的字母
    return ""

def check_question_is_mcq(question):
    #多选题问题中一定有多个换行符
    newlines = re.findall(r'\n', question)
    #多选题中一定有选项A
    if "A" in question and len(newlines) >= 2:
        return True
    return False

def extract_first_option(text):
    if not text:
        return ""
    
    # 这里的正则逻辑：
    # 1. 优先匹配括号里的字母，如 (A)
    # 2. 其次匹配 A. 或 A) 或 孤立的 A
    # [A-Z] 表示大写字母
    
    # 模式 1: 匹配 (A)
    match = re.search(r'\(([A-Z])\)', text)
    if match:
        return match.group(1)
    
    # 模式 2: 匹配 A. 或 A) 或 A 开头后跟空格
    match = re.search(r'([A-Z])[\.\)\s]', text)
    if match:
        return match.group(1)

    # 模式 3: 直接找第一个出现的大写字母（兜底方案）
    match = re.search(r'([A-Z])', text)
    if match:
        return match.group(1)
        
    return ""


def normalize_final_answer(final_answer: str) -> str:
    """Normalize a final answer to a quantitative reasoning question.

    Args:
        final_answer: The answer string to normalize

    Returns:
        Normalized answer string
    """
    final_answer = final_answer.split("=")[-1]

    # Apply substitutions and removals
    for before, after in SUBSTITUTIONS:
        final_answer = final_answer.replace(before, after)
    for expr in REMOVED_EXPRESSIONS:
        final_answer = final_answer.replace(expr, "")

    # Extract and normalize LaTeX math
    final_answer = re.sub(r"(.*?)(\$)(.*?)(\$)(.*)", "$\\3$", final_answer)
    final_answer = re.sub(r"(\\text\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\textbf\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\overline\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\boxed\{)(.*)(\})", "\\2", final_answer)

    # Normalize shorthand TeX:
    #  \fracab -> \frac{a}{b}
    #  \frac{abc}{bef} -> \frac{abc}{bef}
    #  \fracabc -> \frac{a}{b}c
    #  \sqrta -> \sqrt{a}
    #  \sqrtab -> sqrt{a}b
    final_answer = re.sub(r"(frac)([^{])(.)", "frac{\\2}{\\3}", final_answer)
    final_answer = re.sub(r"(sqrt)([^{])", "sqrt{\\2}", final_answer)
    final_answer = final_answer.replace("$", "")

    # Normalize numbers
    if final_answer.replace(",", "").isdigit():
        final_answer = final_answer.replace(",", "")

    return final_answer.strip()


def check_format_match(response: str) -> bool:
    pattern = re.compile(r"<answer>.*?</answer>", re.DOTALL)
    format_match = re.search(pattern, response)
    return True if format_match else False

def string_match(response: str, ground_truth: str) -> float:
    match = re.findall(r"(?i)Answer\s*:\s*([^\n]+)", response)
    answer = match[-1] if match else "[INVALID]"
    if normalize_final_answer(answer) == normalize_final_answer(ground_truth):
        return True
    else:
        return False

def compute_score(reward_inputs: list[dict[str, Any]], format_weight: float = 0.0) -> list[dict[str, float]]:
    # 初始化一个与 reward_inputs 相同长度的列表，用于存储最终分数
    # 这样可以确保结果的顺序与输入一致
    # print("***************************")
    # print("reward_inputs:", reward_inputs[0])
    # print("reward_inputs:", reward_inputs[0].keys())
    # print("***************************")
    final_scores_list = [None] * len(reward_inputs) 
    
    # 用于存储需要并发执行的任务
    tasks_to_run = [] # 存储 (函数, 参数元组, 原始索引, format_score)
    
    for idx, reward_input in enumerate(reward_inputs):
        format_match = check_format_match(reward_input["response"])
        content_match = re.search(r"<answer>(.*?)</answer>", reward_input["response"])
        answer_text = content_match.group(1).strip() if content_match else reward_input["response"].strip()
        format_score = 1.0 if format_match else 0.0
        # 预先处理可以立即确定的分数
        if not answer_text:
            final_scores_list[idx] = {
                "overall": 0.0,
                "format": 0.0,
                "accuracy": 0.0,
            }
            continue 

        if answer_text and len(answer_text) >= 300:
            accuracy_score = 0.0
            final_scores_list[idx] = {
                "overall": (1 - format_weight) * accuracy_score + format_weight * format_score,
                "format": format_score,
                "accuracy": accuracy_score,
            }
            continue 

        if string_match(answer_text, reward_input["ground_truth"]):
            accuracy_score = 1.0
            final_scores_list[idx] = {
                "overall": (1 - format_weight) * accuracy_score + format_weight * format_score,
                "format": format_score,
                "accuracy": accuracy_score,
            }
            continue 
        if reward_input["ability"] == "counting":
            accuracy_score = counting_reward(answer_text, ground_truth=reward_input["ground_truth"])
            # print("accuracy_score", accuracy_score)
            final_scores_list[idx] = {
                "overall": (1 - format_weight) * accuracy_score + format_weight * format_score,
                "format": format_score,
                "accuracy": accuracy_score,
            }
            continue
        
        mcq_option = extract_mcq_option(reward_input["ground_truth"], reward_input["extra_info"]['question'])
        if mcq_option:
            answer = extract_first_option(answer_text)
            if mcq_option and mcq_option == answer:
                accuracy_score = 1.0
                final_scores_list[idx] = {
                    "overall": (1 - format_weight) * accuracy_score + format_weight * format_score,
                    "format": format_score,
                    "accuracy": accuracy_score,
                }
                continue

        if grade_answer(answer_text, reward_input["ground_truth"]):
            accuracy_score = 1.0
            final_scores_list[idx] = {
                "overall": (1 - format_weight) * accuracy_score + format_weight * format_score,
                "format": format_score,
                "accuracy": accuracy_score,
            }
            continue
        # counting任务计算连续的reward
        # print("datasource:", reward_input["data_source"])
        # 将需要调用 judge model 的任务添加到列表中
        question = reward_input["extra_info"]['question']
        if reward_input["data_source"] == "reason":
            tasks_to_run.append((compute_score_math, (question, reward_input["ground_truth"], answer_text), idx, format_score))
        else: # "general"
            tasks_to_run.append((compute_score_general, (answer_text, reward_input["ground_truth"], question), idx, format_score))

    # 使用 ThreadPoolExecutor 并发执行任务
    # max_workers 可以根据你的服务器性能和API限制进行调整
    with ThreadPoolExecutor(max_workers=32) as executor: # 适当增加 max_workers
        future_to_task_info = {}
        for func, args, original_idx, format_s in tasks_to_run:
            future = executor.submit(func, *args)
            future_to_task_info[future] = (original_idx, format_s)

        for future in as_completed(future_to_task_info):
            original_idx, format_s = future_to_task_info[future]
            try:
                accuracy_score = future.result()
            except Exception as exc:
                print(f'Generated an exception for task {original_idx}: {exc}')
                accuracy_score = 0.0 # 发生错误时，将准确率设为0

            # 将结果存储到正确的位置
            final_scores_list[original_idx] = {
                "overall": (1 - format_weight) * accuracy_score + format_weight * format_s,
                "format": format_s,
                "accuracy": accuracy_score,
            }
            print(f"original_idx:{original_idx}. accuracy:{accuracy_score}. format:{format_s}")

    return final_scores_list


def compute_score_general(predict_str: str, ground_truth: str, question_text) -> float:
    full_prompt = get_prompt(question_text, ground_truth, predict_str)


    for it in range(5): 
        try:
            client = random.choice(client_list)
            chat_response = client.chat.completions.create(
                model=model, # 确保这是正确的模型名称
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": full_prompt},
                ],
                seed = random.randint(0, 1000000),
                temperature=0.3,
                max_tokens=8192,
            )
            response = chat_response.choices[0].message.content.strip()
            break
        except Exception as e:
            print(f' [ERROR general] generative_verify error: {e}')
            # 可以在这里添加等待时间，避免频繁重试导致API限制
            # import time
            time.sleep(5) 
            continue
    _score = 0
    f_response = response
    # 假设这里的 `<最终结果>` 和 `boxed{}` 是模型返回的特定格式
    if '<最终结果>' in f_response:
        # 使用正则表达式更健壮地匹配 <最终结果>...</最终结果>
        match = re.search(r'<最终结果>(.*?)</最终结果>', f_response, re.DOTALL)
        if match:
            f_response = match.group(1).strip()
        else:
            # 如果没有匹配到闭合标签，则从 <最终结果> 之后开始取
            f_response = f_response.split('<最终结果>')[-1].strip()

    if 'boxed' in f_response:
        # 使用正则表达式匹配 boxed{...}
        match = re.search(r'boxed\{(.*?)\}', f_response, re.DOTALL)
        if match:
            f_response = match.group(1).strip()
        else:
            # 如果没有匹配到闭合括号，则从 boxed{ 之后开始取
            f_response = f_response.split('boxed{')[-1].strip()

    if 'Yes' in f_response:
        _score = 1
    else:
        _score = 0
    
    acc_reward = 1.0 if _score else 0.0
    print(f'DEBUG JUDGE GENERAL {f_response=} {_score=} response: {f_response}')
    return acc_reward

MATH_VERIFY_PROMPT = """# CONTEXT #
I am a teacher, and I have some high-level math problems. I am tasked with evaluating the correctness of a student's answer.
Below, I am provided with a problem and a reference answer. Additionally, a student's answer is provided. My job is to assess whether the student's answer captures the same meaning as the reference answer, even when expressed with different wording or format.

# OBJECTIVE #
I need you to judge whether the student's answer is correct given the ground truth answer.

Your tasks include:
1. Identify Mathematical or Notational Equivalence: Pay special attention to any LaTeX expressions in both answers. Confirm that the mathematical relationships, variables, and operations conveyed are equivalent.

# TONE #
Professional, scientific.

# RESPONSE: MARKDOWN REPORT #
## Equivalence Judgement
[Whether the student's answer share the same meaning with the reference answer. (TRUE or FALSE)]

# ATTENTION #
 - The reference answer is ALWAYS correct. You should carefully judge whether the student gives the same answer as reference answer.
 - The Equivalence Judgement is only TRUE or FALSE. The answer is FALSE even if the student's final answer almost correct with a minor mistakes.
 - Don't give extra explanation.

**Question**:
{query}

**Reference Answer**
{gold_ans}

## Student Final Answer
{pred_ans}"""

def extract_numeric_value(text: str) -> float | None:
    """从字符串中提取第一个数字（支持整数和小数）"""
    # 移除千分位逗号
    text = text.replace(',', '')
    # 匹配整数或浮点数
    match = re.search(r"[-+]?\d*\.\d+|\d+", text)
    if match:
        try:
            return float(match.group())
        except ValueError:
            return None
    return None

def counting_reward(response: str, ground_truth: str) -> float:
    """
    针对计数任务的奖励函数：
    1. 提取 \boxed{} 中的内容
    2. 转换为数字
    3. 根据差异计算 0 到 1 之间的分数
    """
    
    # 2. 转换为数值
    ans_val = extract_numeric_value(response)
    gt_val = extract_numeric_value(ground_truth)
    
    # 如果无法解析数字，且字符串不完全相等，给 0 分
    if ans_val is None or gt_val is None:
        return 1.0 if grade_answer(ans_val, gt_val) else 0.0

    # 3. 计算差异得分
    if ans_val == gt_val:
        return 1.0
    
    # 计算绝对差值
    diff = abs(ans_val - gt_val)
    
    # 方案 A: 相对误差衰减 (推荐)
    # 逻辑：差异越小分数越高。如果差值达到或超过标准答案本身，分数为 0
    # 公式：max(0, 1 - |pred - gt| / gt)
    # 例如：GT=47, Pred=46 -> Score = 1 - 1/47 = 0.978
    if gt_val != 0:
        score = max(0.0, 1.0 - (diff / abs(gt_val)))
    else:
        # 如果 GT 是 0，则使用平滑的指数衰减
        score = math.exp(-diff) 
    print("counting reward:", score)
    return float(score)

# def compute_score_count(query: str, ground_truth: str, model_answer: str) -> float:
#     return counting_reward(model_answer, ground_truth=ground_truth)

def compute_score_math(query: str, ground_truth: str, model_answer: str) -> float:
    full_prompt = MATH_VERIFY_PROMPT.format(
        query=query,
        gold_ans=ground_truth,
        pred_ans=model_answer,
    )

    response = ""
    # 这里的重试逻辑也可以考虑提取出来或者统一管理
    for it in range(5): 
        try:
            client = random.choice(client_list)
            chat_response = client.chat.completions.create(
                model=model, # 确保这是正确的模型名称
                messages=[
                    {"role": "user", "content": full_prompt},
                ],
                seed = random.randint(0, 1000000),
                temperature=0.5,
            )
            response = chat_response.choices[0].message.content.strip()
            break
        except Exception as e:
            print(f' [ERROR math] generative_verify error: {e}')
            # 可以在这里添加等待时间，避免频繁重试导致API限制
            # import time
            time.sleep(5) 
            continue

    judgement = response.split('## Equivalence Judgement')[-1].lower()
    if 'true' in judgement and 'false' not in judgement:
        print(f'DEBUG JUDGE MATH True: {response}')
        return 1.0
    elif 'false' in judgement and 'true' not in judgement:
        print(f'DEBUG JUDGE MATH False: {response}')
        return 0.0
    else:
        print(f' [ERROR math] verify bug output: {response}')
        return 0.0
