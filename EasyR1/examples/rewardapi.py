import base64
import random
from openai import OpenAI # 导入OpenAI库


openai_api_key = "uKkw9zzvcRbHq0cOtoO8F1dVGQrj8kGK" 
api_base = "https://antchat.alipay.com/v1"
client = OpenAI(
    api_key=openai_api_key,
    base_url=api_base,
)

def get_prompt(_question, _answer, _pred):
    with open('/ossfs/workspace/EasyR1/examples/reward_function/verify_prompt.md', 'r', encoding='utf-8') as file:
        judge_system_prompt = file.read()
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
    return judge_system_prompt, full_prompt

def compute_score_general(predict_str: str, ground_truth: str, question_text) -> float:
    system_prompt, full_prompt = get_prompt(question_text, ground_truth, predict_str)

    chat_response = client.chat.completions.create(
        model="Qwen3-32B",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": full_prompt},
        ],
        seed = random.randint(0, 1000000),
        temperature=0.3,
        max_tokens=8192,
    )
    response = chat_response.choices[0].message.content.strip()
    _score = 0
    f_response = response
    if '<最终结果>' in f_response:
        f_response = f_response.split('<最终结果>')[-1].strip().split('<\最终结果>')[0].strip()
    if 'boxed' in f_response:
        f_response = f_response.split('boxed{')[-1].strip().split('}')[0].strip()
    if 'Yes' in f_response:
        _score = 1
    else:
        _score = 0
    if _score:
        acc_reward = 1.0
    else:
        acc_reward = 0.0
    print(f'DEBUG JUDGE {f_response=} {_score=}')
    return acc_reward

predict_str = '''<think>Here's a breakdown of the 8 cars in the image:
Far left: A dark-colored car, partially visible, parked on the left edge.
Second from left: A light-colored (possibly silver or white) sedan, fully visible, connected to a charging station.
Third from left: A dark-colored sedan, fully visible, connected to a charging station.
Fourth from left (center-left): A dark-colored sedan, fully visible, connected to a charging station.
Center-right (further back): A dark-colored SUV or sedan, partially visible in the background, further down the charging lane.
Third from right: A light-colored (possibly silver or white) sedan, fully visible, connected to a charging station.
Second from right: A dark-colored car, partially visible, parked on the right edge.
Far back (center): A dark-colored sedan or SUV, fully visible in the background, parked further into the facility.</think><answer>8</answer>'''
ground_truth = "6"
question = 'how many cars in the image?'
score = compute_score_general(predict_str, ground_truth, question)
print(f"Score: {score}")