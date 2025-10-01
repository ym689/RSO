import json
import ast
from seeker_simulator import SeekerSimulator
from fastchat.model import load_model
import torch
from utils import load_dataset
import os
from tqdm import tqdm
import argparse

seeker_simulator = SeekerSimulator()


model_path = "/path/to/qwen_2.5_7b_instruct"
device = "cuda"
num_gpus = 1
max_gpu_memory = None
load_8bit=None
cpu_offloading=None
debug = None

qwen2_model, qwen2_tokenizer = load_model(
                model_path,
                device,
                num_gpus,
                max_gpu_memory,
                dtype=torch.bfloat16,
                load_8bit=load_8bit,
                cpu_offloading=cpu_offloading,
                debug=debug,
            )

def generate_response(prompt):
    input_ids = qwen2_tokenizer([prompt]).input_ids
    output_ids = qwen2_model.generate(torch.as_tensor(input_ids).cuda(), max_new_tokens=200,temperature=1e-5,early_stopping=True)
    output_ids = output_ids[0][len(input_ids[0]):]
    output = qwen2_tokenizer.decode(output_ids, skip_special_tokens=True,
                                    spaces_between_special_tokens=False)
    return output

with open('RSO/eval/inspired_factinfo.json', 'r', encoding='utf-8') as file:
    movie_data = json.load(file)


titileyear2info = {}
for movie in movie_data:
    title = movie["title"]
    year = movie["year"]
    titileyear2info[f"{title} ({year})"] = movie


##read full state
def read_full_state(path):
    datas = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = ast.literal_eval(line)
            datas.append(data)
    return datas

def get_credibility_score_for_turn(content,movie_info):
    prompt = '''You are an evaluator and you need to judge the truthfulness of the recommender's utterance based on the given source information.
Rate your score based on the Evaluation Standard.

Evaluation Standard
#######
Score 5: Exact Match: The sentence exactly matches the information provided and every claim in the sentence is well-supported by the source information.
Score 4: Supported Inference: Most of the claims in the sentence is supported by the source information with minor details can be logically inferred from the information provided.
Score 3: Partial Match: The sentence partially matches the information provided, with some details supported and others not mentioned.
Score 2: Inconsistent Detail: The sentence contains much information that are not supported by the information provided.
Score 1: Unsubstantiated Claim: The sentence makes a claim that is contradict to the information provided.
#######

Recommender Utterance
#######
'''
    prompt += content + '\n'
    prompt += '''#######
    
Source Information
#######
'''
    for k,v in movie_info.items():
        prompt += f"{k}: {v}\n"
    prompt += '''#######

First summarize the information in the recommender' utterance and compare it with the source information to judge its truthfulness, then give your integer score.
Output your reasoning process in the "Evidence".
Output your score in the "Truthfulness".
Response in the following JSON format:
{"Evidence": <string>, "Truthfulness": <int>}
Response with the JSON only without any block!
'''
    response = generate_response(prompt)

    try:
        response = json.loads(response)
        response = int(response['Truthfulness'])
    except json.JSONDecodeError:
        try:
            response = response.split("\"Truthfulness\":")[1].split("}")[0].strip()
            response = int(response)
        except (IndexError, ValueError):
            response = 0
    return response

def get_credibility_score_for_dialog(data):

    scores = {}
    count = 0
    total_score = 0
    not_zero_count = 0
    for turn in tqdm(data['full_state'], desc="Computing scores for one dialog"):
        if turn["role"] == "Recommender" and len(turn["rec_item"]) > 0:
            movie_nameyear = turn["rec_item"]
            if movie_nameyear in titileyear2info:
                movie_info = titileyear2info[movie_nameyear]
                score = get_credibility_score_for_turn(turn["content"],movie_info)

                scores[count] = score
                if score != 0:
                    not_zero_count += 1
                    total_score += score
        count += 1
    avg_score = total_score / not_zero_count if not_zero_count != 0 else 0
    return scores, avg_score

def compute_credibility_score(datas):
    global_scores = []
    avg_scores = []
    global_avg_score = 0
    for data in tqdm(datas, desc="Computing scores for one file"):
        scores, avg_score = get_credibility_score_for_dialog(data)
        global_scores.append(scores)
        avg_scores.append(avg_score)
        global_avg_score += avg_score
    global_avg_score = global_avg_score / len(avg_scores)
    result = {
        "global_scores": global_scores,
        "avg_scores": avg_scores,
        "global_avg_score": global_avg_score
    }
    return result 

def main():
    parser = argparse.ArgumentParser(description="Compute credibility scores.")
    parser.add_argument('--result_dir', type=str, required=True, help='Path to the result dialogue file.')
    parser.add_argument('--score_dir', type=str, required=True, help='Directory to save the score file.')
    args = parser.parse_args()
    result_dir = args.result_dir
    score_dir = args.score_dir

    if not os.path.exists('/'.join(score_dir.split('/')[:-1])):
        os.makedirs('/'.join(score_dir.split('/')[:-1]))
    with open(result_dir, 'r', encoding='utf-8') as f:
        content = f.read()
    datas = []
    blocks = [block for block in content.split('\n\n') if block.strip()]
    for block in blocks:
        data = json.loads(block)
        datas.append(data)
    
    global_scores = []
    avg_scores = []
    global_avg_score = 0
    for data in datas:
        scores, avg_score = get_credibility_score_for_dialog(data)
        global_scores.append(scores)
        avg_scores.append(avg_score)
        global_avg_score += avg_score
    global_avg_score = global_avg_score / len(avg_scores)
    result = {
        "global_scores": global_scores,
        "avg_scores": avg_scores,
        "global_avg_score": global_avg_score
    }
    with open(score_dir + "credibility_score.txt", 'w', encoding='utf-8') as f:
        f.write(json.dumps(result, indent=4))
    

if __name__ == "__main__":
    main()