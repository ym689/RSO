import json
import ast
from seeker_simulator import SeekerSimulator
from fastchat.model import load_model
import torch
from utils import load_dataset
import os
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import argparse

seeker_simulator = SeekerSimulator()

sentence_trans = SentenceTransformer("/path/to/st_all_MiniLM-L6-v2")

def call_embedding(text):
    
    return sentence_trans.encode(text)

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
    output_ids = qwen2_model.generate(torch.as_tensor(input_ids).cuda(), max_new_tokens=300,temperature=1e-5,early_stopping=True)
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

user_id2genre_dict = {}
with open('RSO/eval/inspired_dialog_ids2user_id_and_genre_dict.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        data = json.loads(line)
        user_id2genre_dict[data["user_id"]] = data["genre_dict"]

def get_watching_intension(user_id,refer,type_):
    preferred_genres = [user_id2genre_dict[user_id][0].title()] if len(user_id2genre_dict[user_id]) > 0 else ["Action"]
    if type_ == "pre":
        prompt_ = seeker_simulator.get_seeker_prompt_for_eval(user_id, type_="pre",preferred_genres=preferred_genres)
        prompt_ = prompt_ + str(refer) + '\n'
        print(prompt_)
    elif type_ == "after":
        prompt_ = seeker_simulator.get_seeker_prompt_for_eval(user_id,type_="after",preferred_genres=preferred_genres)
        prompt_ = prompt_ + str(refer) + '\n'
    elif type_ == "true":
        prompt_ = seeker_simulator.get_seeker_prompt_for_eval(user_id, type_="true",preferred_genres=preferred_genres)
        for k,v in refer.items():
            prompt_ = prompt_ + f"{k}: {v}\n"
    
    prompt_ += '''######

Pretend you have no knowledge about the recommended movies, and the only information source about the movie is the information provided above.
You can only consider your watching intention based on the information given above.
First summarize the movie information above and consider how it matches the scoring criteria, then score your watching intention.
Output your reasons to the score in the "Evidence" as breif as possible.
Response in the following JSON format:
{"Evidence": <string>, "Watching Intention": <int>}
Response with the JSON only without any block!
'''
    response = generate_response(prompt_)

    try:
        response = json.loads(response)
        response = float(response['Watching Intention'])
    except json.JSONDecodeError:
        try:
            response = response.split("\"Watching Intention\":")[1].split("}")[0].strip()
            response = float(response)
        except (IndexError, ValueError):
            response = -1.0
    

    return response

def get_conversation_history(data):
    conversation_history = []
    for item in data["full_state"]:
        if item["role"] == 'critic':
            continue
        conversation_history.append({
            "role": item["role"],
            "content":item["content"]
        })
    conver = ""
    for turn in conversation_history:
            conver += f"{turn['role']}: {turn['content']}\n"
    return conver

def get_movie_nameyear(data):
    all_candidate_nameyear = []
    for item in data["full_state"]:
        if item['role'] == 'Recommender':
            if item["all_candidate_items_name"] is not None:
                all_candidate_nameyear.extend(item["all_candidate_items_name"])
    all_nameyear = list(set(all_candidate_nameyear))
    all_nameyear_with_plot = []
    for nameyear in all_nameyear:
        all_nameyear_with_plot.append(f"{nameyear}:{titileyear2info[nameyear]['long_plot']}")
    all_nameyear_emb = call_embedding(all_nameyear_with_plot)
    last_turn = ""
    for turn in data['full_state'][-6:-4]:
        last_turn += f"{turn['role']}: {turn['content']}\n"
    for turn in data['full_state'][-3:-1]:
        last_turn += f"{turn['role']}: {turn['content']}\n"
    last_turn_emb = call_embedding(last_turn).reshape(1,-1)
    sim_mat = cosine_similarity(last_turn_emb,all_nameyear_emb)
    rank_arr = np.argsort(sim_mat, axis=-1).tolist()
    rank_arr = np.flip(rank_arr, axis=-1)[:, :1]
    indice = rank_arr[0][0]
    return all_nameyear[indice]


def get_persuasiveness_score_for_dialogue(user_id,data):

    print(data["full_state"][2]["Seeker_prompt"])
    movie_nameyear = get_movie_nameyear(data)
    movie_info = titileyear2info[movie_nameyear]
    pre_score = get_watching_intension(user_id, movie_nameyear,type_="pre")
    after_score = get_watching_intension(user_id, get_conversation_history(data),type_="after")
    true_score = get_watching_intension(user_id, movie_info,type_="true")
    if pre_score == -1.0 or after_score == -1.0 or true_score == -1.0:
        return {'pre': pre_score, 'after': after_score, 'true': true_score, 'persuasiveness': "None"}
    if after_score <= true_score:
        if pre_score != true_score:
            persuasiveness = 1 - ((true_score - after_score)/(true_score - pre_score))
        else:
            persuasiveness = -2
    else:
        persuasiveness = -1

    return {'user_id': user_id, 'movie_nameyear': movie_nameyear, 'pre': pre_score, 'after': after_score, 'true': true_score, 'persuasiveness': persuasiveness}


def main():

    parser = argparse.ArgumentParser(description="Compute persuasiveness scores.")
    parser.add_argument('--result_dir', type=str, required=True, help='Path to the result dialogue file.')
    parser.add_argument('--score_dir', type=str, required=True, help='Directory to save the score file.')
    args = parser.parse_args()
    result_dir = args.result_dir
    score_dir = args.score_dir
    import os
    if not os.path.exists('/'.join(score_dir.split('/')[:-1])):
        os.makedirs('/'.join(score_dir.split('/')[:-1]))
    with open(result_dir, 'r', encoding='utf-8') as f:
        content = f.read()
    datas = []
    blocks = [block for block in content.split('\n\n') if block.strip()]
    for block in blocks:
        data = json.loads(block)
        datas.append(data)
    test_ds = load_dataset("inspired")["test"]
    user_ids = []
    for i in range(len(test_ds)):
        user_ids.append(test_ds[i]["user_id"])
    with open(score_dir + f"persuasiveness_score.txt", 'w', encoding='utf-8') as f:
        for i in tqdm(range(len(datas)), desc="Computing scores"):
            data = datas[i]
            user_id = user_ids[i]
            scores = get_persuasiveness_score_for_dialogue(user_id, data)
            f.write(f"{scores}\n")
            f.flush()

    
if __name__ == "__main__":
    main()