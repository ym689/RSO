import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from fastchat.model import load_model, get_conversation_template

import openai
from agent import PPDPP
from utils import *
from prompt import *
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import json
import nltk
nltk.data.path.append("/path/to/nltk_punkt/")
import re
import time
import numpy as np

system_role = {'inspired':'Recommender', 'redial':'Recommender'}
user_role = {'inspired':'Seeker', 'redial':'Seeker'}
message_format = {'inspired':InspiredMessages, 'redial':RedialMessages}

sentence_trans = SentenceTransformer("/path/to/st_all_MiniLM-L6-v2")
YOUR_API_KEY = ""
def call_embedding(text):
    return sentence_trans.encode(text)


class Env(object):
    def __init__(self, args, dataset, mode, env_model=None, env_tokenizer=None):
        if 'llama2' in [args.system, args.user, args.critic]:
            if mode == 'train':
                # print(f"load_8bit is {args.load_8bit}")
                 self.llama2_model, self.llama2_tokenizer = load_model(
                    args.model_path,
                    args.device,
                    args.num_gpus,
                    args.max_gpu_memory,
                    dtype=torch.bfloat16,
                    load_8bit=args.load_8bit,
                    cpu_offloading=args.cpu_offloading,
                    debug=args.debug,
                )
            else:
                self.llama2_model = env_model
                self.llama2_tokenizer = env_tokenizer
        
        elif 'qwen2.5' in [args.system, args.user, args.critic]:
            if mode == 'train':
                self.qwen2_model, self.qwen2_tokenizer = load_model(
                    args.model_path,
                    args.device,
                    args.num_gpus,
                    args.max_gpu_memory,
                    dtype=torch.bfloat16,
                    load_8bit=args.load_8bit,
                    cpu_offloading=args.cpu_offloading,
                    debug=args.debug,
                )
            else:
                self.qwen2_model = env_model
                self.qwen2_tokenizer = env_tokenizer

        if args.critic_path is not None:
            self.critic_model, self.critic_tokenizer = load_model(
                args.critic_path,
                args.critic_device if args.critic_device is not None else args.device,
                args.num_gpus,
                args.max_gpu_memory,
                dtype=torch.bfloat16,
                load_8bit=args.load_8bit,
                cpu_offloading=args.cpu_offloading,
                debug=args.debug,
            )
        else:
            self.critic_model = None
            self.critic_tokenizer = None

        self.use_strategy = args.use_strategy
        self.use_credibility = args.use_credibility
        self.use_personalization = args.use_personalization

        self.args = args
        self.dataset = dataset[mode]
        self.max_turn = args.max_turn
        self.conversation = []
        self.cur_conver_step = 0
        self.test_num = 0
        self.mode = mode
        self.candidate_item_num = args.candidate_item_num
        self.item_emb_arr = np.load(args.item_emb_path)
        self.seeker_simulator = SeekerSimulator(demographic_path=args.demographic_path, personality_path=args.personality_path)
        self.real_user = args.real_user
        with open(args.id2info_path, 'r') as f:
            self.id2info = json.load(f)
        self.item_to_recommend = []

        set_random_seed(args.seed)



    def reset(self):
        self.cur_conver_step = 0
        if self.mode == 'train':
            self.case = np.random.choice(self.dataset)
        elif self.mode == 'test':
            self.case = self.dataset[self.test_num]
            self.test_num += 1
        if self.args.data_name == 'inspired':
            common_genre_dict = ["action", "comedy", "drama"]
            if len(self.case['genre_dict']) == 0:
                self.case["genre_dict"] = [common_genre_dict[0].title()]
            else:
                self.case['genre_dict'] = [self.case['genre_dict'][0].title()]
        elif self.args.data_name == 'redial':
            self.case["genre_dict"] = [self.case['genre_dict'][0]]
            
        
        if self.real_user:
            usr_input = input("Please enter your first meaasge as seeker: ")
            self.conversation = [{"role":"Seeker", "content":usr_input}]
        else:
            self.conversation = [{"role":"Seeker", "content":"Hello."}]

        print(self.conversation)
        return self.conversation, self.case

    def rec_reasoning(self, conversation, candidate_items, user_preferences):
        if user_preferences is None:
            result = {
"Recommendation Item": "Not Available",
"Factual Information": "Not Available"
}       
            print("User preference is None, rec_reasoning return all result as not available: ",result)
            return result  
        if not self.use_personalization:
            prompt = '''
You are a recommender assisting the user in finding the best recommendation. Given the conversation history and candidate list your task is to:

1. Select the most suitable recommendation item: Choose the item from the candidate list that best aligns with the conversation history.
2. Summarize relevant factual information: Carefully select and present factual details about the recommended item that will help to make the seeker accept the recommendation.
    - If the seeker has explicitly mentioned which aspect of the recommended item's information they need, please choose that specific part of the factual information and summarize it concisely.
    - If the seeker has not explicitly mentioned which aspect of the recommended item's information they need, please infer the most relevant factual information based on the conversation history.
    - You should focus more on the last sentence of the seeker's response, because it contains the most relevant information regarding the seeker's requirement.
    
Instructions:
    - Consider the user's emotional context: Make your decision thoughtfully, ensuring that the recommendation aligns with the user's desires, tastes, or current mood based on prior interactions.
    - Summarize factual information concisely: Choose only the most relevant details that will aid the user in understanding the item, without overwhelming them with excessive data. Best less than 100 words.
    - Ensure accuracy: All factual information you summarize should be aligned with the data provided and be accurate.

Note:
1. If you think all the candidate items are not suitable, you can output "not available" for both "Recommendation Item" and "Factual Information".
2. Only recommend when you have enough information to make a thoughtful recommendation that aligns with mood. Don't rush into recommendations.
3. If the Candidate List is available and the Candidate Items Factual Information is not available, you should summarize information from your internal knowledge base.
Candidate List:\n<Candidate List>
###
Candidate Items Factual Information:\n<Candidate Items Information>
###
Conversation History:\n<Conversation History>
###
Last Seeker Response:\n<Last Seeker Response>
######
Please response in the following format:
</begin>
{
"Recommendation Item": <Recommended Item Name>,
"Factual Information": <Concise Factual Details>
}
</end>

Assistant:
'''
            #process the candidate_items
            if len(candidate_items) == 0:
                prompt = prompt.replace("<Candidate List>", "not available")
                return None
            else:
                candidate_item_list = []
                for item in candidate_items:
                    if self.args.data_name == 'inspired':
                        candidate_item_list.append(f"{item['title']}({item['year']})")
                    elif self.args.data_name == 'redial':
                        candidate_item_list.append(f"{item['name']}")
                prompt = prompt.replace("<Candidate List>",", ".join(candidate_item_list))

            #process the recommend_info, maybe can chosen by some arguments
            if not self.use_credibility or len(candidate_items)==0:
                prompt = prompt.replace("<Candidate Items Information>", "not available")
            else:
                info_list = []
                for item in candidate_items:
                    if self.args.data_name == 'inspired':
                        info_list.append(f"{item['title']} ({item['year']}):{json.dumps(item, indent=0)}".replace("\n",""))
                    elif self.args.data_name == 'redial':
                        info_list.append(f"{item['name']}:{json.dumps(item, indent=0)}".replace("\n",""))
                prompt = prompt.replace("<Candidate Items Information>", "\n ".join(info_list))
                #inject user preferences

            conversation_history = ""
            for turn in conversation:
                conversation_history += f"{turn['role']}: {turn['content']}\n"
            prompt = prompt.replace("<Conversation History>", conversation_history)
            last_seeker_response = conversation[-1]['content']
            prompt = prompt.replace("<Last Seeker Response>", last_seeker_response)
            response = self.generate_response(self.args.system, prompt, system_role[self.args.data_name], temp_max_new_tokens=200)

            print("rec_reasoning response: ", response)

            if "not available" in response.lower():
                return None
            if "</begin>" in response:
                try:
                    response_ = dict(json.loads(response.split("</begin>")[1].split("</end>")[0].strip()))
                except json.JSONDecodeError:
                    response_ = {}  
                    try:
                        response_["Recommendation Item"] = response.split("Recommendation Item:")[1].split("Factual Information:")[0].strip()
                        response_["Factual Information"] = response.split("Factual Information:")[1].strip()
                    except (IndexError, ValueError):
                        print("Error: Unable to extract 'Recommendation Item' or 'Factual Information'. Returning None.")
                        return None
            else:
                try:
                    response_["Recommendation Item"] = response.split("Recommendation Item:")[1].split("Factual Information:")[0].strip()
                    response_["Factual Information"] = response.split("Factual Information:")[1].strip()
                except (IndexError, ValueError):
                    print("Error: Unable to extract 'Recommendation Item' or 'Factual Information'. Returning None.")
                    return None
           
            if not isinstance(response_["Recommendation Item"], str):
                response_["Recommendation Item"] = str(response_["Recommendation Item"])
            item_name_embed = call_embedding(candidate_item_list)
            response_item_embed = call_embedding(response_["Recommendation Item"]).reshape(1,-1)
            sim_mat = cosine_similarity(response_item_embed, item_name_embed)
            response_["Recommendation Item"] = candidate_item_list[np.argmax(sim_mat)]
            return response_



        prompt = '''
You are a recommender assisting the user in finding the best recommendation. Given the conversation history, candidate list, and user preferences, your task is to:

1. Select the most suitable recommendation item: Choose the item from the candidate list that best aligns with the user's preferences and the conversation history.
2. Summarize relevant factual information: Carefully select and present factual details about the recommended item that will help to make the seeker accept the recommendation.
    - If the seeker has explicitly mentioned which aspect of the recommended item's information they need, please choose that specific part of the factual information and summarize it concisely.
    - If the seeker has not explicitly mentioned which aspect of the recommended item's information they need, please infer the most relevant factual information based on the conversation history.
    - You should focus more on the last sentence of the seeker's response, because it contains the most relevant information regarding the seeker's requirement.
    
Instructions:
    - Consider the user's preferences and emotional context: Make your decision thoughtfully, ensuring that the recommendation aligns with the user's desires, tastes, or current mood based on prior interactions.
    - Summarize factual information concisely: Choose only the most relevant details that will aid the user in understanding the item, without overwhelming them with excessive data. Best less than 100 words.
    - Ensure accuracy: All factual information you summarize should be aligned with the data provided and be accurate.

Note:
1. If you think all the candidate items are not suitable, you can output "not available" for both "Recommendation Item" and "Factual Information".
2. Only recommend when you have enough information to make a thoughtful recommendation that aligns with their preferences and mood. Don't rush into recommendations.
3. If the Candidate List is available and the Candidate Items Factual Information is not available, you should summarize information from your internal knowledge base.
Candidate List:\n<Candidate List>
###
Candidate Items Factual Information:\n<Candidate Items Information>
###
User Preferences:\n<User Preferences>
###
Conversation History:\n<Conversation History>
###
Last Seeker Response:\n<Last Seeker Response>
######
Please response in the following format:
</begin>
{
"Recommendation Item": <Recommended Item Name>,
"Factual Information": <Concise Factual Details>
}
</end>

Assistant:
'''
        
        #process the candidate_items
        if len(candidate_items) == 0:
            prompt = prompt.replace("<Candidate List>", "not available")
            return None
        else:
            candidate_item_list = []
            for item in candidate_items:
                if self.args.data_name == 'inspired':
                    candidate_item_list.append(f"{item['title']}({item['year']})")
                elif self.args.data_name == 'redial':
                    candidate_item_list.append(f"{item['name']}")
            prompt = prompt.replace("<Candidate List>",", ".join(candidate_item_list))

        #process the recommend_info, maybe can chosen by some arguments
        if not self.use_credibility or len(candidate_items)==0:
            prompt = prompt.replace("<Candidate Items Information>", "not available")
        else:
            info_list = []
            for item in candidate_items:
                if self.args.data_name == 'inspired':
                    info_list.append(f"{item['title']} ({item['year']}):{json.dumps(item, indent=0)}".replace("\n",""))
                elif self.args.data_name == 'redial':
                    info_list.append(f"{item['name']}:{json.dumps(item, indent=0)}".replace("\n",""))
            prompt = prompt.replace("<Candidate Items Information>", "\n ".join(info_list))
            #inject user preferences
        if self.use_personalization:
            prompt = prompt.replace("<User Preferences>", user_preferences if user_preferences else "not available")
        else:
            prompt = prompt.replace("<User Preferences>", "not available")

        conversation_history = ""
        for turn in conversation:
            conversation_history += f"{turn['role']}: {turn['content']}\n"
        prompt = prompt.replace("<Conversation History>", conversation_history)
        last_seeker_response = conversation[-1]['content']
        prompt = prompt.replace("<Last Seeker Response>", last_seeker_response)
        response = self.generate_response(self.args.system, prompt, system_role[self.args.data_name], temp_max_new_tokens=200)
        
        print("rec_reasoning response: ", response)

        if "not available" in response.lower():
            return None
        if "</begin>" in response:
            try:
                response_ = dict(json.loads(response.split("</begin>")[1].split("</end>")[0].strip()))
            except json.JSONDecodeError:
                response_ = {}  
                try:
                    response_["Recommendation Item"] = response.split("Recommendation Item:")[1].split("Factual Information:")[0].strip()
                    response_["Factual Information"] = response.split("Factual Information:")[1].strip()
                except (IndexError, ValueError):
                    print("Error: Unable to extract 'Recommendation Item' or 'Factual Information'. Returning None.")
                    return None
        else:
            try:
                response_["Recommendation Item"] = response.split("Recommendation Item:")[1].split("Factual Information:")[0].strip()
                response_["Factual Information"] = response.split("Factual Information:")[1].strip()
            except (IndexError, ValueError):
                print("Error: Unable to extract 'Recommendation Item' or 'Factual Information'. Returning None.")
                return None
        if not isinstance(response_["Recommendation Item"], str):
            response_["Recommendation Item"] = str(response_["Recommendation Item"])
        item_name_embed = call_embedding(candidate_item_list)
        response_item_embed = call_embedding(response_["Recommendation Item"]).reshape(1,-1)
        sim_mat = cosine_similarity(response_item_embed, item_name_embed)
        response_["Recommendation Item"] = candidate_item_list[np.argmax(sim_mat)]
        return response_

    def get_candidate_item(self, conversation_history):

        conv_str = ""
        for context in conversation_history[-2:]:
            conv_str += f"{context['role']}: {context['content']}"
        if len(conv_str)  < 20:
            return [], None
        conv_embed = call_embedding(conv_str).reshape(1,-1)
        # conv_embed = np.asarray(conv_embed).reshape(1,-1)

        sim_mat = cosine_similarity(conv_embed, self.item_emb_arr)        
        rank_arr = np.argsort(sim_mat, axis=-1).tolist()
        rank_arr = np.flip(rank_arr, axis=-1)[:, :50]
        # item_rank_arr = self.id2item_id_arr[rank_arr].tolist()
        # item_rank_arr = [[self.id2entityid[item_id] for item_id in item_rank_arr[0]]]
        all_candidate_items_name = []
        for i in range(rank_arr.shape[1]):
            if self.args.data_name == 'inspired':
                all_candidate_items_name.append(f"{self.id2info[str(rank_arr[0][i])]['title']} ({self.id2info[str(rank_arr[0][i])]['year']})")
            elif self.args.data_name == 'redial':
                all_candidate_items_name.append(f"{self.id2info[str(rank_arr[0][i])]['name']}")

        self.item_to_recommend = []
        for i in range(self.candidate_item_num):
            self.item_to_recommend.append(self.id2info[str(rank_arr[0][i])])
        
        return self.item_to_recommend, all_candidate_items_name # a dict contains movie's metadata

    def get_user_preference(self, conversation:list[dict]):
        #todo

        prompt = """
Given the conversation history between the Recommender and Seeker, summarize the Seeker's movie preferences. 

Conversation history:
<CONVERSATION_HISTORY>

###
Please respond following the instructions below:

1. Review the conversation history and look for any information about the Seeker's movie preferences (such as favorite genres, actors, directors, movie themes, or specific likes/dislikes mentioned).
2. If there is enough information to infer their preferences, provide a concise summary of their movie preferences in a single sentence. Focus on precision and clarity, e.g., "The Seeker prefers action movies with a focus on adventure and thrilling plots."
3. If there is insufficient information, such as the conversation is just started or the Seeker has not mentioned any specific preferences, output "Not available."

Directly output only the Seeker's preference summary or "Not available". 

Please response in the following format:
</begin>
{
"Seeker's preference summary": <Seeker's preference summary>
}
</end>

Assistant:
"""
        conversation_history = ""
        for turn in conversation:
            conversation_history += f"{turn['role']}: {turn['content']}\n"
        prompt = prompt.replace("<CONVERSATION_HISTORY>", conversation_history)
        response = self.generate_response(self.args.system, prompt, system_role[self.args.data_name])
        try:
            response = dict(json.loads(response.split("</begin>")[1].split("</end>")[0].strip()))
            response = response["Seeker's preference summary"]
        except (IndexError, json.JSONDecodeError, KeyError):
            print("Error: Unable to extract 'Seeker's preference summary'. Initial response: ", response)
            try:
                # Fallback: Try to extract directly if response contains the key phrase
                if "Seeker's preference summary" in response:
                    response = response.split("Seeker's preference summary:")[1].strip()
                    # Remove any trailing text after a newline
                    response = response.split('\n')[0].strip()
                else:
                    response = "Not available"
            except Exception:
                response = "Not available"
        if "not available" in response.lower():
            response = None
            print("Concluded user preference: Not available.")
        else:
            print("Concluded user preference: ", response)
        return response
    
    def step(self, state, policy:PPDPP,is_test=False, full_state=None, record_full_state=False):#state is a list of dict, each dict has two keys: role and content
        done = 0
        print('---------------step:{}-------------'.format(self.cur_conver_step))
        torch.cuda.empty_cache()

        action = policy.select_action(self.conversation)
        print("Chosen strategy: ", action)
        # print(action)

        condidate_items, all_candidate_items_name = self.get_candidate_item(self.conversation)
        user_preference = self.get_user_preference(self.conversation)

        pre_rec = self.rec_reasoning(self.conversation, condidate_items, user_preference)
        messages = message_format[self.args.data_name](self.case, 'system', self.conversation, action,candidate_items=condidate_items, user_preferences=user_preference, pre_rec=pre_rec, use_strategy=self.use_strategy, nostrategy_type=self.nostrategy_type, use_personalization=self.use_personalization, use_credibility=self.use_credibility)
        response = self.generate_response(self.args.system, messages, system_role[self.args.data_name])
        response = self.postprocess_response(response, user_role[self.args.data_name])
        response = response.split("Recommender:")[0].strip()
        response = response.split("User:")[0].strip()

        self.conversation.append({"role":system_role[self.args.data_name],"content":response})
        rec_item = ""
        rec_info = {"name":""}
        for item in self.item_to_recommend:
            if self.args.data_name == 'inspired':
                if item['title'] in response:
                    rec_item = f"{item['title']} ({item['year']})"
                    rec_info = item
                    break
            elif self.args.data_name == 'redial':
                if item['name'].split('(')[0].strip() in response:
                    rec_item = f"{item['name']}"
                    rec_info = item
                    break

        if record_full_state:
            full_state.append({"role":system_role[self.args.data_name],"content":response, "user_preference":user_preference, "Recommender_prompt":messages, 
                               "all_candidate_items_name":all_candidate_items_name, "rec_item":rec_item, "rec_info":rec_info})
        print(self.conversation[-1])

        messages = message_format[self.args.data_name](self.case, 'user', self.conversation,seeker_simulator=self.seeker_simulator)
        user_response = self.generate_response(self.args.user, messages, user_role[self.args.data_name])
        user_response = self.postprocess_response(user_response, system_role[self.args.data_name])
        user_response = user_response.split("Recommender:")[0].strip()
        user_response = user_response.split("User:")[0].strip()
            
        if self.real_user:
            user_response = input("Please enter your message as user: ")
        else:
            self.conversation.append({"role":user_role[self.args.data_name], "content":user_response})
        if record_full_state:
            if self.real_user:
                full_state.append({"role":user_role[self.args.data_name], "content":user_response, "Seeker_prompt":"Real person input"})
            else:
                full_state.append({"role":user_role[self.args.data_name], "content":user_response, "Seeker_prompt":messages})
        print(self.conversation[-1])

        messages = message_format[self.args.data_name](self.case, 'critic', self.conversation)
        if record_full_state:
            reward, reward_outputs = self.compute_reward(self.args.critic, messages, self.case, record_full_state=record_full_state)
            full_state.append({"role":"critic", "content":reward_outputs, "critic_prompt":messages, "reward":reward})
        else:
            reward = self.compute_reward(self.args.critic, messages, self.case)

        
        print("reward: ", reward)
        if reward > 0.7:
            print('--> Goal completed !')
            done = 1
        else:
            if self.cur_conver_step == self.max_turn - 1:
                print('--> Maximum number of turns reached !')
                done = -1
            else:
                print('--> On-going !')
                
        self.cur_conver_step += 1
        if record_full_state:
            return self.conversation, reward, done, full_state
        else:
            return self.conversation, reward, done, None
    
    def postprocess_response(self, response, role):
        if role in response:
            response = response.split(role)[0].strip()
        sents = nltk.sent_tokenize(response)
        if len(sents) == 1:
            if response[-1] not in ['.','!','?',':']:
                return response + '.'
            return response.strip()
        try:
            if sents[-1].strip()[-1] not in ['.','!','?',':']:
                return ' '.join(sents[:-1]).strip()
            else:
                return response.strip()
        except Exception as e:
            return response.strip()

    def generate_response(self, model, messages, role, temp_max_new_tokens=None):
        if self.mode == 'test':
            temperature = 0
        else:
            temperature = 0.7
        if model == 'llama2':
            prompt = llama2_prompt(messages, role)
            input_ids = self.vicuna_tokenizer([prompt]).input_ids
            max_new_tokens = self.args.max_new_tokens if temp_max_new_tokens is None else temp_max_new_tokens
            output_ids = self.vicuna_model.generate(
                torch.as_tensor(input_ids).cuda(),
                attention_mask=torch.ones_like(torch.as_tensor(input_ids)).cuda(),
                max_new_tokens=max_new_tokens,
                temperature = temperature+1e-5,
                early_stopping=True,
                pad_token_id=self.vicuna_tokenizer.eos_token_id 
            )
            output_ids = output_ids[0][len(input_ids[0]):]
            output = self.vicuna_tokenizer.decode(output_ids, skip_special_tokens=True,
                                    spaces_between_special_tokens=False)
        elif model == 'qwen2.5':
            prompt = qwen2_prompt(messages, role)
            input_ids = self.qwen2_tokenizer([prompt]).input_ids
            max_new_tokens = self.args.max_new_tokens if temp_max_new_tokens is None else temp_max_new_tokens
            output_ids = self.qwen2_model.generate(
                torch.as_tensor(input_ids).cuda(),
                max_new_tokens=max_new_tokens,
                temperature = temperature+1e-5,
                early_stopping=True
            )
            output_ids = output_ids[0][len(input_ids[0]):]
            output = self.qwen2_tokenizer.decode(output_ids, skip_special_tokens=True,
                                    spaces_between_special_tokens=False)
            
        return output
    
    def compute_reward(self, model, messages, case, record_full_state=False):
        if self.critic_model is not None: #if set critic model to be different with the base model, e.g. Qwen3
            prompt = llama2_prompt(messages, 'critic')
            input_ids = self.critic_tokenizer([prompt]).input_ids
            output_ids = self.critic_model.generate(
                torch.as_tensor(input_ids).cuda(),
                max_new_tokens=400,
                temperature = 1.1,#1.1
                do_sample = True,
                early_stopping=True,
                num_return_sequences=10,
                top_p=0.95, 
            )
            outputs = []
            for o in output_ids:
                output_id = o[len(input_ids[0]):]
                output = self.critic_tokenizer.decode(output_id, skip_special_tokens=True,
                                    spaces_between_special_tokens=False)
                outputs.append(output.split("</think>")[-1].strip("\n") if "</think>" in output else "")  
        elif model == 'llama2':
            prompt = llama2_prompt(messages, 'critic')
            #print(prompt)
            input_ids = self.vicuna_tokenizer([prompt]).input_ids
            output_ids = self.vicuna_model.generate(
                torch.as_tensor(input_ids).cuda(),
                max_new_tokens=16,
                temperature = 1.1,#1.1
                do_sample = True,
                early_stopping=True,
                num_return_sequences=10,
                top_p=0.95, 
            )
            outputs = []
            for o in output_ids:
                output_id = o[len(input_ids[0]):]
                output = self.vicuna_tokenizer.decode(output_id, skip_special_tokens=True,
                                    spaces_between_special_tokens=False)
                outputs.append(output)  
        elif model == 'qwen2.5':
            prompt = qwen2_prompt(messages, 'critic')
            input_ids = self.qwen2_tokenizer([prompt]).input_ids
            output_ids = self.qwen2_model.generate(
                torch.as_tensor(input_ids).cuda(),
                max_new_tokens=16,
                temperature = 1.1,
                do_sample = True,
                early_stopping=True,
                num_return_sequences=10,
            )
            outputs = []
            for o in output_ids:
                output_id = o[len(input_ids[0]):]
                output = self.qwen2_tokenizer.decode(output_id, skip_special_tokens=True,
                                    spaces_between_special_tokens=False)
                outputs.append(output)
        rewards = []
        print(outputs)
        for output in outputs:
            if '3' in output or "reject" in output.lower():
                rewards.append(0.2)
            elif '5' in output or "accept" in output.lower() and 'no' not in output.lower():
                rewards.append(1.0)
            elif '2' in output or 'no' in output.lower() and "interest" in output.lower():
                rewards.append(0.4)
            elif '4' in output or 'yes' in output.lower() and "interest" in output.lower():
                rewards.append(0.8)
            else:
                rewards.append(0.6)
        if len(rewards) == 0:
            reward = 0
        else:
            reward = sum(rewards)/len(rewards)
        print(reward)
        if record_full_state:
            return reward, outputs
        else:
            return reward