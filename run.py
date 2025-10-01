import os

from env import Env
from agent import PPDPP
from utils import *
from itertools import count
from tqdm import tqdm
import argparse
from transformers import BertTokenizer, RobertaTokenizer, BertConfig, RobertaConfig
from fastchat.model import add_model_args
import copy
import json

import numpy as np
import torch

class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy().tolist()
        elif np.issubdtype(type(obj), np.integer):
            return int(obj)
        elif np.issubdtype(type(obj), np.floating):
            return float(obj)
        return super().default(obj)

tok = {'bert': BertTokenizer, 'roberta': RobertaTokenizer}
cfg = {'bert': BertConfig, 'roberta': RobertaConfig}
lr2lr_str = {
    1e-2:"1e-2",
    1e-3:"1e-3",
    1e-4:"1e-4",
    1e-5:"1e-5",
    1e-6:"1e-6"
}
beta2beta_str ={
    0.1:"0.1",
    0.01:"0.01",
    0.001:"0.001",
}
with open(f'RSO/data/rl_data/inspired/kg/id2info.json', 'r', encoding="utf-8") as f:#for inspired
    id2info = json.load(f)

def train(args, config, dataset, filename, tokenizer):
    env = Env(args, dataset, mode='train') # env init
    set_random_seed(args.seed)
    policy = PPDPP(args, config, tokenizer) # policy network init

    # load policy parameters
    if args.sft_dir is not None:
        print('Staring loading policy model from {}'.format(args.sft_dir))
        policy.load_model(data_name=args.data_name, filename=args.sft_dir)
    
    if args.load_rl_epoch > 0:
        print('Staring loading rl model in epoch {}'.format(args.load_rl_epoch))
        policy.load_model(data_name=args.data_name, filename=filename, epoch_user=args.load_rl_epoch)
    

    test_performance = []
    if args.do_eval:
        for i in range(args.eval_begin_epoch, args.eval_end_epoch + 1):
            if i > 0:
                print('Staring loading rl model in epoch {}'.format(i))
                policy.load_model(data_name=args.data_name, filename=filename, epoch_user=i)
            if args.do_eval:        
                SR15_mean = evaluate(args, dataset, policy, filename, i, env)
                test_performance.append(SR15_mean)
    

    if not args.do_train:
        return
    for train_step in range(args.load_rl_epoch + 1, args.max_steps+1):
        SR, AvgT, total_reward = 0., 0., 0.
        loss = torch.tensor(0, dtype=torch.float, device=args.device)
        for i_episode in tqdm(range(args.sample_times),desc='sampling'):
            #blockPrint()
            print('\n================new tuple:{}===================='.format(i_episode))
            state, case = env.reset()#state means the selected conversation

            epi_reward = 0
            done = False
            for t in count():   # user  dialog
                # action = policy.select_action(state)#todo: complete both situations that the first sentence is the system's or the user's
                state, reward, done, full_state = env.step(state,policy)
                
                epi_reward += reward
                reward = torch.tensor([reward], device=args.device, dtype=torch.float)
                policy.rewards.append(reward)

                if done:
                    if done == 1:
                        SR += 1
                    AvgT += t+1
                    total_reward += epi_reward
                    break

            newloss = policy.optimize_model()
            if newloss is not None:
                loss += newloss
            
        enablePrint() # Enable print function
        print('loss : {} in epoch_uesr {}'.format(loss.item()/args.sample_times, args.sample_times))
        print('SR:{}, AvgT:{}, rewards:{} Total epoch_uesr:{}'.format(SR / args.sample_times,
                    AvgT / args.sample_times, total_reward / args.sample_times, args.sample_times))
        
        if train_step % args.save_num == 0:
            policy.save_model(data_name=args.data_name, filename=filename, epoch_user=train_step, )

        if train_step % args.eval_num == 0:
            print(f"Finish training epoch {train_step}, Please eval after all training is done")
        
    print(test_performance)

def evaluate(args, dataset, policy, filename, i_episode, train_env):
    if 'llama2' in [args.system, args.user, args.critic] :
        test_env = Env(args, dataset, mode='test', env_model=train_env.llama2_model, env_tokenizer=train_env.llama2_tokenizer)
    elif 'qwen2.5' in [args.system, args.user, args.critic]:
        test_env = Env(args, dataset, mode='test', env_model=train_env.qwen2_model, env_tokenizer=train_env.qwen2_tokenizer)
    else:
        test_env = Env(args, dataset, mode='test') # env init
    set_random_seed(args.seed)

    SR, AvgT, total_reward = 0, 0, 0
    SR_turn = [0]* args.max_turn
    turn_result = []
    result = []
    test_size = len(test_env.dataset)
    print('Test size: ', test_size)
    test_filename = 'Evaluate-epoch-{}-'.format(i_episode) + filename
    record_filename = 'Record-epoch-{}-'.format(i_episode) + filename
    learning_rate_str = lr2lr_str[args.learning_rate]
    beta_str = beta2beta_str[args.entropy_coef]
    if not args.use_credibility:
        REC_PATH = TMP_DIR[args.data_name] + '/nocredibility' + f'/beta_{beta_str}' + f'/lr_{learning_rate_str}' + '/eval_result/' + record_filename + '.txt'
        REC_FULL_STATE_PATH = TMP_DIR[args.data_name] + '/nocredibility' + f'/beta_{beta_str}' + f'/lr_{learning_rate_str}' + '/eval_result/' + "full_state_" + record_filename + '.txt'
    elif not args.use_personalization:
        REC_PATH = TMP_DIR[args.data_name] + '/nopersonalization' + f'/beta_{beta_str}' + f'/lr_{learning_rate_str}' + '/eval_result/' + record_filename + '.txt'
        REC_FULL_STATE_PATH = TMP_DIR[args.data_name] + '/nopersonalization' + f'/beta_{beta_str}' + f'/lr_{learning_rate_str}' + '/eval_result/' + "full_state_" + record_filename + '.txt'
    elif not args.use_strategy:
        REC_PATH = TMP_DIR[args.data_name] + '/nostrategy' + f'/beta_{beta_str}' + f'/lr_{learning_rate_str}' + '/eval_result/' + record_filename + '.txt'
        REC_FULL_STATE_PATH = TMP_DIR[args.data_name] + '/nostrategy' + f'/beta_{beta_str}' + f'/lr_{learning_rate_str}' + '/eval_result/' + "full_state_" + record_filename + '.txt'
    else:
        REC_PATH = TMP_DIR[args.data_name] + f'/beta_{beta_str}' + f'/lr_{learning_rate_str}' + '/eval_result/' + record_filename + '.txt'
        REC_FULL_STATE_PATH = TMP_DIR[args.data_name] + f'/beta_{beta_str}' + f'/lr_{learning_rate_str}' + '/eval_result/' + "full_state_" + record_filename + '.txt'
    path_dir = os.path.dirname(REC_PATH)
    if not os.path.exists(path_dir):
        os.makedirs(path_dir,exist_ok=True)

    import json
    file_path = REC_PATH
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        blocks = [block for block in content.split('\n\n') if block.strip()]

        dialogs = []
        for block in blocks:
            dialogs.append(json.loads(block))
        test_env.test_num = len(dialogs)
        rec_file = open(REC_PATH, 'a')
        full_state_file = open(REC_FULL_STATE_PATH, 'a')
    else:
        rec_file = open(REC_PATH, 'w')
        full_state_file = open(REC_FULL_STATE_PATH, 'w')
    for test_num in tqdm(range(test_size)):  #test_size
        #blockPrint()
        print('\n================test tuple:{}===================='.format(test_num))
        epi_reward = 0
        done = 0
        is_last_turn = False
        state, case = test_env.reset()
        genre_dict = case['genre_dict']
        target_list = []

        for k, v in id2info.items():
            if 'genre' in v and set(genre_dict).issubset(set(v['genre'])):
                target_list.append(v['name'])

        if len(target_list) == 0:
            raise Exception("empty target list")
        
        full_state = copy.deepcopy(state)
        for t in count():  
            state, reward, done, full_state = test_env.step(state, policy, is_test=True, full_state=full_state, record_full_state=True)
            if full_state is not None:
                rec_success_rec_1 = False
                rec_success_rec_5 = False
                rec_success_rec_10 = False
                rec_success = False
                for item in full_state[::-1]:
                    if item["role"] == 'Recommender':
                        all_candidate_items_name = item["all_candidate_items_name"]
                        if all_candidate_items_name is None:
                            item["rec_success_rec_1"] = False
                            item["rec_success_rec_5"] = False
                            item["rec_success_rec_10"] = False
                            item["rec_success"] = False
                            break
                        #cal rec_success_rec_1:
                        for target in target_list:
                            if target == all_candidate_items_name[0]:
                                rec_success_rec_1 = True
                                break
                        #cal rec_success_rec_5:
                        for target in target_list:
                            if target in all_candidate_items_name[:5]:
                                rec_success_rec_5 = True
                                break
                        #cal rec_success_rec_10:
                        for target in target_list:
                            if target in all_candidate_items_name[:10]:
                                rec_success_rec_10 = True
                                break
                            
                        #cal rec_success:
                        rec_item = item["rec_item"]
                        rec_info = item["rec_info"]
                        if rec_item in target_list:
                            rec_success = True
                        else:
                            rec_success = False
                        item["rec_success_rec_1"] = rec_success_rec_1
                        item["rec_success_rec_5"] = rec_success_rec_5
                        item["rec_success_rec_10"] = rec_success_rec_10
                        item["rec_success"] = rec_success
                        break

            if done: 
                if done == 1:  
                    SR_turn = [v+1 if i>t  else v for i, v in enumerate(SR_turn) ]
                    SR += 1
                total_reward += epi_reward
                AvgT += t+1

                rec_file.write(json.dumps({
                    'dialog': state,
                    'reward': epi_reward
                }, indent=4, ensure_ascii=False, cls=CustomJSONEncoder) + '\n\n')
                
                full_state_file.write(json.dumps({
                    'full_state': full_state,
                    'reward': epi_reward
                }, indent=4, ensure_ascii=False, cls=CustomJSONEncoder) + '\n\n')
                break

        enablePrint()

        torch.cuda.empty_cache()
    
    
            
    
    SR_mean = float(SR)/test_size
    AvgT_mean = float(AvgT)/test_size
    reward_mean = total_reward/test_size
    SR_all = [SR_mean, AvgT_mean, reward_mean]
    nocredibility = not args.use_credibility
    nopersonalization = not args.use_personalization
    nostrategy = not args.use_strategy
    save_rl_mtric(dataset=args.data_name, filename=test_filename, epoch=test_num, SR=SR_all,learning_rate=args.learning_rate, beta=args.entropy_coef, mode='test', nocredibility=nocredibility, nopersonalization=nopersonalization, nostrategy=nostrategy)  # save RL SR
    print('save test evaluate successfully!')

    SRturn_all = [0] * args.max_turn
    for i in range(len(SRturn_all)):
        SRturn_all[i] = float(SR_turn[i])/test_size
    print('success turn:{}'.format(SRturn_all))
    print('SR:{}, AvgT:{}, reward:{}'.format(SR_mean, AvgT_mean, reward_mean))
    if not args.use_credibility:
        PATH = TMP_DIR[args.data_name] + '/nocredibility' + f'/beta_{beta_str}' + f'/lr_{learning_rate_str}' + '/eval_result/' + test_filename + '.txt'
    elif not args.use_personalization:
        PATH = TMP_DIR[args.data_name] + '/nopersonalization' + f'/beta_{beta_str}' + f'/lr_{learning_rate_str}' + '/eval_result/' + test_filename + '.txt'
    elif not args.use_strategy:
        PATH = TMP_DIR[args.data_name] + '/nostrategy' + f'/beta_{beta_str}' + f'/lr_{learning_rate_str}' + '/eval_result/' + test_filename + '.txt'
    else:
        PATH = TMP_DIR[args.data_name] + f'/beta_{beta_str}' + f'/lr_{learning_rate_str}' + '/eval_result/' + test_filename + '.txt'
    with open(PATH, 'a') as f:
        f.write('Training epocch:{}\n'.format(i_episode))
        f.write('===========Test Turn===============\n')
        f.write('Testing {} user tuples\n'.format(test_num))
        for i in range(len(SRturn_all)):
            f.write('Testing SR-turn@{}: {}\n'.format(i, SRturn_all[i]))
        f.write('================================\n')
    with open(PATH, 'a') as f:
        f.write('{}\t{}\t{}\t{}\n'.format(i_episode, SR_mean, AvgT_mean, reward_mean))
    return SR_all

    


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_strategy', action='store_true', help='Whether to use strategy.')
    parser.add_argument('--use_credibility', action='store_true', help='Whether to use credibility.')
    parser.add_argument('--use_personalization', action='store_true', help='Whether to use personalization.')
    parser.add_argument('--entropy_coef', type=float, default=0.01, help='Entropy regularization coefficient. ')
    
    parser.add_argument('--seed', '-seed', type=int, default=1, help='random seed.')
    parser.add_argument('--num_gpus', type=int, default=1, help='number of gpus.')
    parser.add_argument('--epochs', '-me', type=int, default=50000, help='the number of RL train epoch')
    parser.add_argument('--gamma', type=float, default=0.999, help='reward discount factor.')
    parser.add_argument('--learning_rate', type=float, default=1e-2, help='learning rate.')

    parser.add_argument('--data_name', type=str, default='inspired', choices=['inspired','redial'],
                        help='One of {inspired, redial}.')
    parser.add_argument('--system', type=str, default='qwen2.5', choices=['llama2', 'qwen2.5'],
                        help='One of {llama2, qwen2.5}.')
    parser.add_argument('--user', type=str, default='qwen2.5', choices=['llama2', 'qwen2.5'],
                        help='One of {llama2, qwen2.5}.')
    parser.add_argument('--critic', type=str, default='qwen2.5', choices=['llama2', 'qwen2.5'],
                        help='One of {llama2, qwen2.5}.')
    parser.add_argument('--sft_dir', default='/path/to/sft_dir/', #../pretrain/outputs/best_pretrain.pt
                        type=str, help="Pretrain model path.")
    parser.add_argument('--max_turn', type=int, default=10, help='max conversation turn')
    parser.add_argument('--mode', type=str, default='train', help='the mode in [train, test]')
    parser.add_argument('--load_rl_epoch', type=int, default=0, help='load agent from epoch')#if no rl ckpt, set to 0
    parser.add_argument("--id2info_path", type=str, default="RSO/data/rl_data/inspired/kg/id2all_info.json")# for inspired
    parser.add_argument("--candidate_item_num", type=int, default=3)
    parser.add_argument("--demographic_path", type=str, default="RSO/data/rl_data/human_file/seeker_demographic.tsv")
    parser.add_argument("--personality_path", type=str, default="RSO/data/rl_data/human_file/seeker_personality.tsv")
    parser.add_argument("--critic_path", type=str, default=None) #if need to use another critic model differing from user\recommender, set the path here
    parser.add_argument("--critic_device", type=str, default=None)#for no enough memory, set to another gpu

    parser.add_argument("--cache_dir", default='/path/to/cache_dir/', type=str, help="The cache directory.")
    parser.add_argument("--item_emb_path", default='RSO/data/rl_data/inspired/kg/descriptions_embs.npy', type=str, help="The item embedding path, end with '.npy'")#for inspired
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--max_seq_length", default=512, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--model_path", type=str, default="/path/to/qwen_2.5_7b_instruct")
    parser.add_argument("--model_name", type=str, default="roberta")
    parser.add_argument("--model_name_or_path", default='/path/to/roberta-large', type=str, help="model name or path")

    parser.add_argument("--do_lower_case", action='store_false', help="Set this flag if you are using an uncased model.")

    parser.add_argument('--max_steps', type=int, default=10, help='max training steps')
    parser.add_argument('--sample_times', type=int, default=100, help='the epoch of sampling')
    parser.add_argument('--eval_num', type=int, default=1, help='the number of steps to evaluate RL model and metric')
    parser.add_argument('--save_num', type=int, default=1, help='the number of steps to save RL model and metric')


    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true', help="Whether to run eval.")
    parser.add_argument("--eval_begin_epoch", type=int, default=0, help="The epoch to start eval.")
    parser.add_argument("--eval_end_epoch", type=int, default=10, help="The epoch to end eval.")
    parser.add_argument("--real_user", action='store_true', help="Whether to use real user.")

    add_model_args(parser)
    args = parser.parse_args()
    
    print(args)
    print(args.device)
    print('data_set:{}'.format(args.data_name))

    dataset = load_dataset(args.data_name)
    filename = '{}-{}-{}-{}-{}'.format(args.data_name,args.sft_dir.replace("/","#"),args.system,args.user,args.critic)

    config = cfg[args.model_name].from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
    tokenizer = tok[args.model_name].from_pretrained(args.model_name_or_path, do_lower_case=args.do_lower_case, cache_dir=args.cache_dir)

    if args.sft_dir:
        args.sft_dir = os.path.join(args.sft_dir, "inspired", args.model_name, 'best_checkpoint')# no redial sft model, use inspired ckpt to initial
    if not os.path.exists(args.sft_dir):
        print("no sft model, randomly initialize policy model")
        args.sft_dir = None

    train(args, config, dataset, filename, tokenizer)

if __name__ == '__main__':
    main()