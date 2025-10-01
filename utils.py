import pickle
import numpy as np
import random
import torch
import os
import sys



TMP_DIR = {
    "inspired": "/path/to/save/rl/dialogue/inspired",
    "redial": "/path/to/save/rl/dialogue/redial"
}

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
def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__

def load_dataset(data_name):
    dataset = {'train':[], 'test':[], 'valid':[]}
    for key in dataset:
        if os.path.exists("RSO/data/rl_data/%s/%s-%s.txt"%(data_name, data_name, key)):
            with open("/data1/yanming/RSO/data/rl_data/%s/%s-%s.txt"%(data_name, data_name, key),'r') as infile:
                for line in infile:
                    dataset[key].append(eval(line.strip('\n')))
    return dataset


def set_cuda(args):
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
    devices_id = [int(device_id) for device_id in args.gpu.split()]
    device = (
        torch.device("cuda:{}".format(str(devices_id[0])))
        if use_cuda
        else torch.device("cpu")
    )
    return device, devices_id


def save_rl_mtric(dataset, filename, epoch, SR, learning_rate, beta, mode='train', nocredibility=False,nopersonalization=False, nostrategy=False):
    learning_rate_str = lr2lr_str[learning_rate]
    beta_str = beta2beta_str[beta]
    if nocredibility:
        PATH = TMP_DIR[dataset] + "/nocredibility" + f'/beta_{beta_str}' + f'/lr_{learning_rate_str}' + '/eval_result/' + filename + '.txt'
    elif nopersonalization:
        PATH = TMP_DIR[dataset] + "/nopersonalization" + f'/beta_{beta_str}' + f'/lr_{learning_rate_str}' + '/eval_result/' + filename + '.txt'
    elif nostrategy:
        PATH = TMP_DIR[dataset] + "/nostrategy" + f'/beta_{beta_str}' + f'/lr_{learning_rate_str}' + '/eval_result/' + filename + '.txt'
    else:
        PATH = TMP_DIR[dataset] + f'/beta_{beta_str}' + f'/lr_{learning_rate_str}' + '/eval_result/' + filename + '.txt'
    dir_path = os.path.dirname(PATH)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)

    if mode == 'train':
        with open(PATH, 'a') as f:
            f.write('===========Train===============\n')
            f.write('Starting {} user epochs\n'.format(epoch))
            f.write('training SR: {}\n'.format(SR[0]))
            f.write('training Avg@T: {}\n'.format(SR[1]))
            f.write('training Rewards: {}\n'.format(SR[2]))
            f.write('================================\n')
            # f.write('1000 loss: {}\n'.format(loss_1000))
    elif mode == 'test':
        with open(PATH, 'a') as f:
            f.write('===========Test===============\n')
            f.write('Testing {} user tuples\n'.format(epoch))
            f.write('Testing SR: {}\n'.format(SR[0]))
            f.write('Testing Avg@T: {}\n'.format(SR[1]))
            f.write('Testing Rewards: {}\n'.format(SR[2]))
            f.write('================================\n')