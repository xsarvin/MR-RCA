import pickle
from config_llm import *
import argparse
import numpy as np
from mr_rca import root_cause
import random
import os
import json


def save(best_res, config):
    filename = f"./result/_{config.method}_{config.s_dataset}_{config.t_dataset}_bs{config.batch_size}_emd{config.embed_dim}_hid{config.hidden_dim}_beta{config.beta}_el{config.event_len}_eps{config.epsilon}.txt"
    # filename = f"./result/_uda_rca_use_only_GAT{config.only_use_gat}_use_gat_linear_{config.use_gat_linear}_ratio{config.ratio}.txt"
    # filename = f"./result/_uda_rca_AIops2022_AIops2021_bs45_emd5_hid5_beta{b}_el20.txt"
    with open(filename, "w") as fw:
        json.dump(best_res, fw)


def seed_everything(seed=2024):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)



def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser()
parser.add_argument('-m', '--method', type=str, default="MR_RCA",
                    choices=["MR_RCA" ], help="method choice")
parser.add_argument('-d', '--dataset', type=str, default="AIops2022",
                    choices=["AIops2021", "AIops2022"],
                    help="dataset choice")

parser.add_argument('-e', '--entity', type=str, default="node",
                    choices=["pod", "node","service"],
                 )


#########################hyperparameter experiment######

parser.add_argument('-t', '--beta', type=float, default=0.7,
                  )
parser.add_argument('--alpha', type=float, default=0.6,
                    )
parser.add_argument('--k', type=int, default=3)

parser.add_argument('--lamb', type=int, default=60,
                    )

parser.add_argument('--consistency_num', type=int, default=1
                    )

seed_everything()
args = parser.parse_args()
print(args)

result = root_cause(args)

