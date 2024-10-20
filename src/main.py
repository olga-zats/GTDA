import os
import argparse
import random
import pprint

import numpy as np
import torch

from trainers import TrainerTCN
from batch_gen import BatchGeneratorTCN
from batch_gen_assembly import BatchGeneratorAssembly101TCN

from torch.utils.tensorboard import SummaryWriter


seed = 1538574472
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
parser = argparse.ArgumentParser()


# paths
parser.add_argument('--gt_path')
parser.add_argument('--features_path')
parser.add_argument('--model_dir')
parser.add_argument('--results_dir')
parser.add_argument('--load_type', default='numpy')

# run params
parser.add_argument('--debug', action='store_true')
parser.add_argument('--num_workers', type=int, default=16)
parser.add_argument('--action', default='train')

# optimization 
parser.add_argument('--lr', default=0.0005, type=float)
parser.add_argument('--bz', default=8, type=int)
parser.add_argument('--num_epochs', default=100, type=int)

# eval
parser.add_argument('--epoch', default=85, type=int)
parser.add_argument('--vis', action='store_true')

# data
parser.add_argument('--ds', default='bf')  # assembly
parser.add_argument('--sample_rate', default=1, type=int)
parser.add_argument('--part_obs', action='store_true')


# model params
parser.add_argument('--model', default='pred-tcn')
parser.add_argument('--num_stages', default=5, type=int)
parser.add_argument('--obs_stages', default=0, type=int)
parser.add_argument('--ant_stages', default=5, type=int)
parser.add_argument('--num_layers', default=9, type=int)
parser.add_argument('--layer_type', default='gated', type=str)

parser.add_argument('--use_features', action='store_true')
parser.add_argument('--use_inp_ch_dropout', action='store_true')

parser.add_argument('--kernel_size', default=3, type=int)
parser.add_argument('--model_dim', default=64, type=int)
parser.add_argument('--input_dim', default=2048, type=int)
parser.add_argument('--channel_dropout_prob', default=0.4, type=float)


# diffusion
parser.add_argument('--num_samples', default=1, type=int)
parser.add_argument('--num_infr_diff_timesteps', default=50, type=int)
parser.add_argument('--num_diff_timesteps', default=1000, type=int)
parser.add_argument('--conditioned_x0', action='store_true')
parser.add_argument('--diff_loss_type', type=str, default='l2') # ce, sigm
parser.add_argument('--diff_obj', type=str, default='pred_x0')


# datasets
# Bf 
parser.add_argument('--split')
parser.add_argument('--vid_list_file')
parser.add_argument('--vid_list_file_test')
parser.add_argument('--mapping_file')


args = parser.parse_args()
assert args.obs_stages + args.ant_stages == args.num_stages


# Log arguments
arguments = ""
print("*-----------------------------*")
for arg in ["model", "layer_type", "bz", "num_stages", "obs_stages", "ant_stages",
            "kernel_size", "num_layers", "lr", "ds", "sample_rate", "use_features",
            "channel_dropout_prob", "split"]:
    print("   ", arg, " = ", str(getattr(args, arg)))
    arguments += arg + " = " + str(getattr(args, arg)) + ", "
print("*-----------------------------*")


''' INITIALIZATION'''
# GENERAL
exp_name = f'{args.ds}_{args.model}_os_{args.obs_stages}_as_{args.ant_stages}_m_dim_{args.model_dim}' 
if args.layer_type != 'base':
    exp_name += f'_lt_{args.layer_type}'
if args.part_obs:
    exp_name += f'_pvo'
if args.use_inp_ch_dropout:
    exp_name += f'_inp_ch_dr'
if args.num_layers != 9:
    exp_name += f'_nl_{args.num_layers}'


# DIFF
if 'diff' in args.model:
    exp_name += f'_ns_{args.num_diff_timesteps}'
    exp_name += f'_lt_{args.diff_loss_type}'
    exp_name += f'_obj_{args.diff_obj}'
    exp_name += f'_cond_x0_{args.conditioned_x0}'


if args.ds in ['bf', '50s']:
    exp_name += f'_split_{args.split}'

print(f'Dir : {exp_name}')


# Tensorboard
writer = SummaryWriter(log_dir=os.path.join(args.model_dir, 'logs_' + exp_name))
writer.add_text(f"{exp_name}", pprint.pformat(args).replace("\n", "\n\n"))


# Fix seed
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
arguments += "  ,  fixed_seed"
writer.add_text("args", arguments)


# Init save/write directories
model_dir = os.path.join(args.model_dir, exp_name)
results_dir = os.path.join(args.results_dir, exp_name)
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
if not os.path.exists(results_dir):
    os.makedirs(results_dir)


# Dataset parameters
if args.ds != 'assembly': 
    mapping_file = args.mapping_file
    file_ptr = open(mapping_file, 'r')
    actions = file_ptr.read().split('\n')[:-1]
    file_ptr.close()
    actions_dict = dict()
    for a in actions:
        actions_dict[a.split()[1]] = int(a.split()[0])

    # class num
    num_classes = len(actions_dict)
    args.num_classes = num_classes
    print(f'Number of classes : {args.num_classes}')


# Data loaders
pred_perc = .5
eval_obs_perc = [.2, .3]
if args.ds == 'assembly':
    # training
    train_batch_gen = BatchGeneratorAssembly101TCN(args.sample_rate, 'train', obs_perc=0, args=args)
    
    # validation
    val_batch_gens = []
    for obs_p in eval_obs_perc:
        val_batch_gens.append(BatchGeneratorAssembly101TCN(args.sample_rate, 'val', obs_perc=obs_p, args=args))

    #
    actions_dict = train_batch_gen.action_to_idx
    args.num_classes = train_batch_gen.num_classes

else:
    # training
    train_batch_gen = BatchGeneratorTCN(mode='train',
                                        actions_dict=actions_dict,
                                        sample_rate=args.sample_rate,
                                        vid_list_file=args.vid_list_file,
                                        pred_perc=pred_perc, 
                                        obs_perc=0,
                                        args=args)    
    # validation
    val_batch_gens = []
    for obs_p in eval_obs_perc:
        val_batch_gens.append(BatchGeneratorTCN(mode='eval',
                                                actions_dict=actions_dict,
                                                sample_rate=args.sample_rate,
                                                vid_list_file=args.vid_list_file_test,
                                                pred_perc=pred_perc,
                                                obs_perc=obs_p,
                                                args=args))

# Train / Evaluate
trainer = TrainerTCN(args)
if args.action == "train":
    trainer.train(args=args,
                  save_dir=model_dir,
                  batch_gen=train_batch_gen,
                  val_batch_gens=val_batch_gens,
                  device=device,
                  num_workers=args.num_workers,
                  writer=writer,
                  results_dir=results_dir,
                  actions_dict=actions_dict)
else:
    for i, obs_p in enumerate(eval_obs_perc):
        _  = trainer.validate(args=args,
                              epoch=args.epoch,
                              obs_perc=obs_p,
                              batch_gen=val_batch_gens[i],
                              device=device,
                              num_workers=args.num_workers,
                              model_dir=model_dir,
                              actions_dict=actions_dict,
                              sample_rate=args.sample_rate,
                              eval_mode=False)
writer.close()
