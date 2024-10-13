import argparse
import torch
from diff_evaluate import EvaluatorTCN


seed = 1538574472
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()

# runtime params
parser.add_argument('--num_workers', type=int, default=16)
parser.add_argument('--exp_name_prefix', type=str, default="ckpt_")
parser.add_argument('--model', default='pred-tcn')

parser.add_argument('--ds', default='bf')
parser.add_argument('--sample_rate', default=1, type=int)
parser.add_argument('--layer_type', default='gated')

# eval
parser.add_argument('--epoch', default=50, type=int)

# diff
parser.add_argument('--num_samples', default=1, type=int)
parser.add_argument('--test_num_samples', default=1, type=int)
parser.add_argument('--num_infr_diff_timesteps', default=250, type=int)
parser.add_argument('--num_diff_timesteps', default=1000, type=int)
parser.add_argument('--conditioned_x0', action='store_true')
parser.add_argument('--diff_loss_type', type=str, default='l2') 
parser.add_argument('--diff_obj', type=str, default='pred_x0')

# model params
parser.add_argument('--num_stages', default=5, type=int)
parser.add_argument('--obs_stages', default=1, type=int)
parser.add_argument('--ant_stages', default=4, type=int)
parser.add_argument('--num_layers', default=10, type=int)

parser.add_argument('--use_features', action='store_true')
parser.add_argument('--use_inp_ch_dropout', action='store_true')
parser.add_argument('--part_obs', action='store_true')


# datasets
# Bf and 50s
parser.add_argument('--split')
parser.add_argument('--mapping_file')


args = parser.parse_args()


# Dataset parameters
pred_perc = .5
eval_obs_perc = [.2, .3]

if args.ds == 'assembly': 
    args.num_classes = 202
    actions_dict = {}
else:
    # mapping actions
    mapping_file = args.mapping_file
    file_ptr = open(mapping_file, 'r')
    actions = file_ptr.read().split('\n')[:-1]
    file_ptr.close()
    actions_dict = dict()
    for a in actions:
        actions_dict[a.split()[1]] = int(a.split()[0])

    # class num
    num_classes = len(actions_dict)
    hl_num_classes = [2, 10]
    args.num_classes = num_classes


# Train / Evaluate
evaluator = EvaluatorTCN(args)
for obs_perc in eval_obs_perc:
    results = evaluator.evaluate(args,
                 obs_perc,
                 actions_dict,)

    print(f'Obs : {obs_perc}, num samples : {args.test_num_samples}')
    print(results)
