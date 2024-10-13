import os
import pickle
import torch
import numpy as np
from evaluation import *


class EvaluatorTCN:
    def __init__(self, args, causal=False):
        super(EvaluatorTCN, self).__init__()
        self.num_classes = args.num_classes

    # ------------------------------------------------------------------ VALIDATION ---------------------------------------------------------------
    def evaluate(self, args, obs_perc, actions_dict):
        eval_percentages = [0.1, 0.2, 0.3, 0.5]
        test_num_samples = args.test_num_samples

        with torch.no_grad():
            # LOGGERS
            n_T_classes_all_files = np.zeros((len(eval_percentages), args.num_classes))
            n_F_classes_all_files = np.zeros((len(eval_percentages), args.num_classes))
            max_n_T_classes_all_files = np.zeros((len(eval_percentages), args.num_classes))
            max_n_F_classes_all_files = np.zeros((len(eval_percentages), args.num_classes))
            frame_div_all_files = np.zeros((len(eval_percentages)))

            # EVAL
            if args.ds == "assembly":
                ds = "assembly"
            else:
                ds = f"{args.ds}_{args.split}"


            # GET PREDICTIONS
            result_file = (
                f"/home/user/diff_results/final/{ds}/{args.model}"
                f"_epoch_{args.epoch}"
                f"_ns_{args.num_stages}"
                f"_ds_{args.num_diff_timesteps}"
                f"_ids_{args.num_infr_diff_timesteps}"
                f"_num_samples_{args.num_samples}"
                f"_dlt_{args.diff_loss_type}"
                f"_cond_x0_{args.conditioned_x0}"
                f"/obs_{obs_perc}.pkl"
            )         
            assert os.path.exists(result_file), f"specified file {result_file} with samples does not exist"
            result_dict = pickle.load(open(result_file, "rb"))
            pred_dict = {k: v for k, v in result_dict.items() if "gt" not in k}
            gt_dict = {k: v for k, v in result_dict.items() if "gt" in k}
            assert len(pred_dict) == len(gt_dict)


            # GO THROUGH PREDICTIONS
            gt_keys = list(gt_dict.keys())
            for k in range(len(gt_keys)):
                gt_key = gt_keys[k]
                pred_key = gt_key.replace("gt_", "")
                assert pred_key in list(pred_dict.keys())

                gt_content = gt_dict[gt_key]
                pred_content = pred_dict[pred_key]
                init_vid_len = len(gt_content)
                assert len(pred_content) >= test_num_samples

                # GO THROUGH TEST SAMPLES
                past_len = int(obs_perc * init_vid_len)  
                for i in range(len(eval_percentages)):
                    eval_perc = eval_percentages[i]
                    eval_len = int((eval_perc + obs_perc) * init_vid_len)

                    # MAX eval
                    max_classes_n_T = np.zeros(self.num_classes)
                    max_classes_n_F = np.zeros(self.num_classes)
                    max_moc = -1.0


                    # DIVERSITY
                    frame_ed = 0
                    p_n = 0
                    for s1 in range(test_num_samples - 1):
                        for s2 in range(s1 + 1, test_num_samples):
                            assert s1 != s2
                            rec_1 = pred_content[s1]
                            rec_2 = pred_content[s2]
                            f_ed = frame_wise(rec_1[past_len:eval_len], rec_2[past_len:eval_len])
                            frame_ed += f_ed
                            p_n += 1
                    frame_div_all_files[i] += (frame_ed / p_n)


                    # ACCURACY
                    for s in range(test_num_samples):
                        recognition = pred_content[s]
                        recognition = recognition[:eval_len]
                        _, _, classes_n_T, classes_n_F, _ = eval_file(
                            gt_content, recognition, past_len, actions_dict
                        )

                        # MEAN MOC
                        n_T_classes_all_files[i] += classes_n_T
                        n_F_classes_all_files[i] += classes_n_F

                        count_mask = (classes_n_T + classes_n_F) != 0
                        moc = np.mean(
                            classes_n_T[count_mask]
                            / (classes_n_T[count_mask] + classes_n_F[count_mask])
                        )
                        if moc > max_moc:
                            max_moc = moc
                            max_classes_n_T = classes_n_T
                            max_classes_n_F = classes_n_F

                    max_n_T_classes_all_files[i] += max_classes_n_T
                    max_n_F_classes_all_files[i] += max_classes_n_F



            # LOG
            results = ""
            for j in range(len(eval_percentages)):
                acc, n = 0, 0
                max_acc, max_n = 0, 0

                # MoC (Mean over Classes metric)
                for i in range(self.num_classes):
                    if n_T_classes_all_files[j, i] + n_F_classes_all_files[j, i] != 0:
                        acc += float(n_T_classes_all_files[j, i]) / (
                            n_T_classes_all_files[j, i] + n_F_classes_all_files[j, i]
                        )
                        n += 1

                    if (
                        max_n_T_classes_all_files[j, i]
                        + max_n_F_classes_all_files[j, i]
                        != 0
                    ):
                        max_acc += float(max_n_T_classes_all_files[j, i]) / (
                            max_n_T_classes_all_files[j, i] + max_n_F_classes_all_files[j, i]
                        )
                        max_n += 1


                # LOG
                results += (
                    "\nPred perc "
                    + str(int(100 * eval_percentages[j]))
                    + " , MoC:  %.4f" % (float(acc) / n)
                    + " , Max MoC:  %.4f" % (float(max_acc) / max_n)
                    + " , Frame Div: %.4f" % (float(frame_div_all_files[j]) / len(gt_keys))
                )
        return results
