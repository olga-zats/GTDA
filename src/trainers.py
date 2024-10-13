# generic
import os
import pickle
import torch
import torch.nn as nn
from torch import optim
import numpy as np
from torch.utils.data import DataLoader
from einops import rearrange
from collections import defaultdict

# models
from models import *
from models_bit_diff import BitDiffPredictorTCN
from bit_diffusion import GaussianBitDiffusion
from ema import *

# utils
from evaluation import *


class TrainerTCN:
    def __init__(self, args, causal=False):
        super(TrainerTCN, self).__init__()

        # PARAMS
        self.num_classes = args.num_classes
        self.obs_stages = args.obs_stages
        self.ant_stages = args.ant_stages        
        self.m_name = args.model
        self.input_dim = args.input_dim

        # MODELS
        if self.m_name == "pred-tcn":
            self.prob = False
            self.model = PredictorTCN(args, causal=causal)
            self.model_dim = args.model_dim

            ''' LOSSES '''
            # Losses
            self.ce = nn.CrossEntropyLoss(ignore_index=self.num_classes+1, reduction='none')

        elif self.m_name == 'bit-diff-pred-tcn':
            self.prob = True
            self.model = BitDiffPredictorTCN(args, causal=causal)
            self.model_dim = args.model_dim
            self.diffusion = GaussianBitDiffusion(
                self.model, 
                condition_x0=args.conditioned_x0,
                num_classes=self.num_classes,
                timesteps=args.num_diff_timesteps,
                ddim_timesteps=args.num_infr_diff_timesteps,
                objective=args.diff_obj,
                loss_type=args.diff_loss_type,
            )



    # ------------------------------------------------------------------- TRAINING ---------------------------------------------------------------
    def train(self,
              args,
              save_dir,
              batch_gen,
              val_batch_gens,
              device,
              num_workers,
              writer,
              results_dir,
              actions_dict):

        # MODEL
        if self.prob:
            self.diffusion = self.diffusion.to(device)
            self.diffusion.train()
            self.ema_diffusion = EMA(model=self.diffusion,
                                    beta=0.995,
                                    update_every=10)
            self.ema_diffusion.eval()
        else:
            self.model.to(device)
            self.model.train()


        # INIT OPTIMIZER & SCHEDULER
        if self.prob:
            optimizer = optim.Adam(params=self.diffusion.parameters(), lr=args.lr)
        else:
            optimizer = optim.Adam(params=list(self.model.parameters()), lr=args.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=.5)


        # TRAIN
        print('Start Training...')
        for epoch in range(args.num_epochs + 1):
            epoch_loss = 0
            epoch_ce_loss = 0
        
            correct, total = 0, 0
            correct_past, total_past = 0, 0
            correct_future, total_future = 0, 0
            
            dataloader = DataLoader(
                batch_gen,
                batch_size=args.bz,
                shuffle=True,
                num_workers=num_workers,
                collate_fn=batch_gen.custom_collate
            )
            
            for itr, sample_batched in enumerate(dataloader):
                batch_total_loss, batch_ce_loss, \
                batch_correct, batch_total, \
                batch_correct_past, batch_total_past, \
                batch_correct_future, batch_total_future = self.train_single_batch(sample_batched, optimizer, device)

                # losses
                epoch_loss += batch_total_loss
                epoch_ce_loss += batch_ce_loss
                
                # acc
                total += batch_total
                correct += batch_correct

                total_past += batch_total_past
                correct_past += batch_correct_past

                total_future += batch_total_future
                correct_future += batch_correct_future


                # log
                if itr % 5 == 0:
                    print("[epoch %f]: epoch loss = %f, ce loss = %f, past acc = %f, fut acc = %f " % (epoch + itr / len(dataloader),
                                                                                                    epoch_loss / (itr + 1),
                                                                                                    epoch_ce_loss / (itr + 1),
                                                                                                    float(correct_past) / total_past,
                                                                                                    float(correct_future) / total_future))

                if self.prob:
                    self.ema_diffusion.update()    


            # LR step
            lr = optimizer.param_groups[0]['lr']
            print(f'Current learning rate : {lr}')
            scheduler.step(epoch_loss)
 

            # Logging
            print("[epoch %d]: epoch loss = %f,  acc = %f" % (epoch, epoch_loss / len(dataloader), float(correct) / total))

            # acc
            writer.add_scalar("training_accuracies/MoF_past", float(correct_past) / total_past, global_step=epoch)
            writer.add_scalar("training_accuracies/MoF_future", float(correct_future) / total_future, global_step=epoch)
            writer.add_scalar("training_accuracies/MoF_all", float(correct) / total, global_step=epoch)
            
            # loss
            writer.add_scalar("training_losses/total_loss",  float(epoch_loss) / len(dataloader), global_step=epoch)
            writer.add_scalar("training_losses/ce_loss", float(epoch_ce_loss) / len(dataloader), global_step=epoch)
        
 

            # EVAL
            if (epoch) % 5 == 0: 
                torch.save(self.model.state_dict(), save_dir + "/epoch-" + str(epoch) + ".model")
                if self.prob:
                    torch.save(self.ema_diffusion.state_dict(), save_dir + "/ema_diff_epoch-" + str(epoch) + ".model")
 
                if not self.prob:
                    obs_percs = [.2, .3,]
                    for i_o, obs_perc in enumerate(obs_percs):
                        returned_metrics = self.validate(args,
                                                        epoch,
                                                        obs_perc,
                                                        val_batch_gens[i_o],
                                                        device,
                                                        num_workers,
                                                        results_dir,
                                                        actions_dict,
                                                        args.sample_rate,
                                                        eval_mode=True)

                        # Logging
                        writer.add_scalar("testing_loss/ce_loss/obs_" + str(obs_perc), returned_metrics[0][-1], global_step=epoch)
                        writer.add_scalar("testing_loss/total_loss/obs_" + str(obs_perc), returned_metrics[0][-2], global_step=epoch)
                        
                        for res in returned_metrics:
                            writer.add_scalar("testing_metrics_" + str(obs_perc) + "/MoF_obs_" + str(obs_perc) + "_pred_" + str(res[0]), res[1], global_step=epoch)
                            writer.add_scalar("testing_metrics_" + str(obs_perc) + "/MoC_obs_" + str(obs_perc) + "_pred_" + str(res[0]), res[2], global_step=epoch)
                            writer.add_scalar("testing_metrics_" + str(obs_perc) + "/Max_MoC_obs_" + str(obs_perc) + "_pred_" + str(res[0]), res[3], global_step=epoch)

                    self.model.train()




    def train_single_batch(self, sample_batched, optimizer, device):
        # INPUT
        # DET
        features_tensor = sample_batched[0]
        classes_tensor = sample_batched[1]

        # PROB
        classes_one_hot_tensor = sample_batched[3]

        mask_tensor = sample_batched[4]
        mask_past_tensor = sample_batched[5]
        mask_future_tensor = sample_batched[6]

        # DEVICE
        features_tensor = features_tensor.to(device)
        classes_tensor = classes_tensor.to(device)
        
        mask_tensor = mask_tensor.to(device)
        mask_past_tensor = mask_past_tensor.to(device)
        mask_future_tensor = mask_future_tensor.to(device)

        # MASKS
        masks = []
        for _ in range(self.obs_stages):
            masks.append(mask_past_tensor)
        for _ in range(self.ant_stages):
            masks.append(mask_tensor)
        

        ''' PREDICTION '''
        # DETERMINISTIC
        if not self.prob:
            tcn_predictions, _ = self.model(features_tensor, masks)  # List[B x C x T], _


            # LOSS
            loss, ce_loss = 0, 0
            for st, p in enumerate(tcn_predictions):
                # CE
                ce = self.ce(p.transpose(2, 1).contiguous().view(-1, self.num_classes), classes_tensor.view(-1))  # BT x C
                ce = torch.sum(ce * masks[st].transpose(2, 1).view(-1)) / torch.sum(masks[st])
                loss += ce
                ce_loss += ce


        # PROB.
        else:
            classes_one_hot_tensor = classes_one_hot_tensor.to(device)
            loss, tcn_predictions = self.diffusion({'mask_past': rearrange(mask_past_tensor, 'b c t -> b t c'),
                                                    'x_0': rearrange(classes_one_hot_tensor, 'b c t -> b t c'),
                                                    'obs': rearrange(features_tensor, 'b c t -> b t c'),
                                                    'masks_stages': [rearrange(mt, 'b c t -> b t c') for mt in masks]})

            ce_loss  = torch.tensor(0.)            


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        ''' RESULTS '''
        # LOSSES
        batch_loss = loss.item()
        ce_loss = ce_loss.item()


        # ACCURASIES
        _, predicted = torch.max(tcn_predictions[-1].data, 1)

        correct = ((predicted == classes_tensor).float() * mask_tensor.squeeze(1)).sum().item()
        correct_past = ((predicted == classes_tensor).float() * mask_past_tensor.squeeze(1)).sum().item()
        correct_future = ((predicted == classes_tensor).float() * mask_future_tensor.squeeze(1)).sum().item()

        total_future = torch.sum(mask_future_tensor).item()
        total_past = torch.sum(mask_past_tensor).item()
        total = torch.sum(mask_tensor).item()

        return batch_loss, ce_loss, \
               correct, total, correct_past, total_past, \
               correct_future, total_future




    # ------------------------------------------------------------------ VALIDATION ---------------------------------------------------------------
    def validate(self,
                 args,
                 epoch,
                 obs_perc,
                 batch_gen,
                 device,
                 num_workers,
                 model_dir,
                 actions_dict,
                 sample_rate,
                 eval_mode):
        
        with torch.no_grad():

            # MODEL
            if not eval_mode:
                self.model.to(device)
                self.model.eval()
                self.model.load_state_dict(torch.load(model_dir + '/epoch-' + str(epoch) + ".model"))
                
                if self.prob:
                    # diff
                    self.diffusion.to(device)
                    self.diffusion.eval()
                    self.diffusion.model.load_state_dict(torch.load(model_dir + '/epoch-' + str(epoch) + ".model"), strict=True)

                    # ema
                    self.ema_diffusion = EMA(model=self.diffusion,
                                            beta=0.995,
                                            update_every=10)
                    self.ema_diffusion.load_state_dict(torch.load(model_dir + "/ema_diff_epoch-" + str(epoch) + ".model"), strict=False)
                    self.ema_diffusion.eval()
            else:
                self.model.eval()


            # PERCS     
            eval_percentages = [.1, .2, .3, .5]


            # LOGGERS
            # Metric loggers
            n_T_classes_all_files = np.zeros((len(eval_percentages), len(actions_dict)))
            n_F_classes_all_files = np.zeros((len(eval_percentages), len(actions_dict)))
            num_frames_all_files = np.zeros(len(eval_percentages))
            errors_frames_all_files = np.zeros(len(eval_percentages))
            if self.prob:
                max_n_T_classes_all_files = np.zeros((len(eval_percentages), len(actions_dict)))
                max_n_F_classes_all_files = np.zeros((len(eval_percentages), len(actions_dict)))
          

            # EVAL
            loss, ce_loss = 0, 0          
            dataloader = DataLoader(
                batch_gen, 
                batch_size=1, 
                shuffle=False, 
                num_workers=args.num_workers, 
                collate_fn=batch_gen.custom_collate
            )


            # SAVE (DIFF PREDS)
            if self.prob:
                if args.ds == 'assembly':
                    ds = 'assembly'
                else:
                    ds = f'{args.ds}_{args.split}'

                result_dict = defaultdict(list)
                result_file = f'./diff_results/{ds}/{args.model}'\
                              f'_epoch_{epoch}'\
                              f'_ns_{args.num_stages}'\
                              f'_ds_{args.num_diff_timesteps}'\
                              f'_ids_{args.num_infr_diff_timesteps}'\
                              f'_num_samples_{args.num_samples}' \
                              f'_dlt_{args.diff_loss_type}' \
                              f'_cond_x0_{args.conditioned_x0}'
                if not os.path.exists(result_file):
                    os.makedirs(result_file)
                result_file += f'/obs_{obs_perc}.pkl'

                

            # ITERATE
            for itr, sample_batched in enumerate(dataloader):

                # DATA
                features = sample_batched[0] 

                classes_tensor = sample_batched[1]  
                classes_all_tensor = sample_batched[2]
                classes_one_hot_tensor = sample_batched[3] 

                mask_past_tensor = sample_batched[5]
                
                # DEVICE
                features = features.to(device)
                classes_tensor = classes_tensor.to(device)
                mask_past_tensor = mask_past_tensor.to(device)
              

                # MASK
                masks = []
                for _ in range(self.obs_stages):
                    masks.append(mask_past_tensor)
                for _ in range(self.ant_stages):
                    masks.append(torch.ones(1, 1, features.size(-1), device=device))                


                ''' PREDICTIONS '''
                # DETERMINISTIC
                if not self.prob:
                    tcn_predictions, _ = self.model(features, masks)


                    ''' LOSSES'''
                    for st, p in enumerate(tcn_predictions):
                        ce = self.ce(p.transpose(2, 1).contiguous().view(-1, self.num_classes), classes_tensor.view(-1))  # BT x C
                        ce = torch.sum(ce * masks[st].transpose(2, 1).view(-1)) / torch.sum(masks[st])
                    
                        loss += ce.item()
                        ce_loss += ce.item()
                
                # PROB
                else:
                    classes_one_hot_tensor = classes_one_hot_tensor.to(device)
                    tcn_predictions = self.ema_diffusion.ema_model.predict(
                            x_0 = rearrange(classes_one_hot_tensor, 'b c t -> b t c'),
                            obs = rearrange(features, 'b c t -> b t c'),
                            mask_past = rearrange(mask_past_tensor, 'b c t -> b t c'),
                            masks_stages = [rearrange(mask_tensor, 'b c t -> b t c') for mask_tensor in masks],
                            n_samples=args.num_samples,
                            n_diffusion_steps=args.num_infr_diff_timesteps)
                    tcn_predictions = tcn_predictions.contiguous()
                    loss = 0.
                    ce_loss = 0.


                ''' ACCURACIES '''
                # META INFO
                init_vid_len = sample_batched[-2]
                meta_dict = sample_batched[-1]
                file_names = meta_dict['file_names']    

                # FINAL PRED AND GT
                gt_content = classes_all_tensor[0].numpy()
                assert len(gt_content) == init_vid_len


                # ACCUM PREDICTIONS
                if not self.prob:
                    _, predicted = torch.max(tcn_predictions[-1].data, 1)
                    predicted = predicted.squeeze()
                    tcn_fin_prediction = []
                    for i in range(len(predicted)):
                        tcn_fin_prediction = np.concatenate((tcn_fin_prediction, [predicted[i].item()] * sample_rate))

                else:
                    # iterate through samples
                    tcn_fin_predictions = []
                    for s in range(tcn_predictions.shape[0]):
                        tcn_fin_prediction = []
                        _, predicted = torch.max(tcn_predictions[s].data, 1)
                        predicted = predicted.squeeze()
                        for i in range(len(predicted)):
                            tcn_fin_prediction = np.concatenate((tcn_fin_prediction, [predicted[i].item()] * sample_rate))

                        # save / accumulate
                        result_dict[file_names[0]].append(tcn_fin_prediction)  
                        tcn_fin_predictions.append(tcn_fin_prediction)

                    result_dict[f'gt_{file_names[0]}'] = classes_all_tensor[0].numpy()
                    tcn_fin_predictions = np.stack(tcn_fin_predictions, axis=0)
                    

                # COMPUTE EVAL METRICS
                past_len = int(obs_perc * init_vid_len)  # observation length
                for i in range(len(eval_percentages)):
                    eval_perc = eval_percentages[i]
                    eval_len = int((eval_perc + obs_perc) * init_vid_len)

                    if not self.prob:
                        errors_frames, num_frames, classes_n_T, classes_n_F, _ = eval_file(
                            gt_content, 
                            tcn_fin_prediction[:eval_len],
                            past_len,
                            actions_dict
                        )
                        n_T_classes_all_files[i] += classes_n_T
                        n_F_classes_all_files[i] += classes_n_F
                        num_frames_all_files[i] += num_frames
                        errors_frames_all_files[i] += errors_frames


                    else:

                        # Top-1 MoC
                        max_n_T_classes = np.zeros(self.num_classes)
                        max_n_F_classes = np.zeros(self.num_classes) 
                        max_moc = -1.

                        for s in range(tcn_fin_predictions.shape[0]):
                            errors_frames, num_frames, classes_n_T, classes_n_F, _ = eval_file(
                                gt_content,
                                tcn_fin_predictions[s][:eval_len],
                                past_len,
                                actions_dict
                            )
                            n_T_classes_all_files[i] += classes_n_T
                            n_F_classes_all_files[i] += classes_n_F
                            
                        
                            # choose the max one
                            all_pred = classes_n_T + classes_n_F
                            moc = np.mean(classes_n_T[all_pred != 0] / (classes_n_T[all_pred != 0] + classes_n_F[all_pred != 0]))
                            if moc > max_moc:
                                max_moc = moc   
                                max_n_T_classes = classes_n_T        
                                max_n_F_classes = classes_n_F
                               
                        max_n_T_classes_all_files[i] += max_n_T_classes
                        max_n_F_classes_all_files[i] += max_n_F_classes



            ''' SAVE SAMPLED RESULTS '''
            if self.prob:
                result_file_ptr = open(result_file, 'wb')
                pickle.dump(result_dict, result_file_ptr)
                result_file_ptr.close()


            ''' ACCUMULATE RESULTS '''
            loss = loss / len(dataloader)
            ce_loss = ce_loss / len(dataloader)
            results = f"Loss : {round(loss, 4)}, CE Loss : {round(ce_loss, 4)}\n\n"


            returned_metrics = []
            for j in range(len(eval_percentages)):
                acc, n = 0, 0  # Mean MoC
                max_acc, max_n = 0, 0  # Top-1 MoC

                # MoC (Mean over Classes metric)
                for i in range(len(actions_dict)):
                    if n_T_classes_all_files[j, i] + n_F_classes_all_files[j, i] != 0:
                        acc += (float(n_T_classes_all_files[j, i]) / (n_T_classes_all_files[j, i] + n_F_classes_all_files[j, i]))
                        n += 1
                    
                    if self.prob and max_n_T_classes_all_files[j, i] + max_n_F_classes_all_files[j, i] != 0:
                        max_acc += (float(max_n_T_classes_all_files[j, i]) / (max_n_T_classes_all_files[j, i] + max_n_F_classes_all_files[j, i]))
                        max_n += 1  

   

                #  NORMALIZE
                mof = 100 * (1.0 - float(errors_frames_all_files[j]) / num_frames_all_files[j])
                moc = float(acc) / n
           

                # LOG
                results += "Pred perc " + str(int(100 * eval_percentages[j])) + \
                           " , Acc: %.4f" % (100 * (1.0 - float(errors_frames_all_files[j]) / num_frames_all_files[j])) + \
                           " , MoC:  %.4f\n" % (float(acc) / n) 
                if self.prob:
                    results +=  ", Max MoC:  %.4f\n" % (float(max_acc) / max_n)
                    max_moc = float(max_acc) / max_n   
                else:
                    max_moc = 0            

                returned_metrics.append(
                    [
                        eval_percentages[j], \
                        mof, \
                        moc, \
                        max_moc, \
                        loss, \
                        ce_loss
                    ]
                )
            if not eval_mode:
                print(results)
                print()
    
    
        return returned_metrics



0
