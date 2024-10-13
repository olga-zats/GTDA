
from os.path import isfile, join
import torch.nn.functional as F
import torch
import numpy as np
from torch.utils.data import Dataset
import os
from tqdm import tqdm
from numpy.lib.format import open_memmap
import pandas as pd
import lmdb
from copy import copy

import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))


VIEWS = ['C10095_rgb', 'C10115_rgb', 'C10118_rgb', 'C10119_rgb', 'C10379_rgb', 'C10390_rgb', 'C10395_rgb', 'C10404_rgb',
         'HMC_21176875_mono10bit', 'HMC_84346135_mono10bit', 'HMC_21176623_mono10bit', 'HMC_84347414_mono10bit',
         'HMC_21110305_mono10bit', 'HMC_84355350_mono10bit', 'HMC_21179183_mono10bit', 'HMC_84358933_mono10bit']


class BatchGeneratorAssembly101TCN(Dataset):
    def __init__(self, sample_rate,  mode='train', obs_perc=0., args=None):

        self.features_path = args.features_path
        self.gt_path = args.gt_path
        self.toy_to_idx = None
       
        # DATA LOADING
        self.env = None
        self.frames_format = "{}/{}_{:010d}.jpg"

        self.mode = mode
        self.sample_rate = sample_rate
        self.part_obs = args.part_obs

        self.args = args
        self._load_type = args.load_type
     
        assert mode in [
            "train",
            "val",
        ], "Split '{}' not supported".format(mode)


        print()
        print('Assembly dataset')
        print(f'Dataset mode : {mode}')
        print(f'Sample rate : {sample_rate}')

    
        # how much video is allowed to be observed
        self.obs_perc = obs_perc
        self.obs_perc_list = [.2, .3, .5] 
        if obs_perc != 0:
            self.obs_perc_list = [obs_perc]
        if self.mode != 'train':
            assert obs_perc != 0
        print(f'Obs perc : {obs_perc}')


        self._construct_loader()
        print(f'Number {mode} input sequences is: {len(self._segm_path)}')
        print()



    def __len__(self):
        """
        Returns:
            (int): the number of videos or features in the dataset.
        """
        return len(self._segm_path)



    def _construct_loader(self):
        """
        Construct the list of features and segmentations.
        """

        ''' ANNOTATIONS '''
        annotation_path = os.path.join(self.gt_path, "annotations", "coarse-annotations")
        path_to_csv_file = os.path.join("./datasets/assembly101" , f"{self.mode}.csv")
        assert os.path.exists(path_to_csv_file), "{} file not found".format(path_to_csv_file)
        assert os.path.exists(annotation_path), "{} file not found".format(annotation_path)

        data_df = pd.read_csv(path_to_csv_file)
        actions_df = pd.read_csv(join(annotation_path, "actions.csv"))
        self.num_classes = len(actions_df)
        print(f'Num classes : {self.num_classes}')
        self.action_to_idx = {row['action_cls']:row['action_id'] for i, row in actions_df.iterrows()}


        # Accumulators
        self._segm_path = []
        self._segmentations = []
        self._start_frames = []
        self._end_frames = []
        self._seq_lens = []
        self._obs_percs = []
        self._feat_path = []
        self._meta_inf = []  


        for _, entry in tqdm(data_df.iterrows(), total=len(data_df)):
            sample = entry.to_dict()

            feature_path = None
            if self._load_type == 'numpy':
                feature_path = join(self.features_path,
                                    "TSM_features", 
                                    entry['video_id'],
                                    entry['view'],
                                    "features.npy")
                if not isfile(feature_path):
                    print(f"Numpy feature map "
                         f"{sample['action_type']}_{sample['video_id']}_{sample['view']} not found.")
                    continue

            # LOAD ANNS
            segm_filename = f"{sample['action_type']}_{sample['video_id']}.txt"
            segm_path = join(annotation_path, "coarse_labels", segm_filename)
            segm, start_frame, end_frame = self._load_segmentations(segm_path, actions_df)
            if len(segm) == 0:
                continue
            if end_frame - start_frame < 1:
                continue


            # COLLECT
            for obs_p in self.obs_perc_list:
                self._obs_percs.append(obs_p)
                self._end_frames.append(min(end_frame, sample['video_end_frame']))
                self._start_frames.append(start_frame)
                self._segmentations.append(segm)
                self._segm_path.append(segm_path)
                self._meta_inf.append([sample['video_id'], sample['view'], sample['action_type']]) 

                if self.features_path is not None:
                    self._feat_path.append(feature_path)


    def _load_segmentations(self, segm_path, actions_df):
        segment_labels = []
        start_indices = []
        end_indices = []

        with open(segm_path, 'r') as f:
            lines = list(map(lambda s: s.split("\n"), f.readlines()))
            for line in lines:
                start, end, lbl = line[0].split("\t")[:-1]
                start_indices.append(int(start))
                end_indices.append(int(end))
                action_id = actions_df.loc[actions_df['action_cls'] == lbl, 'action_id']
                segm_len = int(end) - int(start)
                segment_labels.append(np.full(segm_len, fill_value=action_id.item()))

        segmentation = np.concatenate(segment_labels)
        num_frames = segmentation.shape[0]

        # start and end frame idx @30fps
        start_frame = min(start_indices)
        end_frame = max(end_indices)
        assert num_frames == (end_frame-start_frame), \
            "Length of Segmentation doesn't match with clip length."

        return segmentation, start_frame, end_frame



    def _load_features(self, data_dict):
        if self._load_type == 'numpy':
            features = open_memmap(data_dict['feat_path'], mode='r')  # [D, T]
            features = features[:, data_dict['start_frame']:data_dict['end_frame']]  # [D, T]
        else:
            pass
        return features


    def __getitem__(self, index):

        # LOAD DATA and FEATURES
        content = self._segmentations[index]
        obs_percentage = self._obs_percs[index]

        # training
        if self.obs_perc == 0:
            assert self.mode == 'train'
            if np.random.random() < .4:
                obs_percentage = 0.15 + .25 * np.random.random()

        pred_percentage = 1. - obs_percentage
        if self.part_obs:
            pred_percentage = 0.5
        assert pred_percentage + obs_percentage <= 1.

        vid_len = len(content)

        # META INFO
        filename = self._meta_inf[index][0]
        view = self._meta_inf[index][1]
        action_type = self._meta_inf[index][2]


        # LABELS
        obs_lim = int(obs_percentage * len(content))
        pred_lim = int((obs_percentage + pred_percentage) * len(content))
        content_past_future = content[:pred_lim]

        # MASKS
        mask_past = np.zeros(len(content_past_future))
        mask_past[:obs_lim] = 1
        mask_past = mask_past[::self.sample_rate]

        mask_future = np.zeros(len(content_past_future))
        mask_future[obs_lim:] = 1
        mask_future = mask_future[::self.sample_rate]


        # ONE-HOT LABELS
        classes_one_hot = np.zeros((self.num_classes, len(content_past_future)))  # C x T
        for i in range(int(len(content_past_future))):
            classes_one_hot[content[i]][i] = 1
        classes_one_hot = classes_one_hot[:, ::self.sample_rate]


        # FEATURES
        data_dict = {}
        start_frame = self._start_frames[index]
        end_frame = self._end_frames[index]
        data_dict['start_frame'] = start_frame
        data_dict['end_frame'] = end_frame
        if self._load_type == 'lmdb':
            pass
        else:
            data_dict['feat_path'] = self._feat_path[index]


        features = self._load_features(data_dict)
        features = features[:, :int(obs_percentage*len(content))]
        features = features[:, ::self.sample_rate]


        # GATHER
        assert len(content) == vid_len
        sample = {'features': features, 
                  'classes': content_past_future[::self.sample_rate],
                  'classes_all':content,
                  'classes_one_hot': classes_one_hot,
                  'mask_past': mask_past,
                  'mask_future': mask_future,
                  'vid_len': vid_len,
                  'file_name': action_type  + '_'+ filename + '_' + view}
        return sample
    

    def custom_collate(self, batch):
        batch_vid_len = [item['vid_len'] for item in batch]
        batch_features = [item['features'] for item in batch]

        batch_classes = [item['classes'] for item in batch]
        batch_classes_all = [item['classes_all'] for item in batch]
        batch_classes_one_hot = [item['classes_one_hot'] for item in batch]
     
        batch_mask_past = [item['mask_past'] for item in batch]
        batch_mask_future = [item['mask_future'] for item in batch]

        bz = len(batch_features)
        length_of_seq = list(map(len, batch_classes))
        length_of_seq_all = list(map(len, batch_classes_all))

      
        # META INFO
        file_names = np.asarray([item['file_name'] for item in batch])

        # PADDING
        features_tensor = torch.zeros(bz, np.shape(batch_features[0])[0], max(length_of_seq), dtype=torch.float)

        classes_tensor = torch.ones(bz, max(length_of_seq), dtype=torch.long) * (self.num_classes + 1)
        classes_tensor_all = torch.ones(bz, max(length_of_seq_all), dtype=torch.long) * (self.num_classes + 1)
        classes_tensor_one_hot = torch.zeros(bz, self.num_classes, max(length_of_seq), dtype=torch.float)
     
        mask_tensor = torch.zeros(bz, 1, max(length_of_seq), dtype=torch.float)
        mask_past_tensor = torch.zeros(bz, 1, max(length_of_seq), dtype=torch.float)
        mask_future_tensor = torch.zeros(bz, 1, max(length_of_seq), dtype=torch.float)

        for i in range(bz):
            features_tensor[i, :, :np.shape(batch_features[i])[1]] = torch.from_numpy(copy(batch_features[i]))
          
            classes_tensor[i, :np.shape(batch_classes[i])[0]] = torch.from_numpy(copy(batch_classes[i]))
            classes_tensor_all[i, :np.shape(batch_classes_all[i])[0]] = torch.from_numpy(copy(batch_classes_all[i]))
            classes_tensor_one_hot[i, :, :np.shape(batch_classes_one_hot[i])[1]] = torch.from_numpy(copy(batch_classes_one_hot[i]))
          
            mask_tensor[i, 0, :np.shape(batch_classes[i])[0]] = torch.ones(np.shape(batch_classes[i])[0])
            mask_past_tensor[i, 0, :np.shape(batch_mask_past[i])[0]] = torch.from_numpy(copy(batch_mask_past[i]))
            mask_future_tensor[i, 0, :np.shape(batch_mask_future[i])[0]] = torch.from_numpy(copy(batch_mask_future[i]))


        # SORT BY LENGTH
        lengths = torch.tensor(length_of_seq)
        vid_lengths = torch.tensor(batch_vid_len)
        _, perm_idx = torch.sort(torch.tensor(length_of_seq), 0, descending=True)
   

        # PERMUTE
        vid_lengths = vid_lengths[perm_idx]
        features_tensor = features_tensor[perm_idx]

        classes_tensor = classes_tensor[perm_idx]
        classes_tensor_all = classes_tensor_all[perm_idx]
        classes_tensor_one_hot = classes_tensor_one_hot[perm_idx]

        mask_tensor = mask_tensor[perm_idx]
        mask_past_tensor = mask_past_tensor[perm_idx]
        mask_future_tensor = mask_future_tensor[perm_idx]
        lengths = lengths[perm_idx]
      
        # META INFO
        file_names = file_names[perm_idx.tolist()]
        meta_dict = {'file_names': file_names}

        return (features_tensor, 
                classes_tensor,
                classes_tensor_all, 
                classes_tensor_one_hot,
                mask_tensor,
                mask_past_tensor,
                mask_future_tensor, 
                vid_lengths,
                meta_dict)



