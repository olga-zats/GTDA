import torch
import numpy as np
from torch.utils.data import Dataset
import os


class BatchGeneratorTCN(Dataset):
    def __init__(
        self, mode, actions_dict, sample_rate, vid_list_file, pred_perc, obs_perc, args
    ):
        print(f"Dataset: {args.ds}")
        print(f"Mode : {mode}")
        print(f"Vid list file : {vid_list_file}")
        print(f"Observation perc : {obs_perc}")

        # Set params
        self.mode = mode
        self.num_classes = args.num_classes
        self.actions_dict = actions_dict
        self.dataset = args.ds

        self.sample_rate = sample_rate
        self.obs_perc = obs_perc
        self.pred_perc = pred_perc
        self.part_obs = args.part_obs

        # sanity check
        if self.mode != "train":
            assert self.obs_perc != 0

        self.features_path = args.features_path
        self.gt_path = args.gt_path

        # Annotations
        file_ptr = open(vid_list_file, "r")
        list_of_vid = file_ptr.read().split("\n")[:-1]
        file_ptr.close()

        self.list_of_examples = list()
        for vid in list_of_vid:
            if obs_perc != 0:
                self.list_of_examples.append([vid, obs_perc])
            else:
                assert self.mode == 'train'
                self.list_of_examples.append([vid, 0.2])
                self.list_of_examples.append([vid, 0.3])
                self.list_of_examples.append([vid, 0.5])
               

    def __len__(self):
        return len(self.list_of_examples)


    def label_to_id(self, content):
        classes = np.zeros(len(content))
        for i in range(len(content)):
            classes[i] = self.actions_dict[content[i]]
        return classes


    def __getitem__(self, idx):
        # Load feats and anns
        file_name = self.list_of_examples[idx][0].split(".")[0]
        features_name = os.path.join(self.features_path, file_name + ".npy")
        gt_name = os.path.join(self.gt_path, self.list_of_examples[idx][0])

        assert os.path.exists(features_name), "{} features file not found".format(features_name)
        assert os.path.exists(gt_name), "{} annot. file not found".format(gt_name)

        features = np.load(features_name)  # D x T
        file_ptr = open(gt_name, "r")
        content = file_ptr.read().split("\n")[:-1]
        vid_len = len(content)

        # Obs limit
        obs_percentage = self.list_of_examples[idx][1]
        if self.obs_perc == 0:
            assert self.mode == "train"
            # rand aug
            if np.random.random() < 0.4:
                obs_percentage = 0.15 + 0.25 * np.random.random()
        else:
            assert obs_percentage in [0.2, 0.3]
        obs_lim = int(obs_percentage * len(content))

        # Pred limit
        pred_percentage = 1.0 - obs_percentage
        if self.part_obs:
            pred_percentage = 0.5
        pred_lim = int((obs_percentage + pred_percentage) * len(content))
        assert pred_lim <= len(content)


        """ ANNOTATIONS """
        # all labels
        content_past_future = self.label_to_id(content)
        content_past_future = content_past_future[:pred_lim]

        # masks
        mask_past = np.zeros(len(content_past_future))
        mask_past[:obs_lim] = 1
        mask_past = mask_past[:: self.sample_rate]

        mask_future = np.zeros(len(content_past_future))
        mask_future[obs_lim:] = 1
        mask_future = mask_future[:: self.sample_rate]

        # classes (one hot)
        classes_one_hot = np.zeros((self.num_classes, len(content_past_future)))  # C x T
        for i in range(len(content_past_future)):
            classes_one_hot[int(content_past_future[i])][i] = 1
        classes_one_hot = classes_one_hot[:, :: self.sample_rate]


        """ FEATURES """
        # features
        features_past = features[:, :obs_lim]
        features_past = features_past[:, ::self.sample_rate]

        # all classes
        content = self.label_to_id(content)
        assert len(content) == vid_len
        content_past_future = content_past_future[::self.sample_rate]


        sample = {
            "features": features_past,
            "classes": content_past_future,
            "classes_all": content,
            "classes_one_hot": classes_one_hot,
            "mask_past": mask_past,
            "mask_future": mask_future,
            "vid_len": vid_len,
            "file_name": file_name,
        }
        return sample



    def custom_collate(self, batch):
        """COLLECT"""
        # Features
        batch_features = [item["features"] for item in batch]

        # Labels
        batch_classes = [item["classes"] for item in batch]
        batch_classes_all = [item["classes_all"] for item in batch]
        batch_classes_one_hot = [item["classes_one_hot"] for item in batch]

        # Masks
        batch_mask_past = [item["mask_past"] for item in batch]
        batch_mask_future = [item["mask_future"] for item in batch]

        # META INFO
        file_names = np.asarray([item["file_name"] for item in batch])
        batch_vid_len = [item["vid_len"] for item in batch]


        """ PAD """
        # PADDING (based on the LONGEST sequence in the batch)
        bz = len(batch_features)
        length_of_seq = list(map(len, batch_classes))
        length_of_seq_all = list(map(len, batch_classes_all))

        features_tensor = torch.zeros(bz, np.shape(batch_features[0])[0], max(length_of_seq), dtype=torch.float)  # B x D x T_max
        
        classes_tensor = torch.ones(bz, max(length_of_seq), dtype=torch.long) * (self.num_classes + 1)
        classes_tensor_all = torch.ones(bz, max(length_of_seq_all), dtype=torch.long) * (self.num_classes + 1)
        classes_tensor_one_hot = torch.zeros(bz, self.num_classes, max(length_of_seq), dtype=torch.float)

        mask_tensor = torch.zeros(bz, 1, max(length_of_seq), dtype=torch.float)
        mask_past_tensor = torch.zeros(bz, 1, max(length_of_seq), dtype=torch.float)
        mask_future_tensor = torch.zeros(bz, 1, max(length_of_seq), dtype=torch.float)

        for i in range(bz):
            features_tensor[i, :, : np.shape(batch_features[i])[1]] = torch.from_numpy(batch_features[i])

            classes_tensor[i, : np.shape(batch_classes[i])[0]] = torch.from_numpy(batch_classes[i])
            classes_tensor_all[i, : np.shape(batch_classes_all[i])[0]] = torch.from_numpy(batch_classes_all[i])
            classes_tensor_one_hot[i, :, : np.shape(batch_classes_one_hot[i])[1]] = torch.from_numpy(batch_classes_one_hot[i])

            mask_tensor[i, 0, : np.shape(batch_classes[i])[0]] = torch.ones(np.shape(batch_classes[i])[0])
            mask_past_tensor[i, 0, : np.shape(batch_mask_past[i])[0]] = torch.from_numpy(batch_mask_past[i])
            mask_future_tensor[i, 0, : np.shape(batch_mask_future[i])[0]] = torch.from_numpy(batch_mask_future[i])

        # SORT BY LENGTH and PERMUTE
        # lengths = torch.tensor(length_of_seq)
        vid_lengths = torch.tensor(batch_vid_len)
        _, perm_idx = torch.sort(torch.tensor(length_of_seq), 0, descending=True)
        vid_lengths = vid_lengths[perm_idx]

        features_tensor = features_tensor[perm_idx]

        classes_tensor = classes_tensor[perm_idx]
        classes_tensor_all = classes_tensor_all[perm_idx]
        classes_tensor_one_hot = classes_tensor_one_hot[perm_idx]

        mask_tensor = mask_tensor[perm_idx]
        mask_past_tensor = mask_past_tensor[perm_idx]
        mask_future_tensor = mask_future_tensor[perm_idx]

        # META INFO
        file_names = file_names[perm_idx.tolist()]
        meta_dict = {"file_names": file_names}

        return (
            features_tensor,
            classes_tensor,
            classes_tensor_all,
            classes_tensor_one_hot,
            mask_tensor,
            mask_past_tensor,
            mask_future_tensor,
            vid_lengths,
            meta_dict,
        )
