import os
import torch
import numpy as np
from torch.utils.data import Dataset

class PromptDataset(Dataset):
    """
    Maximo video dataset with prompt
    """
    def __init__(
        self,
        path,
        vis_processor,
        max_length=30,
    ) -> None:
        super().__init__()
        self.path = path
        self.preprocess = vis_processor
        self.max_length = max_length
        self.reset()

    def reset(self):
        self.motion_length = 0
        self.motions = []
        for motion in os.listdir(self.path):
            if "_prompt" in motion:
                continue
            # if "Turning_90_Degrees_Left_With_Bow_17" == motion.split(".npy")[0]:
            #     continue
            self.motions.append(motion.split(".npy")[0])
            shape = int(self.motions[-1].split("_")[-1])
            self.motion_length += shape
        # self.motions = ["Turning_90_Degrees_Left_With_Bow_17"]
        # self.character_num = np.load(os.path.join(path, self.motions[0]+".npy")).shape[0]
        self.data_queue = np.load(os.path.join(self.path, self.motions[0]+".npy"))
        self.motions = self.motions[1:]
        # self.characterpairs = list(combinations(range(character_num), pair_num))
    
    def __len__(self):
        return self.motion_length // self.max_length
    
    def fill_data(self):
        while self.data_queue.shape[1] < self.max_length:
            if len(self.motions) == 0:
                break
            data = np.load(os.path.join(self.path, self.motions[0]+".npy"))
            shape = int(self.motions[0].split("_")[-1])
            if data.shape[0] != self.data_queue.shape[0]:
                self.motions = self.motions[1:]
                continue
            assert data.shape[1] == shape, self.motions[0]+" name error"
            # print(self.motions[0], shape)
            self.data_queue = np.concatenate([self.data_queue, data], axis=1)
            self.motions = self.motions[1:]
    
    def __getitem__(self, index):
        self.fill_data()
        data = self.data_queue[:, :self.max_length]
        self.data_queue = self.data_queue[:, self.max_length:]
        motion = self.preprocess(data)
        labels = torch.linspace(1, motion.size(1), motion.size(1)).repeat(motion.size(0), 1).long()
        
        return {"motion":motion, "label":labels}