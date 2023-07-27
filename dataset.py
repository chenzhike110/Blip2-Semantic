import os
import torch
import numpy as np
from torch.utils.data import Dataset

class MaximoDataset(Dataset):
    """
    Maximo video dataset
    """
    def __init__(self, path, preprocess, batch_size, train_mask = ["YBot", "XBot"]) -> None:
        super().__init__()
        self.path = path
        self.datalist = []
        self.batch_size = batch_size
        self.train_mask = train_mask
        self.preprocess = preprocess
        self.reset()

    def reset(self):
        self.size = 0
        self.current_index = 0
        for file in os.listdir(self.path):
            if file.endswith(".npy"):
                self.datalist.append(os.path.join(self.path, file))
                self.size = max(self.size, int(file.split("_")[1].strip(".npy")))
        with open(os.path.join(self.path, "labels.txt")) as f:
            self.labels = f.readlines()
        self.datalist.sort(key=lambda x: int(x.split("_")[1].strip(".npy")))
        self.data = np.load(self.datalist[0])
    
    def __len__(self):
        return  self.size // self.batch_size
    
    def __getitem__(self, index):
        images = []
        for i in range(index*self.batch_size, (index+1)*self.batch_size):
            if i >= int(self.datalist[0].split("_")[1].strip(".npy")):
                self.current_index += self.data.shape[0]
                self.data = np.load(self.datalist[1])
                self.datalist = self.datalist[1:]
            images.append(self.preprocess(self.data[i-self.current_index]))
        images = torch.stack(images, dim=0).squeeze()
        mask_ = [label.split("/")[0] in self.train_mask for label in self.labels[index*self.batch_size:(index+1)*self.batch_size]]
        labels = [os.path.join(*label.split("/")[1:]) for label in self.labels[index*self.batch_size:(index+1)*self.batch_size]]
        label_unque = set(labels)
        label_n = torch.zeros(len(labels)).to(images.device)
        labels = np.array(labels)
        for index, label in enumerate(label_unque):
            mask = torch.tensor(labels == label).view_as(label_n)
            label_n[mask] = index

        return {"image":images, "label":label_n.long(), "mask": torch.tensor(mask_)}

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