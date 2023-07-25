import os
os.environ["CUDA_VISIBLE_DEVICES"]="2"
import torch
import numpy as np
from PIL import Image
from lavis.models import load_model_and_preprocess


# setup device to use
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
# state = torch.load("/home/dhz/.cache/torch/hub/checkpoints/instruct_blip_flanxxl_trimmed.pth")
# print(state["model"].keys())
# device = torch.device("cpu")
# loads BLIP-2 pre-trained model
model, vis_processors, text_processors = load_model_and_preprocess(name="blip2_feature_extractor", model_type="pretrain_vitL", is_eval=True, device=device)
model = model.to(device)
# image
img_path = "../MVNet/dataset/processed/45_Degree_Right_Turn_16.npy"
raw_image = np.load(img_path)
arr2im = Image.fromarray(raw_image[0, 0, ...])
image = vis_processors["eval"](arr2im).unsqueeze(0).to(device)
# text
text_input = text_processors["eval"]("Describe the image in details.")
feature = model.extract_features({"image": image, "text_input": [text_input]})
print(feature)