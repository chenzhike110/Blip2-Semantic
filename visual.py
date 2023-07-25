import os
os.environ["CUDA_VISIBLE_DEVICES"]="3"
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from lavis.models import load_model_and_preprocess
# import matplotlib
# matplotlib.use('TKAgg')
import hypertools as hyp
import matplotlib.pyplot as plt

# setup device to use
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
# loads BLIP-2 pre-trained model
model, vis_processors, text_processors = load_model_and_preprocess(name="blip2_feature_extractor", model_type="pretrain_vitL", is_eval=True, device=device)

datapath = "/data2/czk/MVNet/dataset/processed/"
motions = []
for motion in os.listdir(datapath):
    if "_prompt" in motion:
        continue
    motions.append(motion)
features_image_origin = None
features_image_new = None
characterlabels = None
motionlabels = None

query = torch.load("./LAVIS/saved/query_9_1.7022453546524048.pt", map_location=torch.device('cpu')).to(device)

for i in tqdm(range(len(motions))):
    raw_image = np.load(os.path.join(datapath, motions[i]))
    for index in range(raw_image.shape[1]):
        images = []
        for j in range(raw_image.shape[0]):
            arr2im = Image.fromarray(raw_image[j, index, ...])
            image = vis_processors["eval"](arr2im).unsqueeze(0).to(device)
            images.append(image)
        if len(images) == 1:
            continue
        images = torch.stack(images, dim=0).squeeze()
        
        with torch.no_grad():
            # originals
            features_image = model.extract_features_diff({"image": images}, model.query_tokens.clone())
            if features_image_origin is None:
                features_image_origin = features_image.detach().cpu()
            else:
                features_image_origin = torch.cat((features_image_origin, features_image.detach().cpu()), 0)
            # trained query
            features_image = model.extract_features_diff({"image": images}, query)
            if features_image_new is None:
                features_image_new = features_image.detach().cpu()
            else:
                features_image_new = torch.cat((features_image_new, features_image.detach().cpu()), 0)
            
            if characterlabels is None:
                characterlabels = torch.linspace(0, raw_image.shape[0]-1, raw_image.shape[0])
            else:
                characterlabels = torch.cat((characterlabels, torch.linspace(0, raw_image.shape[0]-1, raw_image.shape[0])), 0)
            
            if motionlabels is None:
                motionlabels = torch.ones(raw_image.shape[0])*i
            else:
                motionlabels = torch.cat((motionlabels, torch.ones(raw_image.shape[0])*i), 0)

features_image_origin = features_image_origin.view(features_image_origin.shape[0], -1).numpy()
features_image_new = features_image_new.view(features_image_new.shape[0], -1).numpy()
characterlabels = characterlabels.long().numpy()
motionlabels = motionlabels.long().numpy()

np.save("./LAVIS/features_image_origin.npy", features_image_origin)
np.save("./LAVIS/features_image_new.npy", features_image_new)
np.save("./LAVIS/characterlabels.npy", characterlabels)
np.save("./LAVIS/motionlabels.npy", motionlabels)

hyp.plot(features_image_origin, '.', reduce='TSNE', hue=characterlabels, ndims=2, save_path="./LAVIS/origin_character.png", show=False)
hyp.plot(features_image_origin, '.', reduce='TSNE', hue=motionlabels, ndims=2, save_path="./LAVIS/origin_motion.png", show=False)
hyp.plot(features_image_new, '.', reduce='TSNE', hue=characterlabels, ndims=2, save_path="./LAVIS/new_character.png", show=False)
hyp.plot(features_image_new, '.', reduce='TSNE', hue=motionlabels, ndims=2, save_path="./LAVIS/new_motion.png", show=False)
# hyp.plot(features_image_origin, '.', reduce='TSNE', hue=motionlabels, ndims=2)