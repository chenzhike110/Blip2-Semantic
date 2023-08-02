import os
import torch
import shutil
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image
from dataset import MaximoDataset
import torchvision.transforms as transforms
from lavis.models import load_model_and_preprocess
# import matplotlib
# matplotlib.use('TKAgg')
# import hypertools as hyp
# import matplotlib.pyplot as plt

if __name__ == "__main__":
    parser = argparse.ArgumentParser("trainning text query arguments")
    parser.add_argument('--datapath', default="/data2/czk/MVNet/dataset_all_merge/", type=str, help='Path to dataset file')
    parser.add_argument('--model_name', default="blip2_feature_extractor", help="finetured model")
    parser.add_argument('--batch_size', default=128, type=int, help='batch size for metrics learning')
    parser.add_argument('--query_path', default=None, help="query pt")
    args = parser.parse_args()

    # setup device to use
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    # loads BLIP-2 pre-trained model
    if args.model_name == "blip2_feature_extractor":
        model, vis_processors, text_processors = load_model_and_preprocess(
            name="blip2_feature_extractor", model_type="pretrain_vitL", is_eval=True, device=device
        )
    else:
        model, vis_processors, _ = load_model_and_preprocess(
            name="blip2_t5_instruct", model_type="flant5xxl", is_eval=True, device=device
        )
    if args.query_path is not None:
        checkpoint = torch.load(args.query_path, map_location="cpu")
        msg = model.load_state_dict(checkpoint, strict=False)
        print(msg)
        model = model.to(device)
    preprocess = transforms.Compose([
        transforms.Lambda(lambda x: Image.fromarray(x)),
        transforms.Lambda(lambda x: x.resize((224, 224))),
        transforms.Lambda(lambda x: vis_processors["eval"](x).unsqueeze(0).to(device)),
    ])
    dataset = MaximoDataset(args.datapath, preprocess, args.batch_size, shuffle=False)
    caption = ["Question: Describe the action of the character in detail. Answer:"]
    
    features_image_origin = None

    if os.path.exists("./visualization"):
        shutil.rmtree('./visualization')
        # os.removedirs("./visualization")
    os.mkdir("./visualization")

    for i in tqdm(range(len(dataset))):
        data_list = dataset[i]
        if i%100==0 and i>0:
            features_image_origin = features_image_origin.numpy()
            np.save("./visualization/features_image_origin_{}.npy".format(i), features_image_origin)
            features_image_origin = None
            exit(0)

        with torch.no_grad():
            # origin query
            features_image = model.extract_feature(data_list, caption*data_list["image"].shape[0])
            if features_image_origin is None:
                features_image_origin = features_image.detach().cpu()
            else:
                features_image_origin = torch.cat((features_image_origin, features_image.detach().cpu()), 0)
            # trained query
            # features_image = model(data_list, torch.cat((query_tokens, query), dim=1))
            # if features_image_new is None:
            #     features_image_new = features_image.detach().cpu()
            # else:
            #     features_image_new = torch.cat((features_image_new, features_image.detach().cpu()), 0)

    features_image_origin = features_image_origin.numpy()
    # features_image_new = features_image_new.numpy()
    np.save("./visualization/features_image_origin_end.npy", features_image_origin)
    # np.save("./visualization/features_image_new_end.npy", features_image_new)