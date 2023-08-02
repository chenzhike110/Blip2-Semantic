import os
import torch
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
from dataset import MaximoDataset
from torch.autograd import Variable
import torchvision.transforms as transforms

from torch.utils.tensorboard import SummaryWriter
from lavis.models import load_model_and_preprocess

from pytorch_metric_learning.losses import TripletMarginLoss, NTXentLoss
from pytorch_metric_learning.distances import CosineSimilarity
from loss import center_loss

if __name__ == "__main__":
    parser = argparse.ArgumentParser("trainning text query arguments")
    parser.add_argument('--datapath', default="/data2/czk/MVNet/dataset/", type=str, help='Path to dataset file')
    parser.add_argument('--batch_size', default=36, type=int, help='batch size for metrics learning')
    parser.add_argument('--train_mask', default=["David", "XBot", "Joe", "Aj", "Ganny", "shannon", "Kaya", "YBot"], type=float, help='training characters')
    parser.add_argument('--random_start', action='store_true', help='random initialize the text query')
    parser.add_argument('--model_name', default="blip2_feature_extractor", help="finetured model")
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate for the optimizer')
    args = parser.parse_args()
    
    print(args.random_start)

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
    preprocess = transforms.Compose([
        transforms.Lambda(lambda x: Image.fromarray(x)),
        transforms.Lambda(lambda x: x.resize((224, 224))),
        transforms.Lambda(lambda x: vis_processors["eval"](x).unsqueeze(0).to(device)),
    ])
    dataset = MaximoDataset(args.datapath, preprocess, args.batch_size, args.train_mask)

    writer = SummaryWriter("./runs_{}".format(os.getpid()))
    # loss_fn = TripletMarginLoss(margin=0.3)
    loss_fn = NTXentLoss(temperature=0.07)
    iter = 0

    query_tokens = model.query_tokens.clone()
    caption = "Question: What is the character in the image doing? Answer:"
    text = model.tokenizer(caption, return_tensors="pt", padding=True).to(device)
    text_id = text.input_ids
    query = model.Qformer.bert.embeddings.word_embeddings(text_id.long().to(device))
    query = Variable(query.clone(), requires_grad=True)
    optimizer = torch.optim.Adam([query], lr=args.lr)

    for loop in range(5):
        losses = []
        for i in tqdm(range(len(dataset))):
            data_list = dataset[i]
            optimizer.zero_grad()
            features_image = model(data_list, torch.cat((query_tokens, query), dim=1))
            train_feature = features_image[data_list["mask"]]
            train_labels = data_list["label"][data_list["mask"]]
            metric_loss = loss_fn(train_feature.view(train_feature.shape[0], -1), train_labels)
            central_loss = center_loss(train_feature, train_labels)
            loss = metric_loss + central_loss
            loss.backward()
            losses.append(torch.sum(loss).detach().cpu())
            optimizer.step()
            
            test_loss = center_loss(features_image, data_list["label"], torch.logical_not(data_list["mask"]))
            writer.add_scalar("central_loss", torch.sum(central_loss), iter)
            writer.add_scalar("metric_loss", torch.sum(metric_loss), iter)
            writer.add_scalar("test_loss", torch.sum(test_loss), iter)
            iter += 1
        dataset.reset()
        losses = torch.tensor(losses).mean()
        torch.save(query, "./saved/{}_textquery_{}_{}.pt".format(os.getpid(), loop, losses))