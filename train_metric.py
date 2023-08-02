import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
from dataset import MaximoDataset
from torch.autograd import Variable
import torchvision.transforms as transforms

from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from lavis.models import load_model_and_preprocess

from pytorch_metric_learning.losses import TripletMarginLoss, NTXentLoss
from pytorch_metric_learning.distances import CosineSimilarity
from loss import center_loss, square_loss

if __name__ == "__main__":
    parser = argparse.ArgumentParser("trainning text query arguments")
    parser.add_argument('--datapath', default="/data2/czk/MVNet/dataset_merge/", type=str, help='Path to dataset file')
    parser.add_argument('--batch_size', default=128, type=int, help='batch size for metrics learning')
    parser.add_argument('--train_mask', default=["Aj", "David", "Ganny", "Joe", "Kaya", "XBot", "shannon", "YBot"], type=float, help='training characters')
    parser.add_argument('--optimize_query', action='store_true', help='optimize text query or qformer')
    # parser.add_argument('--random_start', action='store_true', help='random initialize the text query')
    # parser.add_argument('--model_name', default="blip2_feature_extractor", help="finetured model")
    parser.add_argument('--loss_type', default="NXT", type=str)
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--lr', default=1e-5, type=float, help='learning rate for the optimizer')
    args = parser.parse_args()
    
    # print(args.random_start)

    # setup device to use
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    # loads BLIP-2 pre-trained model
    model, vis_processors, text_processors = load_model_and_preprocess(
        name="blip2_feature_extractor", model_type="pretrain_vitL", is_eval=False, device=device
    )
    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        msg = model.load_state_dict(checkpoint, strict=False)
        print(msg)
        model = model.to(device)

    preprocess = transforms.Compose([
        transforms.Lambda(lambda x: Image.fromarray(x)),
        # transforms.Lambda(lambda x: x.resize((224, 224))),
        transforms.Lambda(lambda x: vis_processors["eval"](x).unsqueeze(0).to(device)),
    ])
    dataset = MaximoDataset(args.datapath, preprocess, args.batch_size, args.train_mask)

    writer = SummaryWriter("./runs_{}_{}".format(os.getpid(), args.loss_type))
    if args.loss_type == "NXT":
        loss_fn = NTXentLoss(temperature=0.1)
    else:
        loss_fn = TripletMarginLoss(margin=0.3, distance=CosineSimilarity())
    # loss_fn = NTXentLoss(temperature=0.1)
    iter = 0

    # query_tokens = model.query_tokens.clone()
    # if args.random_start:
    #     text_length = 5
    #     text_id = torch.randint(low=0, high=model.Qformer.bert.embeddings.word_embeddings.num_embeddings, size=(1, text_length))
    #     query = model.Qformer.bert.embeddings.word_embeddings(text_id.long().to(device))
    # else:
    caption = ["Question: Describe the action of the character in detail. Answer:"]
    #     text = model.tokenizer(caption, return_tensors="pt", padding=True).to(device)
    #     text_id = text.input_ids
    #     query = model.Qformer.bert.embeddings.word_embeddings(text_id.long().to(device))
    # query = Variable(query.clone(), requires_grad=True)
    # caption = ["Question: What is the character doing? Answer:"]
    optimizer = torch.optim.Adam(
        [{'params': model.Qformer.parameters()},
         {'params': model.motion_proj.parameters(), 'lr': 1e-3}], 
        lr=args.lr, 
        weight_decay=0.01
    )
    scheduler = CosineAnnealingWarmRestarts(optimizer, 1000, eta_min=1e-6)

    for loop in range(50):
        losses = []
        for i in tqdm(range(len(dataset))):
            data_list = dataset[i]
            optimizer.zero_grad()
            # features_image = model(data_list, torch.cat((query_tokens, query), dim=1))
            # features_image = model(data_list)
            features_image = model.extract_feature(data_list, caption*data_list["image"].shape[0])
            train_feature = features_image[data_list["mask"]]
            train_labels = data_list["label"][data_list["mask"]]
            metric_loss = loss_fn(train_feature.view(train_feature.shape[0], -1), train_labels)
            central_loss = square_loss(train_feature, train_labels)
            # loss = metric_loss
            loss = metric_loss + central_loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            test_loss = square_loss(features_image, data_list["label"], torch.logical_not(data_list["mask"]))
            writer.add_scalar("central_loss", torch.sum(central_loss), iter)
            writer.add_scalar("metric_loss", torch.sum(metric_loss), iter)
            writer.add_scalar("test_loss/epoch_{}".format(loop), torch.sum(test_loss), iter%len(dataset))
            losses.append(torch.sum(test_loss).detach().cpu())
            iter += 1
        dataset.reset()
        losses = torch.tensor(losses).mean()
        if loop % 5 == 0 and loop > 0:
            torch.save(model.state_dict(), "./saved/{}_Qformer_{}_{:.4f}.pt".format(args.loss_type, loop, losses))