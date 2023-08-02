import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
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
from loss import center_loss

def emb2indices(output, emb_layer):
    # output is size: [batch, sequence, emb_length], emb_layer is size: [num_tokens, emb_length]
    emb_weights = emb_layer.weight

    # get indices from embeddings:
    emb_size = output.size(0), output.size(1), -1, -1
    out_size = -1, -1, emb_weights.size(0), -1
    out_indices = torch.argmin(torch.abs(output.unsqueeze(2).expand(out_size) -
                                    emb_weights.unsqueeze(0).unsqueeze(0).expand(emb_size)).sum(dim=3), dim=2)
    return out_indices

if __name__ == "__main__":
    parser = argparse.ArgumentParser("trainning text query arguments")
    parser.add_argument('--datapath', default="/data2/czk/MVNet/dataset_all/", type=str, help='Path to dataset file')
    parser.add_argument('--batch_size', default=2, type=int, help='batch size for metrics learning')
    parser.add_argument('--train_mask', default=["David", "XBot", "Joe", "Aj", "Ganny", "shannon", "Kaya", "YBot"], type=float, help='training characters')
    parser.add_argument('--random_start', action='store_true', help='random initialize the text query')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate for the optimizer')
    args = parser.parse_args()
    
    print(args.random_start)

    # setup device to use
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    # loads BLIP-2 pre-trained model
    model, vis_processors, _ = load_model_and_preprocess(
        name="blip2_t5_instruct", model_type="flant5xxl", is_eval=True
    )
    preprocess = transforms.Compose([
        transforms.Lambda(lambda x: Image.fromarray(x)),
        # transforms.Lambda(lambda x: x.resize((224, 224))),
        transforms.Lambda(lambda x: vis_processors["eval"](x).unsqueeze(0).to(device)),
    ])
    dataset = MaximoDataset(args.datapath, preprocess, args.batch_size, args.train_mask)

    caption = "Question: Describe the motion of the character in detail? Answer:"
    text = model.tokenizer(caption, return_tensors="pt", padding=True).to(device)
    text_id = text.input_ids
    text_query = model.Qformer.bert.embeddings.word_embeddings(text_id.long().to(device))
    layer = model.Qformer.bert.embeddings.word_embeddings

    for i in tqdm(range(len(dataset))):
        data_list = dataset[i]
        indices = torch.zeros((1, text_query.size(1)))
        for i in range(indices.shape[0]):
            query = text_query[0, i].unsqueeze(0).unsqueeze(0)
            indices[i] = emb2indices(query, layer)
        captions = model.tokenizer.batch_decode(indices, skip_special_tokens=True)
        print(captions)
        data_list.update({
            "text_input":captions*data_list["image"].shape[0], 
        })
        result = model.forward_loss(data_list)

    # writer = SummaryWriter("./runs_{}".format(os.getpid()))
    # loss_fn = TripletMarginLoss(margin=0.3)
    # loss_fn = NTXentLoss(temperature=0.05)
    # iter = 0

    # query_tokens = model.query_tokens.clone()
    # if args.random_start:
    #     text_length = 5
    #     text_id = torch.randint(low=0, high=model.Qformer.bert.embeddings.word_embeddings.num_embeddings, size=(1, text_length))
    #     query = model.Qformer.bert.embeddings.word_embeddings(text_id.long().to(device))
    # else:
    #     caption = "Question: What is the character doing? Answer:"
    #     text = model.tokenizer(caption, return_tensors="pt", padding=True).to(device)
    #     text_id = text.input_ids
    #     query = model.Qformer.bert.embeddings.word_embeddings(text_id.long().to(device))
    # query = Variable(query.clone(), requires_grad=True)
    # optimizer = torch.optim.Adam([query], lr=args.lr, weight_decay=0.05)
    # scheduler = CosineAnnealingWarmRestarts(optimizer, 1000, eta_min=1e-6)

    # for loop in range(5):
    #     losses = []
    #     for i in tqdm(range(len(dataset))):
    #         data_list = dataset[i]
    #         optimizer.zero_grad()
    #         features_image = model(data_list, torch.cat((query_tokens, query), dim=1))
    #         train_feature = features_image[data_list["mask"]]
    #         train_labels = data_list["label"][data_list["mask"]]
    #         metric_loss = loss_fn(train_feature.view(train_feature.shape[0], -1), train_labels)
    #         central_loss = center_loss(train_feature, train_labels)
    #         loss = metric_loss + central_loss * 0.5
    #         loss.backward()
    #         optimizer.step()
    #         scheduler.step()
            
    #         test_loss = center_loss(features_image, data_list["label"], torch.logical_not(data_list["mask"]))
    #         writer.add_scalar("central_loss", torch.sum(central_loss), iter)
    #         writer.add_scalar("metric_loss", torch.sum(metric_loss), iter)
    #         writer.add_scalar("test_loss", torch.sum(test_loss), iter)
    #         losses.append(torch.sum(test_loss).detach().cpu())
    #         iter += 1
    #     dataset.reset()
    #     losses = torch.tensor(losses).mean()
    #     torch.save(query, "./saved/{}_textquery_{}_{:.4f}.pt".format(os.getpid(), loop, losses))