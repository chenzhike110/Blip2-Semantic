import os
os.environ["CUDA_VISIBLE_DEVICES"]="2"
import torch
import numpy as np
from tqdm import tqdm
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
from lavis.models import load_model_and_preprocess

# setup device to use
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
# loads BLIP-2 pre-trained model
model, vis_processors, text_processors = load_model_and_preprocess(name="blip2_feature_extractor", model_type="pretrain_vitL", is_eval=True, device=device)

writer = SummaryWriter("./runs")
crit = torch.nn.MSELoss(reduction='sum')
datapath = "/data2/czk/MVNet/dataset/processed/"
motions = []
iter = 0

query_tokens = model.query_tokens.clone()

text_length = 50
text_id = torch.randint(low=0, high=model.Qformer.bert.embeddings.word_embeddings.num_embeddings, size=(1, text_length))
query = model.Qformer.bert.embeddings.word_embeddings(text_id.long().to(device))
query = Variable(query.clone(), requires_grad=True)
optimizer = torch.optim.Adam([query], lr=1e-3)

for motion in os.listdir(datapath):
    if "_prompt" in motion:
        continue
    motions.append(motion)
for loop in range(100):
    losses = []
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
            features_image = model({"image": images}, torch.cat((query_tokens, query), dim=1))
            center = features_image.mean(dim=0).unsqueeze(0).repeat(features_image.shape[0], 1, 1)
            loss = crit(features_image, center)
            loss.backward(torch.ones_like(loss), retain_graph=True)
            writer.add_scalar("loss", torch.sum(loss), iter)
            iter += 1
            losses.append(torch.sum(loss).detach().cpu())
            optimizer.step()
    losses = torch.tensor(losses).mean()
    writer.add_scalar("looploss", losses, loop)
    torch.save(query, "./saved/textquery_{}_{}.pt".format(loop, losses))