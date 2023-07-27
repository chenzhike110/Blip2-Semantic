import os
os.environ["CUDA_VISIBLE_DEVICES"]="2"
import torch
from lavis.models import load_model_and_preprocess

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
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    text_query = torch.load("./saved/16042_textquery_17_0.31481030583381653.pt").to(device)
    model, vis_processors, text_processors = load_model_and_preprocess(name="blip2_feature_extractor", model_type="pretrain_vitL", is_eval=True, device=device)
    layer = model.Qformer.bert.embeddings.word_embeddings
    indices = torch.zeros((text_query.size(1), 1))
    for i in range(indices.shape[0]):
        query = text_query[0, i].unsqueeze(0).unsqueeze(0)
        indices[i] = emb2indices(query, layer)
    captions = model.tokenizer.batch_decode(indices, skip_special_tokens=True)
    print(captions)