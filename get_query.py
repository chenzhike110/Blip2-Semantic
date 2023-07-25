import torch

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
    