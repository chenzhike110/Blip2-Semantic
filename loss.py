import torch

def center_loss(features, labels, mask=None):
    if mask is not None:
        mask = torch.ones(features.shape[0]).long()
    label_unque = torch.unique(labels)
    center = torch.zeros_like(features).to(features.device)
    for label in label_unque:
        center[labels == label] = features[labels == label].mean(dim=0)
    loss = torch.nn.functional.mse_loss(features, center, reduce=False)
    loss = loss[mask]
    loss = torch.sum(loss) / torch.sum(mask)
    return loss
