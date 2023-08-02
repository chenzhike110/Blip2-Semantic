import torch

def center_loss(features, labels, mask=None):
    if mask is None:
        mask = torch.ones(features.shape[0]).long()
    label_unque = torch.unique(labels)
    center = torch.zeros_like(features).to(features.device)
    for label in label_unque:
        center[labels == label] = features[labels == label].mean(dim=0)
    center = torch.nn.functional.normalize(center, dim=-1)
    loss = torch.nn.functional.mse_loss(features, center, reduce=False)
    loss = loss[mask]
    loss = torch.sum(loss) / torch.sum(mask)
    return loss

def square_loss(features, labels, mask=None):
    if mask is None:
        mask = torch.ones(features.shape[0]).long()
    label_unque = torch.unique(labels)
    center = torch.zeros_like(features).to(features.device)
    for label in label_unque:
        center[labels == label] = features[labels == label].mean(dim=0)
    center = torch.nn.functional.normalize(center, dim=-1)
    simularity = torch.diagonal(features @ center.t(), 0)
    loss = torch.nn.functional.mse_loss(simularity, torch.ones_like(simularity), reduce=False)
    loss = loss[mask]
    loss = torch.sum(loss) / torch.sum(mask)
    return loss
