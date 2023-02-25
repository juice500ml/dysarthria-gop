import torch
import torch.nn.functional as F


def _get_losses_lengths(logits, labels):
    probs = F.softmax(logits.transpose(1, 2), dim=1)
    losses = F.nll_loss(probs, labels, ignore_index=-100, reduction="none")
    lengths = (labels >= 0).sum(1)
    return losses, lengths


def samplewise_average_loss(logits, labels):
    losses, lengths = _get_losses_lengths(logits, labels)
    weights = (1/lengths)[:, None].expand(labels.shape) / len(labels)
    return (losses * weights).sum()


def phonewise_average_loss(logits, labels):
    losses, lengths = _get_losses_lengths(logits, labels)
    weights = 1 / lengths.sum()
    return (losses * weights).sum()


def ctc_like_loss(logits, labels):
    losses, lengths = _get_losses_lengths(logits, labels)
    weights = []
    for label, length in zip(labels, lengths):
        label = label[:length]
        _, indices, counts = torch.unique_consecutive(
            label, return_inverse=True, return_counts=True)
        weight = torch.index_select(1 / counts, 0, indices) / counts.shape[0]
        weight = torch.cat((weight, torch.zeros(labels.shape[1] - length, device=weight.device)))
        weights.append(weight)
    weights = torch.stack(weights) / len(labels)
    return (losses * weights).sum()


def get_loss(name):
    return {
        "samplewise_average": samplewise_average_loss,
        "phonewise_average": phonewise_average_loss,
        "ctc_like": ctc_like_loss,
    }[name]
