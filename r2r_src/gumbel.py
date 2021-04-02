import torch
import torch.nn.functional as F


def _sample_logistic(shape, eps=1e-10, out=None):
    if out is None:
        U = torch.rand(shape)
    else:
        U = torch.jit._unwrap_optional(out).resize_(shape).uniform_()
    return torch.log(eps + U) - torch.log(1 - U + eps)


def _gumbel_sigmoid_sample(logits, tau=1.0, eps=1e-10):
    gumbel_noise = _sample_logistic(logits.size(), eps=eps, out=torch.empty_like(logits))
    return torch.sigmoid((logits + gumbel_noise) / tau)


def gumbel_sigmoid(logits, tau=1., hard=False, eps=1e-10, test=False):
    if test == True:
        y_soft = torch.sigmoid(logits)
        y_hard = y_soft > 0.5
        return y_hard.to(logits.dtype)
    y_soft = _gumbel_sigmoid_sample(logits, tau=tau, eps=eps)
    if hard:
        y_hard = y_soft > 0.5
        y_hard = y_hard.to(logits.dtype)
        y = y_hard - y_soft.detach() + y_soft
    else:
        y = y_soft
    return y

