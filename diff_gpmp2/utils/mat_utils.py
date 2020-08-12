#!/usr/bin/env python
import torch

def isotropic_matrix(sig, dim, device=torch.device('cpu')):
  mat = sig * torch.eye(dim, device=device)
  return mat