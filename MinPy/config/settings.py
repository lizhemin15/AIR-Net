import torch

if torch.cuda.is_available():
    cuda_if = True
else:
    cuda_if =False
