import torch
import torch.nn
from torchtyping import TensorType

class Solution:
    def reshape(self, to_reshape: TensorType[float]) -> TensorType[float]:
        M, N = to_reshape.shape
        reshaped = torch.reshape(to_reshape, (M * N // 2, 2))
        return torch.round(reshaped, decimals=4)
    
    def average(self, to_avg: TensorType[float]) -> TensorType[float]:
        torch.mean(to_avg, dim=0)
        return torch.round(torch.mean(to_avg, dim=0),decimals = 4)
        # Compute column-wise mean (average across rows)
        # Use torch.mean(tensor, dim=0)
        pass

    def concatenate(self, cat_one: TensorType[float], cat_two: TensorType[float]) -> TensorType[float]:
        return torch.round(torch.cat((cat_one, cat_two), dim=1),decimals=4)
        # Join two tensors side-by-side along dim=1
        # Use torch.cat((a, b), dim=1)
        pass

    def get_loss(self, prediction: TensorType[float], target: TensorType[float]) -> TensorType[float]:
        return torch.round(torch.nn.functional.mse_loss(prediction, target),decimals=4)
        # Compute Mean Squared Error between prediction and target
        # Use torch.nn.functional.mse_loss(prediction, target)
        pass
