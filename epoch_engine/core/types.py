"Module for custom types."

# Author: Sergey Polivin <s.polivin@gmail.com>
# License: MIT License

from typing import TypeAlias

import torch

TorchDataloader: TypeAlias = torch.utils.data.DataLoader
TorchModel: TypeAlias = torch.nn.Module
TorchTensor: TypeAlias = torch.Tensor
