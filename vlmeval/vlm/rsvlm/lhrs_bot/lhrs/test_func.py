import json
import os
from dataclasses import dataclass
from glob import glob
from typing import Any, Callable, Dict, List, Sequence, Tuple, Union

import numpy as np
import torch
import torchvision.transforms as T
import transformers
from PIL import Image
from transformers import CLIPImageProcessor


class Compose(T.Compose):
    """Custom Compose which processes a list of inputs"""

    def __init__(self, transforms: Sequence[Callable]):
        self.transforms = transforms

    def __call__(self, x: Union[Any, Sequence]):
        if isinstance(x, Sequence):
            for t in self.transforms:
                x = [t(i) for i in x]
        else:
            for t in self.transforms:
                x = t(x)
        return x


pp = Compose([])

print(pp("hello world"))
