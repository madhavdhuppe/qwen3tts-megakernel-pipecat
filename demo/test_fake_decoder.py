import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from megakernel_adapter.fake_decoder import (
    FakeMegakernelDecoder
)


decoder = FakeMegakernelDecoder(
    "Qwen/Qwen2-0.5B"
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_ids = torch.tensor([[1]], device=device)

token = decoder.step(input_ids)

print(token)
