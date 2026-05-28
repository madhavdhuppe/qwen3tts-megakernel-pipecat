import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from megakernel_adapter import Decoder, MODE


decoder = Decoder(
    "Qwen/Qwen2-0.5B"
)

print(type(decoder))
print(f"mode={MODE}")
