import torch
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "TripoSR"))
from tsr.system import TSR
print("CUDA Available:", torch.cuda.is_available())
model = TSR.from_pretrained(
    os.path.join(os.path.dirname(__file__), "TripoSR_weights"),
    config_name="config.yaml",
    weight_name="model.ckpt",
)
model.to("cuda")
print("Model loaded to CUDA!")
