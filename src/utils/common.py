import os
import torch
from huggingface_hub import login
from core.logger import general_logger


def login_to_hf():
    general_logger.info("Logging in to Huggingface")
    hf_token = os.getenv("HF_TOKEN")
    login(token=hf_token)
    general_logger.info("Logged in to Huggingface")


def get_device():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return device
