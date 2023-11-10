import argparse
from io import BytesIO

import requests
import torch
from PIL import Image
from transformers import TextStreamer

from llava.constants import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    IMAGE_TOKEN_INDEX,
)
from llava.conversation import SeparatorStyle, conv_templates
from llava.mm_utils import (
    KeywordsStoppingCriteria,
    get_model_name_from_path,
    process_images,
    tokenizer_image_token,
)
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init


class LlavaDirect:
    def __init__(
        self,
        model_path="liuhaotian/llava-v1.5-13b",
        model_base=None,
        device="cuda",
        conv_mode=None,
        temperature=0.2,
        max_new_tokens=512,
        load_8bit=True,
        load_4bit=False,
        debug=False,
    ):
        # Model
        disable_torch_init()

        self.args = self.parse_args()
        self.model = None
        self.tokenizer = None
        self.image_processor = None
        self.context_len = None
        self.conv = None
        self.roles = None
        self.image = None
        self.image_tensor = None
        self.conv_mode = None

        self.load_model(model_path, model_base, load_8bit, load_4bit, device)

        if "llama-2" in self.model_name.lower():
            self.conv_mode = "llava_llama_2"
        elif "v1" in self.model_name.lower():
            self.conv_mode = "llava_v1"
        elif "mpt" in self.model_name.lower():
            self.conv_mode = "mpt"
        else:
            self.conv_mode = "llava_v0"

        self.conv = conv_templates[self.conv_mode].copy()
        if "mpt" in self.model_name.lower():
            self.roles = ("user", "assistant")
        else:
            self.roles = self.conv.roles

    def parse_args(self):
        parser = argparse.ArgumentParser()
        # add arguments here
        return parser.parse_args()

    def load_model(self, model_path, model_base, load_8bit, load_4bit, device):
        model_name = get_model_name_from_path(model_path)
        (
            self.tokenizer,
            self.model,
            self.image_processor,
            self.context_len,
        ) = load_pretrained_model(
            model_path,
            model_base,
            model_name,
            load_8bit,
            load_4bit,
            device=device,
        )

    def run(self):
        # main execution code here
        return

    def load_image(image_file):
        if image_file.startswith("http://") or image_file.startswith("https://"):
            response = requests.get(image_file)
            image = Image.open(BytesIO(response.content)).convert("RGB")
        else:
            image = Image.open(image_file).convert("RGB")
        return image
