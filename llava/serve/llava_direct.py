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


class CollectingStreamer(TextStreamer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.collected_outputs = []

    def decode(self, token_ids):
        output = super().decode(token_ids)
        self.collected_outputs.append(output)
        return output

    def get_collected_outputs(self):
        return self.collected_outputs

    def get_concatenated_outputs(self):
        return "\n".join(self.collected_outputs)


class LlavaDirect:
    def __init__(
        self,
        model_path=None,
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

        # self.args = self.parse_args()
        self.model = None
        self.tokenizer = None
        self.image_processor = None
        self.context_len = None
        self.conv = None
        self.roles = None
        self.image = None
        self.image_tensor = None
        self.conv_mode = None
        self.debug = debug
        self.model_name = None
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.streamer = None

        if model_path is not None:
            self.load_model(model_path, model_base, load_8bit, load_4bit, device)

    def load_model(
        self,
        model_path,
        model_base=None,
        load_8bit=True,
        load_4bit=False,
        device="cuda",
    ):
        self.model_name = get_model_name_from_path(model_path)
        (
            self.tokenizer,
            self.model,
            self.image_processor,
            self.context_len,
        ) = load_pretrained_model(
            model_path,
            model_base,
            self.model_name,
            load_8bit,
            load_4bit,
            device=device,
        )

        self.reset_conversation()

    def reset_conversation(self):
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
            self.roles = self.conv.roles

    def test(self):
        image = self.load_image(
            "/home/mgm/development/ai/llm/LLaVA/llava/serve/examples/waterview.jpg"
        )
        output = self.process_image(
            image,
            'what do you see here? (Please do not start with "The image features..."; just describe what you see.)',
        )

        # process output and remove leading string "The image features " and remode trailing string "</s>"
        output = output.replace("The image features ", "").replace("</s>", "")
        return output

    # def load_image(self, image_file):
    #     if image_file.startswith("http://") or image_file.startswith("https://"):
    #         response = requests.get(image_file)
    #         image = Image.open(BytesIO(response.content)).convert("RGB")
    #     else:
    #         image = Image.open(image_file).convert("RGB")
    #     return image

    def process_image(self, image_file: str, prompt: str) -> str:
        if image_file.startswith("http://") or image_file.startswith("https://"):
            response = requests.get(image_file)
            image = Image.open(BytesIO(response.content)).convert("RGB")
        else:
            image = Image.open(image_file).convert("RGB")

        self.reset_conversation()

        # Similar operation in model_worker.py
        image_tensor = process_images([image], self.image_processor, self.model.config)
        if type(image_tensor) is list:
            image_tensor = [
                image.to(self.model.device, dtype=torch.float16)
                for image in image_tensor
            ]
        else:
            image_tensor = image_tensor.to(self.model.device, dtype=torch.float16)
        # return image_tensor

        if image is not None:
            # first message
            if self.model.config.mm_use_im_start_end:
                prompt = (
                    DEFAULT_IM_START_TOKEN
                    + DEFAULT_IMAGE_TOKEN
                    + DEFAULT_IM_END_TOKEN
                    + "\n"
                    + prompt
                )
            else:
                prompt = DEFAULT_IMAGE_TOKEN + "\n" + prompt
            self.conv.append_message(self.conv.roles[0], prompt)
            image = None

        self.conv.append_message(self.conv.roles[1], None)
        prompt = self.conv.get_prompt()

        input_ids = (
            tokenizer_image_token(
                prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            )
            .unsqueeze(0)
            .to(self.model.device)
        )
        stop_str = (
            self.conv.sep
            if self.conv.sep_style != SeparatorStyle.TWO
            else self.conv.sep2
        )
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(
            keywords, self.tokenizer, input_ids
        )
        streamer = TextStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True
        )

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor,
                do_sample=True if self.temperature > 0 else False,
                temperature=self.temperature,
                max_new_tokens=self.max_new_tokens,
                streamer=streamer,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
            )

        outputs = self.tokenizer.decode(output_ids[0, input_ids.shape[1] :]).strip()
        self.conv.messages[-1][-1] = outputs

        if self.debug:
            print("\n", {"prompt": prompt, "outputs": outputs}, "\n")

        # process output and remove leading string "The image features " and remode trailing string "</s>"
        outputs = outputs.replace("The image features ", "").replace("</s>", "")
        return outputs
