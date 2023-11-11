import argparse
import base64
import datetime
import io
import json
import os

import requests
from PIL import Image


class StableDiffusionDirect:
    def __init__(self, host, port):
        self.image_dir = None
        self.url = f"http://{host}:{port}"

    def generate_image(
        self,
        prompt,
        iteration,
        steps=5,
        width=768,
        height=768,
        negative_prompt=None,
        cfg_scale=7,
    ):
        if self.image_dir is None:
            print("Please create a directory first.")
            return

        if iteration is None:
            print("Please specify an iteration.")
            return

        payload = {
            "prompt": prompt,
            "steps": steps,
            "width": width,
            "height": height,
            "cfg_scale": cfg_scale,
            "negative_prompt": negative_prompt,
        }

        response = requests.post(url=f"{self.url}/sdapi/v1/txt2img", json=payload)

        r = response.json()

        image = Image.open(io.BytesIO(base64.b64decode(r["images"][0])))
        filename = f"{self.image_dir}/output{iteration}.png"

        image.save(filename)
        return filename

    def create_directory(self, path: str, label: str):
        print("Creating directory...")
        now = datetime.datetime.now()

        # Format date and time
        date_time = now.strftime("%Y-%m-%d%H_%M_%S")

        # Append suffix
        dir_name = f"{path}/{date_time}_{label}"
        self.image_dir = dir_name
        os.makedirs(dir_name, exist_ok=True)
        return dir_name


def main(args):
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path", type=str, default=None
    )  # default="liuhaotian/llava-v1.5-13b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8860)
    args = parser.parse_args()

    main(args)
