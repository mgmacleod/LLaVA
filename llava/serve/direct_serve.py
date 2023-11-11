import argparse
import time
from typing import List, Union

import numpy as np
import requests
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import Response

from llava.serve.direct_ret2i import StableDiffusionDirect
from llava.serve.llava_direct import LlavaDirect


def main(args):
    # ld = LlavaDirect(args.model_path)
    # sd = StableDiffusionDirect(args.host, args.port, args.iterations, args.steps)

    iterations = args.iterations
    sd_prompt = args.sd_prompt
    ll_prompt = args.ll_prompt

    sd_prompt_init = sd_prompt
    ll_prompt_init = ll_prompt

    name = args.name
    image_dir = args.image_dir

    # run_experiment(ld, sd, iterations, sd_prompt, ll_prompt, name, image_dir)

    # next_prompt = ld.process_image(filename, ll_prompt)
    # ll_prompt = next_prompt
    # time.sleep(3)

    return


def run_experiment(ld, sd, iterations, sd_prompt, ll_prompt, name, image_dir):
    sd.create_directory(image_dir, name)

    image_dict = {}

    for i in range(iterations):
        filename = sd.generate_image(sd_prompt, i)
        image_dict[i] = (sd_prompt, filename)
        next_prompt = ld.process_image(filename, ll_prompt)
        sd_prompt = next_prompt
        time.sleep(3)

    for i in range(iterations):
        sd_prompt, filename = image_dict[i]
        print(f"Image {i}: prompt = {sd_prompt} -> {filename}")


app = FastAPI()


@app.post("/register_worker")
async def register_worker(request: Request):
    data = await request.json()
    print(data["image_file"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path", type=str, default=None  # default="liuhaotian/llava-v1.5-7b"
    )

    # parser.add_argument("--sd-prompt", type=str, required=True)
    # parser.add_argument("--ll-prompt", type=str, required=True)
    parser.add_argument("--sd-host", type=str, default="localhost")
    parser.add_argument("--sd-port", type=int, default=8860)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=21001)
    # parser.add_argument("--iterations", type=int, required=True)
    # parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--image-dir", type=str, default="output")
    parser.add_argument("--steps", type=int, default=5)

    args = parser.parse_args()

    ld = LlavaDirect(args.model_path)
    sd = StableDiffusionDirect(args.sd_host, args.sd_port, args.steps)

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    main(args)
