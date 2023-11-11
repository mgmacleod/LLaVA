import argparse
import json
import logging
import threading
import time
from typing import List, Union

import numpy as np
import requests
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import Response

from llava.serve.direct_ret2i import StableDiffusionDirect
from llava.serve.llava_direct import LlavaDirect

# def run_experiment(
#     ld: LlavaDirect, sd: StableDiffusionDirect, iterations, sd_prompt, ll_prompt, name
# ):
#     working_dir = sd.create_directory(image_dir, name)

#     image_dict = {}

#     for i in range(iterations):
#         filename = sd.generate_image(sd_prompt, i)
#         image_dict[i] = (sd_prompt, filename)
#         next_prompt = ld.process_image(filename, ll_prompt)
#         sd_prompt = next_prompt
#         time.sleep(3)

#         # Write the current state of image_dict to a JSON file
#         with open(f"{working_dir}/log_{i}.json", "w") as f:
#             json.dump(image_dict[i], f)

#     data = []
#     for i in range(iterations):
#         sd_prompt, filename = image_dict[i]
#         data.append({"Iteration": i, "prompt": sd_prompt, "filename": filename})

#     with open(f"{working_dir}/log.json", "w") as f:
#         json.dump(data, f)

#     return


def run_experiment(
    ld: LlavaDirect,
    sd: StableDiffusionDirect,
    iterations,
    sd_prompt,
    ll_prompt,
    name,
    steps,
    negative_prompt,
):
    label = f"{name}x{iterations}"
    working_dir = sd.create_directory(image_dir, label)

    # Configure logging
    logger = logging.getLogger("experiment_logger")
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(f"{working_dir}/log.txt")
    formatter = logging.Formatter(
        "%(message)s"
    )  # Log only the message, which will be JSON
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    msg = f"Running experiment {name} with {iterations} iterations, {steps} steps of stable diffusion stably diffusing, and negative prompt {negative_prompt}, starting with prompt {sd_prompt}"
    logger.info(msg)

    image_dict = {}

    for i in range(iterations):
        filename = sd.generate_image(
            sd_prompt, i, steps=steps, negative_prompt=negative_prompt
        )
        image_dict[i] = (sd_prompt, filename)
        next_prompt = ld.process_image(filename, ll_prompt)
        sd_prompt = next_prompt
        time.sleep(3)

        # Log the current state of image_dict
        logger.info(json.dumps(image_dict[i]))

    return


app = FastAPI()


@app.post("/run_experiment")
async def handle_run_experiment(request: Request):
    data = await request.json()
    threading.Thread(
        target=run_experiment,
        args=(
            ld,
            sd,
            data["iterations"],
            data["sd_prompt"],
            data["ll_prompt"],
            data["name"],
            data["steps"],
            data["negative_prompt"],
        ),
    ).start()
    return Response(status_code=200, content="Running experiment...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--sd-host", type=str, default="localhost")
    parser.add_argument("--sd-port", type=int, default=8860)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=21001)
    parser.add_argument("--image-dir", type=str, default="output")

    args = parser.parse_args()

    print("Loading models...")
    print(f"Loading LLaVA model from {args.model_path}...")
    print(f"Loading Stable Diffusion model from {args.sd_host}:{args.sd_port}...")

    print("args.host", args.host)
    print("args.port", args.port)
    ld = LlavaDirect(args.model_path)
    sd = StableDiffusionDirect(args.sd_host, args.sd_port)

    image_dir = args.image_dir
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
