import argparse
import json
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


def run_experiment(
    ld: LlavaDirect, sd: StableDiffusionDirect, iterations, sd_prompt, ll_prompt, name
):
    working_dir = sd.create_directory(image_dir, name)

    image_dict = {}

    for i in range(iterations):
        filename = sd.generate_image(sd_prompt, i)
        image_dict[i] = (sd_prompt, filename)
        next_prompt = ld.process_image(filename, ll_prompt)
        sd_prompt = next_prompt
        time.sleep(3)

    # for i in range(iterations):
    #     sd_prompt, filename = image_dict[i]
    #     print(f"Iteration {i}: prompt = {sd_prompt} -> {filename}")

    # print each (sd_prompt, filename) in image_dict to a log file inside working_dir:
    # with open(f"{working_dir}/log.txt", "w") as f:
    #     for i in range(iterations):
    #         sd_prompt, filename = image_dict[i]
    #         f.write(f"Iteration {i}: prompt = {sd_prompt} -> {filename}\n")

    data = []
    for i in range(iterations):
        sd_prompt, filename = image_dict[i]
        data.append({"Iteration": i, "prompt": sd_prompt, "filename": filename})

    with open(f"{working_dir}/log.json", "w") as f:
        json.dump(data, f)

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
            # data["image_dir"],
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
    parser.add_argument("--steps", type=int, default=50)

    args = parser.parse_args()

    print("Loading models...")
    print(f"Loading LLaVA model from {args.model_path}...")
    print(f"Loading Stable Diffusion model from {args.sd_host}:{args.sd_port}...")

    print("args.host", args.host)
    print("args.port", args.port)
    ld = LlavaDirect(args.model_path)
    sd = StableDiffusionDirect(args.sd_host, args.sd_port, args.steps)

    image_dir = args.image_dir
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
