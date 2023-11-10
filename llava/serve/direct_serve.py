import argparse
import threading
import time
from typing import List, Union

import numpy as np
import requests
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import Response, StreamingResponse

from llava.serve.llava_direct import LlavaDirect


def main(args):
    return


app = FastAPI()


@app.post("/load_image")
def load_image(request: Request):
    data = request.json()
    image_file = data["image_file"]
    return image_file


@app.post("/generate")
def generate(request: Request):
    data = request.json()
    return "hello"


if __name__ == "__main__":
    # print(llava.run())

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="liuhaotian/llava-v1.5-13b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=21001)
    args = parser.parse_args()

    ld = LlavaDirect(args.model_path)

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
