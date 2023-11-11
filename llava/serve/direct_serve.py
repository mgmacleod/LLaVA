import argparse
import time
from typing import List, Union

from llava.serve.direct_ret2i import StableDiffusionDirect
from llava.serve.llava_direct import LlavaDirect


def main(args):
    ld = LlavaDirect(args.model_path)
    sd = StableDiffusionDirect(args.host, args.port, args.steps)

    iterations = args.iterations
    sd_prompt = args.sd_prompt
    ll_prompt = args.ll_prompt

    sd_prompt_init = sd_prompt
    ll_prompt_init = ll_prompt

    name = args.name
    image_dir = args.image_dir

    sd.create_directory(image_dir, name)

    for i in range(iterations):
        filename = sd.generate_image(sd_prompt, i)
        # ld.load_image(f"{image_dir}/output{i}.png")
        next_prompt = ld.process_image(filename, ll_prompt)
        print(f"################################ NEXT_PROMPT=== {next_prompt}")
        sd_prompt = next_prompt
        time.sleep(3)

    return


if __name__ == "__main__":
    # print(llava.run())

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="liuhaotian/llava-v1.5-13b")

    parser.add_argument("--sd-prompt", type=str, required=True)
    parser.add_argument("--ll-prompt", type=str, required=True)
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8860)
    parser.add_argument("--iterations", type=int, required=True)
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--image-dir", type=str, default="output")
    parser.add_argument("--steps", type=int, default=5)

    # parser.add_argument("--model-base", type=str, default=None)
    # parser.add_argument("--device", type=str, default="cuda")
    # parser.add_argument("--conv-mode", type=str, default=None)
    # parser.add_argument("--temperature", type=float, default=0.2)
    # parser.add_argument("--max-new-tokens", type=int, default=512)
    # parser.add_argument("--load-8bit", action="store_true")
    # parser.add_argument("--load-4bit", action="store_true")
    # parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    main(args)

    # ld = LlavaDirect(args.model_path)

    # uvicorn.run(app, host=args.host, port=args.port, log_level="info")
