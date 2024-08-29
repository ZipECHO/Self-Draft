import argparse
import os

from loguru import logger

import self_draft
from self_draft.inference_modes import *

devices = os.environ.get("CUDA_VISIBLE_DEVICES").split(",")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # model and data
    parser.add_argument("--model-path", type=str,
                        default='meta-llama/Llama-2-7b-chat-hf',
                        help="The path to the weights. This can be a local folder or a Hugging Face repo ID.")
    parser.add_argument('--question-file', type=str,
                        default="data/gsm/test.jsonl", help="The path to the question file.")
    parser.add_argument('--corpus-cache-path', type=str,
                        default="data/OANC/clean-5-OANC-tmp-cache.pickle", help="The path to the corpus cache file.")

    # self-draft parameters
    parser.add_argument('--self-draft', type=int, default=1)
    parser.add_argument("--branch-len", type=int, default=6, )
    parser.add_argument("--branch-num", type=int, default=6, )
    parser.add_argument('--sample-number', type=int, default=100)
    parser.add_argument('--context-gram-n', type=int, default=4)

    parser.add_argument("--cpu-offloading", action="store_true")
    parser.add_argument("--num-choices", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=1024,
                        help="The maximum number of new generated tokens.", )
    parser.add_argument("--num-gpus-per-model", type=int, default=1,
                        help="The number of GPUs per model.", )
    parser.add_argument("--num-gpus-total", type=int, default=1,
                        help="The total number of GPUs.")
    parser.add_argument("--max-gpu-memory", type=str,
                        help="Maximum GPU memory used for model weights per GPU.", )
    parser.add_argument("--dtype", type=str, choices=["float32", "float64", "float16", "bfloat16"],
                        help="Override the default dtype. If not set, it will use float16 on GPU and float32 on CPU.",
                        default=None, )
    parser.add_argument("--local-rank", type=int, default=0, )
    parser.add_argument("--use-pp", type=int, default=0, )
    parser.add_argument("--use-tp-ds", type=int, default=0, )
    parser.add_argument("--use-flash", type=int, default=0, )
    parser.add_argument("--dist-workers", default=1)
    # logs
    parser.add_argument('--save-log', type=int, default=1)
    parser.add_argument('--log-path', type=str, default='logs/',help="The path to the log file.")
    parser.add_argument("--log-level", default='SUCCESS')
    parser.add_argument('--search-with-dif-key-len', type=int, default=1)
    parser.add_argument('--init-method', type=str, default='RANDOM_WITH_AUX')
    parser.add_argument('--max-key-len', type=int, default=4)
    parser.add_argument('--max-pad-len', type=int, default=2)
    parser.add_argument('--keep-num', type=int,default=2)
    parser.add_argument('--use-context', type=int, default=1)
    parser.add_argument("--use-corpus", type=int, default=1)
    parser.add_argument('--load-corpus', type=int, default=1)
    parser.add_argument('--flush-interval', type=int, default=2)
    parser.add_argument('--limit-corpus-count', default=1)
    parser.add_argument('--run-mode', type=str, default='base')
    parser.add_argument('--ini-branch-with-corpus', type=int, default=0)
    parser.add_argument('--temperature',type=float, default=0.0)
    args = parser.parse_args()

    args.model_arch = get_model_arch(args)
    args.bench_name = os.path.dirname(args.question_file).split('/')[-1]
    args.data_name = os.path.dirname(args.question_file).split('/')[-1]
    args.model_id = get_model_id(args, devices)

    model, tokenizer = ini_model(args)
    self_draft.augment_all()
    if args.run_mode == 'greedy_search':
        greedy_decoding(args, model)
    else:
        if args.run_mode == 'base':
            base_self_draft(args, model)
        else:
            pass