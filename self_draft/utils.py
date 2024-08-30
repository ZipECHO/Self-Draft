import math
import os
import random
import time
import warnings
# from types import Namespace
from argparse import Namespace
from statistics import stdev
from typing import Optional

import numpy as np
import psutil
import torch
from datasets import load_dataset
from fastchat.llm_judge.common import load_questions
from fastchat.model.model_adapter import Llama2Adapter, raise_warning_for_incompatible_cpu_offloading_configuration
from fastchat.utils import get_gpu_memory
from fastchat.utils import str_to_torch_dtype
from loguru import logger
from transformers import GenerationMixin
from transformers.models.llama import modeling_llama

from .decoding import FUNC_MAP, greedy_search, greedy_search_proxy,sample_proxy,sample
from .distribute_utils import get_device
from .inference_profile import *
from .models.Llama import llama


def ini_model_paras(model, args):
    model.draft_config = Namespace()
    model.cache = {}
    model.cache_other = {}

    attributes = [
        'self_draft', 'model_id', 'use_corpus', 'max_pad_len', 'limit_corpus_count', 'search_with_dif_key_len',
        'use_context', 'branch_num', 'branch_len', 'init_method', 'sample_number', 'flush_interval', 'keep_num',
        'dist_workers', 'use_flash', 'max_key_len', 'context_gram_n']
    for attr in attributes:
        setattr(model.draft_config, attr, getattr(args, attr))
    context_gram_n = min(args.context_gram_n, args.branch_len - 1)
    model.draft_config.context_gram_n = context_gram_n
    return model


def model_info(model):
    info = "\n"
    for key, val in vars(model.draft_config).items():
        info += f'{key}:\t\t{val}\n'

    return info


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


# prepare prompts
def set_user_query(query, messages=None):
    if messages is None:
        messages = [
            {"role": "user", "content": query},
        ]
    else:
        messages.append({"role": "user", "content": query})

    return messages


def update_agent_response(query, message):
    assert message is not None
    message.append({'role': 'assistant', 'content': query})
    return message


def extract_question(question, bench_name, j=0):
    if bench_name.lower() == 'humaneval':
        return question
    if bench_name.lower() == 'mt-bench':
        return question['turns'][j]
    if bench_name.lower() == 'mbpp':
        return question
    if bench_name.lower() == 'gsm':
        return question['question']
    if 'mt' in bench_name:
        return question['turns'][j]



def polish_dialogue(question, bench_name, turn):
    if bench_name == "HumanEval":
        assert isinstance(question, str)
        q = question
        qs = "Continue complete the following python functions: \n" + q

    elif bench_name == 'mbpp':
        qs = question
    elif bench_name == 'ifeval':
        qs = question['prompt']
    elif bench_name == 'gsm':
        qs = question['prompt']
    else:
        if isinstance(question, dict) and 'turns' in question:  # multi-turn e.g. mt-bench
            qs = question["turns"][turn]
        elif isinstance(question, dict) and 'context' in question:  # single turn with context e.g. dolly-15k
            # post_str = "If there is a context, please reply according to the context; " \
            #            "otherwise, please reply directly according to the instruction"
            if not question['context']:
                qs = question['instruction']
            else:
                context = question['context']
                qs = 'Context: ' + context + "\nInstruction: " + question['instruction'] + \
                     '\nPlease response to the Instruction according to the Context.'

        elif isinstance(question, str):
            qs = question
        else:
            raise RuntimeError(f'Unsupported benchmark: {bench_name}')

    return qs


def sample_and_pad(l, sample_list, c):
    _ = random.sample(sample_list, c)
    return [l[0] + _]


def pad_random_words(sentence, pool, c):
    s = random.sample(pool, c)
    return sentence + ' ' + ' '.join(s)


# load eval datasets
def load_human_eval(file_path, begin, end):
    all_data = load_dataset(file_path)
    return [all_data['test'][i]['prompt'] for i in range(len(all_data['test']['prompt'][begin:end]))]


def load_quac(file_path, begin, end):
    pass


def load_mbpp_eval(file_path, begin, end):
    all_data = load_dataset(file_path,'sanitized')
    return all_data['test']['prompt']


def load_text(question_file):
    questions = []
    with open(question_file, 'r') as file:
        for line in file:
            if line:
                questions.append(line.strip())
    return questions


def load_prompts(file_path, begin=None, end=None):
    if 'mt' in file_path:
        return load_questions(file_path, begin, end)
    elif 'HumanEval'.lower() in file_path.lower():
        return load_human_eval(file_path, begin, end)
    elif 'gsm' in file_path.lower():
        return load_questions(file_path, begin, end)
    elif "mbpp" in file_path.lower():
        return load_mbpp_eval(file_path, begin, end)
    else:
        raise RuntimeError(f'Unsupported benchmark: {file_path}')


# get model arch and inference id for log
def get_model_arch(args):
    if args.model_path is not None:
        if args.model_path[-1] == '/':
            p = args.model_path[:-1]
        else:
            p = args.model_path
    return os.path.basename(p)


def get_model_id(args, devices):
    return f'{args.model_arch}-{args.data_name}-context_gram_n-{args.context_gram_n}-branch_len-{args.branch_len}-branch_num-{args.branch_num}' \
           f'-pp-{"_".join(devices)}-draft_branch_ini-{args.init_method}-max-pad-len-{args.max_pad_len}-use_corpus-{args.use_corpus}-self_draft-{args.self_draft}' \
           f'-sample_number-{args.sample_number}-' \
           f'max_new_tokens-{args.max_new_tokens}'


# augment methods
def augment_llama():
    modeling_llama.LlamaForCausalLM.SDforward = llama.SDforward
    modeling_llama.LlamaModel.LlamaModelSDforward = llama.LlamaModelSDforward
    modeling_llama.LlamaModel.prepare_SD_decoder_attention_mask = llama.prepare_SD_decoder_attention_mask


def augment_generate():
    FUNC_MAP["greedy_search"] = greedy_search
    FUNC_MAP['sample'] = sample
    GenerationMixin.greedy_search = greedy_search_proxy
    GenerationMixin.sample = sample_proxy
    return


def augment_all():
    augment_llama()
    augment_generate()


# load model
def load_model(
        model_path: str,
        device: str = "cuda",
        device_map: str = "",
        num_gpus: int = 1,
        max_gpu_memory: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        load_8bit: bool = False,
        cpu_offloading: bool = False,
        revision: str = "main",
        use_flash: bool = False
):
    """Load a model from Hugging Face."""

    adapter = Llama2Adapter()

    # Handle device mapping
    cpu_offloading = raise_warning_for_incompatible_cpu_offloading_configuration(
        device, load_8bit, cpu_offloading
    )

    if device.startswith("cuda"):
        kwargs = {"torch_dtype": torch.float16}
        if num_gpus != 1:
            kwargs["device_map"] = "auto"
            if max_gpu_memory is None:
                kwargs[
                    "device_map"
                ] = "sequential"  # This is important for not the same VRAM sizes
                available_gpu_memory = get_gpu_memory(num_gpus)
                kwargs["max_memory"] = {
                    i: str(int(available_gpu_memory[i] * 0.85)) + "GiB"
                    for i in range(num_gpus)
                }
            else:
                kwargs["max_memory"] = {i: max_gpu_memory for i in range(num_gpus)}

    if cpu_offloading:
        # raises an error on incompatible platforms
        from transformers import BitsAndBytesConfig

        if "max_memory" in kwargs:
            kwargs["max_memory"]["cpu"] = (
                    str(math.floor(psutil.virtual_memory().available / 2 ** 20)) + "Mib"
            )
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_8bit_fp32_cpu_offload=cpu_offloading
        )
        kwargs["load_in_8bit"] = load_8bit
    elif load_8bit:
        if num_gpus != 1:
            warnings.warn(
                "8-bit quantization is not supported for multi-gpu inference."
            )
        else:
            model, tokenizer = adapter.load_compress_model(
                model_path=model_path,
                device=device,
                torch_dtype=kwargs["torch_dtype"],
                revision=revision,
            )

            return model, tokenizer
    kwargs["revision"] = revision

    if dtype is not None:  # Overwrite dtype if it is provided in the arguments.
        kwargs["torch_dtype"] = dtype
    if use_flash:
        kwargs["use_flash_attention_2"] = use_flash
    if len(device_map) > 0:
        kwargs["device_map"] = device_map
    if 'qwen' in model_path.lower():
        kwargs['trust_remote_code'] = True
    # Load model
    model, tokenizer = adapter.load_model(model_path, kwargs)

    if len(device_map) > 0:
        return model, tokenizer

    if (device.startswith("cuda") and num_gpus == 1 and not cpu_offloading) or device in (
            "mps",
            "xpu",
            "npu",
    ):
        model.to(device)

    if device == "xpu":
        model = torch.xpu.optimize(model, dtype=kwargs["torch_dtype"], inplace=True)

    return model, tokenizer


def ini_model(args):
    dtype = str_to_torch_dtype(args.dtype)

    if args.use_pp:
        model_, tokenizer_ = load_model(
            args.model_path,
            use_flash=args.use_flash,
            device=f"cuda",
            device_map="balanced",
            num_gpus=args.num_gpus_per_model,
            max_gpu_memory=args.max_gpu_memory,
            dtype=dtype,
            load_8bit=False,
            cpu_offloading=args.cpu_offloading,
        )

    elif args.use_tp_ds:
        import deepspeed
        torch.cuda.set_device(int(os.getenv('LOCAL_RANK', '0')))
        model_, tokenizer_ = load_model(
            args.model_path,
            use_flash=args.use_flash,
            device_map="cpu",
            num_gpus=args.num_gpus_per_model,
            max_gpu_memory=args.max_gpu_memory,
            dtype=dtype,
            load_8bit=False,
            cpu_offloading=args.cpu_offloading,
        )
        model_ = deepspeed.init_inference(
            model_,
            mp_size=int(os.getenv("WORLD_SIZE", "1")),
            dtype=torch.half
        )
    else:
        model_, tokenizer_ = load_model(
            args.model_path,
            use_flash=args.use_flash,
            device=f"cuda:{get_device(args)}",
            num_gpus=args.num_gpus_per_model,
            max_gpu_memory=args.max_gpu_memory,
            dtype=dtype,
            load_8bit=False,
            cpu_offloading=args.cpu_offloading,
        )
        logger.info('model load finished!')

        model_.tokenizer = tokenizer_

    return model_, tokenizer_
