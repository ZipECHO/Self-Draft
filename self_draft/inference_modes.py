import sys

from fastchat.model import get_conversation_template

from .cache.CorpusCache import CorpusCache
from .utils import *

devices = os.environ.get("CUDA_VISIBLE_DEVICES").split(",")


def run(args, model, corpus_cache=None):
    # basic inference method
    args.model_id = get_model_id(args, devices)
    model = ini_model_paras(model, args)
    model.corpus_cache = corpus_cache

    logger.info('=============' * 4 + 'decoding parameters' + '=============' * 4)
    logger.info(f'{args}')
    logger.info('=============' * 4 + 'model parameters' + '=============' * 4)

    logger.info(model_info(model))
    setup_seed(10)

    questions = load_prompts(args.question_file)

    assert args.num_gpus_total % args.num_gpus_per_model == 0

    get_answers_func = get_model_answers

    chunk_size = len(questions) // (args.num_gpus_total // args.num_gpus_per_model)

    decode_kwargs = {}
    decode_kwargs['temperature'] = getattr(args, 'temperature', 0.7)
    decode_kwargs['top_k'] = getattr(args, 'top_k', 50)
    decode_kwargs['top_p'] = getattr(args, 'top_p', 0.9)
    decode_kwargs['max_new_tokens'] = getattr(args, 'max_new_tokens', 1024)
    decode_kwargs['pad_token_id'] = model.tokenizer.eos_token_id

    ans_handles = []
    for i in range(0, len(questions), chunk_size):
        ans_handles.append(
            get_answers_func(
                model,
                model.tokenizer,
                questions[i: i + chunk_size],
                args.num_choices,
                args.sample_number,
                args.bench_name,
                **decode_kwargs
            )
        )

    logger.info(args)


def set_log(args):
    eval_data = os.path.dirname(args.question_file).split('/')[-1]
    logger_path = None
    if args.save_log:
        logger.remove()
        logger.add(sys.stdout, level=args.log_level)
        now = time.strftime('%Y%m%d%H%M%S', time.localtime())
        logger_path = os.path.join(args.log_path, eval_data, args.model_id + '-' + now + '.log')
        logger_id = logger.add(logger_path, level=args.log_level.upper())
    else:
        logger.remove()
        logger_id = logger.add(sys.stdout, level=args.log_level.upper())
    return logger_path, logger_id


def base_self_draft(args, model):
    logger_path, logger_id = set_log(args)
    if args.load_corpus == 1:
        corpus_cache = load_corpus_cache(args)
    else:
        corpus_cache = None

    run(args, model, corpus_cache)
    if args.save_log:
        logger.success('Finished! All logs saved at ' + logger_path)
        logger.remove(logger_id)
        os.chmod(logger_path, 0o444)

def load_corpus_cache(args):
    logger.info(f'trying corpus cache: {args.corpus_cache_path}')
    logger.info("loading general corpus cache")
    base_corpus_cache = CorpusCache(args.branch_len - 1, args.max_key_len, tokenizer_path=args.model_path,
                                    cache_save_path=args.corpus_cache_path)
    logger.success('loading cache done!')

    return base_corpus_cache


def greedy_decoding(args, model):
    eval_data = os.path.dirname(args.question_file).split('/')[-1]
    args.model_arch = get_model_arch(args)
    logger_path = os.path.join('log/ar', eval_data, f'{args.model_arch}-{eval_data}-{args.max_new_tokens}.log')
    greedy_id = logger.add(logger_path, level=args.log_level)
    args.self_draft = 0
    run(args, model)
    args.self_draft = 1
    logger.remove(greedy_id)
    # os.chmod(logger_path, 0o444)


def grid_search_branch_len_num(args, model):
    base_corpus_cache = load_corpus_cache(args)
    now = time.strftime('%Y%m%d%H%M%S', time.localtime())
    branch_nums = [2, 3, 4, 5, 6, 8, 10, 12]
    branch_lens = [4, 5, 6, 7, 9, 11]
    # branch_lens = [6]
    branch_nums_str = '-'.join([str(i) for i in branch_nums])
    branch_lens_str = '-'.join([str(i) for i in branch_lens])
    logger_path = f'log-grid-0815/{args.bench_name}/grid-{args.model_arch}-branch_nums-{branch_nums_str}-branch_lens-{branch_lens_str}-{now}.log'
    logger.remove()
    logger.add(sys.stdout, level=args.log_level.upper())
    logger_id = logger.add(logger_path, level=args.log_level.upper())
    for branch_num in branch_nums:
        args.branch_num = branch_num
        for l in branch_lens:
            args.branch_len = l
            assert args.self_draft
            args.model_id = get_model_id(args, devices)
            run(args, model, base_corpus_cache)
            del model.context_cache

    if args.save_log:
        logger.remove(logger_id)
        os.chmod(logger_path, 0o444)


def grid_search_branch_len(args, model):
    base_corpus_cache = load_corpus_cache(args)
    now = time.strftime('%Y%m%d%H%M%S', time.localtime())
    branch_nums = [3, 4, 5, 6, 7, 9, 11, 14, 16, 20]
    branch_lens = [4, 6, 8, 10, 12, 14]

    branch_nums_str = '-'.join([str(i) for i in branch_nums])
    branch_lens_str = '-'.join([str(i) for i in branch_lens])

    logger_path = (f'log/{args.bench_name}/grid/grid-{args.model_arch}'
                   f'-branch_nums-{branch_nums_str}'
                   f'-branch_lens-{branch_lens_str}-'
                   f'-{now}.log')
    logger.remove()
    logger.add(sys.stdout, level=args.log_level.upper())
    logger_id = logger.add(logger_path, level=args.log_level.upper())
    # for context_gram_n in context_gram_ns:
    for branch_num in branch_nums:
        for l in branch_lens:
            # if l > context_gram_n:
            args.branch_num = branch_num
            # args.context_gram_n = context_gram_n
            args.branch_len = l
            assert args.self_draft
            args.model_id = get_model_id(args, devices)
            run(args, model, base_corpus_cache)
            del model.context_cache
            # else:
            #     continue

    if args.save_log:
        logger.remove(logger_id)
        os.chmod(logger_path, 0o444)


def get_model_answers(
        model,
        tokenizer,
        questions,
        num_choices,
        sample_number,
        bench_name,
        **kwargs
):
    ds_local_rank = int(os.getenv('LOCAL_RANK', '0'))
    model_id = 'llama-2-7b'
    overall_time = 0
    overall_tp = []
    overall_gen = 0
    overall_steps = 0
    count_gen = 0
    stats = {}
    profile = InferProfile()
    setup_seed(10)
    if sample_number == -1:
        sample_idx = list(range(0, len(questions)))
    else:
        sample_idx = random.sample(range(0, len(questions)), sample_number)
    if int(os.environ.get('MT_SPEC_SMP', 0)):
        sample_idx = [0, 1, 10, 11, 20, 21, 30, 31, 40, 41, 50, 51, 60, 61, 70, 71]
    count_conv = 0
    for i, question_idx in enumerate(sample_idx):
        question = questions[question_idx]
        logger.success(f'decoding questions:{i} / {sample_idx}')
        stats[question_idx] = {}
        for c in range(num_choices):
            torch.manual_seed(c)
            conv = get_conversation_template(model_id)
            turns = []
            prompts = []

            if isinstance(question, dict) and 'turns' in question:
                conv_range = len(question['turns'])
            else:
                conv_range = 1
            message = None

            for j in range(conv_range):
                count_conv += 1
                qs = extract_question(question, bench_name, j)
                # if isinstance(question, dict) and 'turns' in question:
                #     qs = question['turns'][j]
                # else:
                #     qs = question
                conv.append_message(conv.roles[0], qs)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()
                prompts.append(prompt)
                # qs = polish_dialogue(question, bench_name, j)
                # message = set_user_query(query=qs, messages=message)
                # prompt = tokenizer.apply_chat_template(message, tokenize=False)
                input_ids = tokenizer([prompt]).input_ids
                logger.debug('==========' * 5 + f'input prompts' + '==========' * 5)
                logger.debug(prompt)
                logger.debug('==========' * 5 + f'input prompts end' + '==========' * 5)

                if kwargs['temperature'] < 1e-4:
                    kwargs['do_sample'] = False
                else:
                    kwargs['do_sample'] = True

                if True:
                    start_time = time.time()

                    t0 = time.perf_counter()
                    (output_ids, s, profile_) = model.generate(
                        torch.as_tensor(input_ids).cuda(),
                        **kwargs,
                    )
                    profile.incremental_update('decoding_time', time.perf_counter() - t0)
                    assert 'decoding_time' not in profile_.pf
                    profile.incremental_updates(profile_)

                    end_time = time.time()
                    gap_time = end_time - start_time
                    tokens = output_ids.numel() - len(input_ids[0])
                    overall_time += gap_time
                    overall_gen += tokens
                    overall_tp += [tokens / gap_time]
                    overall_steps += profile_.get_profile('forward_count')
                    # overall_steps += profile_.get_profile('forward_count')
                    assert overall_steps == profile.get_profile('forward_count')
                    count_gen += 1

                    stats[question_idx][j] = [gap_time, tokens]
                    if get_device(model.draft_config) == 0 and ds_local_rank == 0:
                        logger.success(
                            f"step {i} \t turn {j}\ttime: {gap_time:.4f}\tgenerated tokens: {tokens}\t"
                            f"Forward steps: {profile_.get_profile('forward_count')}\t"
                            f"Throughput: {(tokens / gap_time):.4f}")

                    if model.config.is_encoder_decoder:
                        output_ids = output_ids[0]
                    else:
                        output_ids = output_ids[0][len(input_ids[0]):]

                    if conv.stop_token_ids:
                        stop_token_ids_index = [
                            i
                            for i, id in enumerate(output_ids)
                            if id in conv.stop_token_ids
                        ]
                        if len(stop_token_ids_index) > 0:
                            output_ids = output_ids[: stop_token_ids_index[0]]

                    output = tokenizer.decode(
                        output_ids,
                        spaces_between_special_tokens=False,
                    )
                    if conv.stop_str and output.find(conv.stop_str) > 0:
                        output = output[: output.find(conv.stop_str)]

                    logger.info('==========' * 5 + f'output:' + '==========' * 5)
                    logger.info(f'{output}')
                    logger.info('==========' * 5 + f'output end' + '==========' * 5)

                    for special_token in tokenizer.special_tokens_map.values():
                        if isinstance(special_token, list):
                            for special_tok in special_token:
                                output = output.replace(special_tok, "")
                        else:
                            output = output.replace(special_token, "")
                    output = output.strip()
                    # message = update_agent_response(output, message)
                    if hasattr(model, 'context_cache'):
                        model.context_cache.stat_info()
                    if len(overall_tp) > 1:
                        info = (f"AVERAGE THROUGHPUT TILL NOW: {(sum(overall_tp) / count_gen):.4f} "
                                f"({stdev(overall_tp):.4f}) "
                                f"AVERAGE DECODING EFFICIENCY TILL NOW: "
                                f"{((overall_gen / overall_steps) if overall_steps else 0):.4f}\t")
                        info += (f"AVERAGE AUX TOKENS NUMBER TILL NOW: "
                                 f"{profile.get_profile('aux_tokens_num') / overall_steps if overall_steps else 0:.4f}\t"
                                 f"AVERAGE CANDIDATE TOKENS NUMBER TILL NOW: "
                                 f"{profile.get_profile('cdt_tokens_num') / overall_steps if overall_steps else 0:.4f}")
                        logger.success(info)
                    if conv.name == "xgen" and output.startswith("Assistant:"):
                        output = output.replace("Assistant:", "", 1).strip()

                    '''
                    except RuntimeError as e:
                        print("ERROR question ID: ", question["question_id"])
                        output = "ERROR"
                    '''
                    turns.append(output)
                    conv.messages[-1][-1] = output

            if model.draft_config.flush_interval == 0:
                pass
                # model.context_cache.reduce_token_map(model.draft_config.keep_num)
            elif (count_conv + 1) % model.draft_config.flush_interval == 0 and hasattr(model, 'context_cache'):
                model.context_cache.reduce_cache(model.draft_config.keep_num)
            if hasattr(model,'context_cache'):
                model.context_cache.c = 0

        if get_device(model.draft_config) == 0 and ds_local_rank == 0:
            logger.debug(profile)
            key_average_keys = ['aux_tokens_num', 'cdt_tokens_num', 'compressed_tokens_num', "decoding_time",
                                'iter_time',
                                'before_forward_time', "prepare_input_time", 'cache_retrieve_time',
                                "corpus_cdt_time", 'context_cdt_time',
                                'forward_time', 'prepare_ids_time', 'LlamaModel_forward_time',
                                'attention_mask_time',
                                'mask_aux_time', 'mask_hm_time', 'mask_fill_time', 'mask_cdt_time',
                                'layer_forward_time',
                                'after_forward_time', 'decode_item_time', 'token_map_update_time',
                                'update_draft_branches_time', 'update_kv_time', 'hit_time',
                                'model_kwargs_update_time']
            logger.debug('=================average profile==============')
            s = ''
            for k in key_average_keys:
                s += f'{k}\t'
            logger.debug(s)
            s = ''
            for k in key_average_keys:
                s += f'{profile.get_profile(k) / overall_steps if overall_steps else 0}\t'
            logger.debug(s)

    if len(overall_tp) > 1:
        logger.success('===============FINAL PPROFILE==================')
        logger.success(
            f"AVERAGE THROUGHPUT1: {(sum(overall_tp) / count_gen):.4f} ({stdev(overall_tp):.4f}) \t"
            f"AVERAGE THROUGHPUT2: {(overall_gen / overall_time):.4f}")
    info = (f"OVERALL GEN: {overall_gen}\tSTEP: {overall_steps}\t"
            f"AVG DECODING EFFICIENCY: {(overall_gen / overall_steps if overall_steps > 0 else 0):.4f} \t")
    info += (f"AVERAGE AUX TOKENS NUMBER: "
             f"{profile.get_profile('aux_tokens_num') / overall_steps if overall_steps else 0:.4f} \t"
             f"AVERAGE CANDIDATE TOKENS NUMBER: "
             f"{profile.get_profile('cdt_tokens_num') / overall_steps if overall_steps else 0:.4f}")
    logger.success(
        info
    )
