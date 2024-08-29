import os
import warnings
from time import perf_counter
from typing import List, Optional, Union

import torch
import torch.distributed as dist
from loguru import logger
from transformers import LogitsProcessorList
from transformers.generation.utils import GenerateEncoderDecoderOutput, GenerateDecoderOnlyOutput
from transformers.generation.utils import LogitsProcessorList, StoppingCriteriaList, GreedySearchOutput

from .cache.context_cache import ContextCache
from .draft_branch import DraftBranch
from .inference_profile import *

COLOR_PRINT = int(os.environ.get("COLOR_PRINT", 0))

FUNC_MAP = {}


def greedy_search_proxy(self, *args, **kwargs):
    SELF_DRAFT = self.draft_config.self_draft
    if SELF_DRAFT:
        return self_draft_greedy_search(self, corpus_cache=self.corpus_cache, chat=False, *args, **kwargs)
    else:
        return FUNC_MAP["greedy_search"](self, *args, **kwargs)


def self_draft_greedy_search(
        self,
        input_ids: torch.LongTensor,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        synced_gpus: bool = False,
        streamer: Optional["BaseStreamer"] = None,
        chat: bool = False,
        corpus_cache=None,
        **model_kwargs,
) -> Union[GreedySearchOutput, torch.LongTensor]:
    """
    CUSTOM forward
    """
    logger.success('Inference with self draft greedy decode method.')
    t0 = perf_counter()
    profile = InferProfile()
    logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
    stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
    if max_length is not None:
        warnings.warn(
            "`max_length` is deprecated in this function, use"
            " `stopping_criteria=StoppingCriteriaList([MaxLengthCriteria(max_length=max_length)])` instead.",
            UserWarning,
        )
        stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
    pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
    eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id
    if isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]
    eos_token_id_tensor = torch.tensor(eos_token_id).to(input_ids.device) if eos_token_id is not None else None
    output_scores = output_scores if output_scores is not None else self.generation_config.output_scores
    output_attentions = (
        output_attentions if output_attentions is not None else self.generation_config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.generation_config.output_hidden_states
    )
    return_dict_in_generate = (
        return_dict_in_generate
        if return_dict_in_generate is not None
        else self.generation_config.return_dict_in_generate
    )

    # init attention / hidden states / scores tuples
    scores = () if (return_dict_in_generate and output_scores) else None
    decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
    cross_attentions = () if (return_dict_in_generate and output_attentions) else None
    decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

    # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
    if return_dict_in_generate and self.config.is_encoder_decoder:
        encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
        encoder_hidden_states = (
            model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
        )

    # keep track of which sequences are already finished
    unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)

    this_peer_finished = False

    WO_CTX = int(os.environ.get("WO_CTX", 0))
    CORPUS_GRAM_N = 4
    BRANCH_GRAM_N = self.draft_config.context_gram_n
    branch_num = self.draft_config.branch_num
    branch_len = self.draft_config.branch_len
    ALWAYS_FWD_ONE = getattr(self.draft_config, 'always_fwd_one', 1)
    self.draft_config.past_ini_len = branch_num + branch_len - 3

    import random
    random.seed(10)
    all_old_tokens = input_ids[0].tolist()
    init_len = len(all_old_tokens)

    draft_branches = DraftBranch(all_old_tokens, branch_num, branch_len,
                                 self.draft_config.max_pad_len, self.draft_config.init_method, ALWAYS_FWD_ONE)

    if hasattr(self, 'context_cache'):
        self.context_cache = ContextCache(self.context_cache, gram_n=self.draft_config.context_gram_n)
    else:
        self.context_cache = ContextCache(gram_n=self.draft_config.context_gram_n)
    steps = 0
    is_prefill = True

    if chat:
        init = self.tokenizer.decode(all_old_tokens, skip_special_tokens=True,
                                     spaces_between_special_tokens=False, clean_up_tokenization_spaces=True, )
        prev = len(init)

    lst_token = None

    step_count = 0

    profile.set_profile('prepare_time', perf_counter() - t0)

    while True:
        ti = perf_counter()
        profile.incremental_update('forward_count', 1)
        if synced_gpus:
            # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
            # The following logic all ows an early break if all peers finished generating their sequence
            this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
            # send 0.0 if we finished, 1.0 otherwise
            dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
            # did all peers finish? the reduced sum will be 0.0 then
            if this_peer_finished_flag.item() == 0.0:
                break
        t0 = perf_counter()
        model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
        profile.incremental_update('prepare_input_time', perf_counter() - t0)
        # profile['prepare_input_time'] += perf_counter() - t0

        del t0

        context_cdt_tokens, corpus_cdt_tokens = [], []
        context_cdt_tup, corpus_cdt_tup = [], []
        context_cdt_count, corpus_cdt_count = 0, 0
        branch_cdt_tup = []
        t1 = perf_counter()
        # ========Get draft sequences=========
        if draft_branches.filled_depth >= branch_len - 2 and self.draft_config.use_context:
            t0 = perf_counter()
            context_cdt_tokens, context_cdt_tup, context_cdt_count \
                = self.context_cache.retrieve(all_old_tokens)
            profile.incremental_update('context_cdt_time', perf_counter() - t0)
            profile.incremental_update('context_cdt_count', len(context_cdt_tup))

        overlap_c_ = 0
        if self.draft_config.use_corpus and not is_prefill:
            _ = self.context_cache.max_val_len - context_cdt_count
            if int(self.draft_config.limit_corpus_count):
                if _ == 0:
                    corpus_cdt_max = 2
                else:
                    corpus_cdt_max = min(3, _)
            else:
                corpus_cdt_max = _
            t0 = perf_counter()
            if WO_CTX:
                corpus_cdt_max = self.context_cache.max_val_len
            (corpus_cdt_tokens, corpus_cdt_tup,
             corpus_cdt_count, overlap_c_) = corpus_cache.retrieve_from_corpus(lst_token,
                                                                               self.draft_config.search_with_dif_key_len,
                                                                               all_old_tokens,
                                                                               self.draft_config.max_key_len,
                                                                               corpus_cdt_max, pre_cdts=context_cdt_tup)
            profile.incremental_update('corpus_cdt_count', len(corpus_cdt_tup))
            profile.incremental_update('corpus_cdt_time', perf_counter() - t0)

        profile.incremental_update('cache_retrieve_time', perf_counter() - t1)
        t_after_retrieve = perf_counter()

        if self.draft_config.use_context and self.draft_config.use_corpus:
            branch_cdt_tup = set(context_cdt_tup)
            assert len(branch_cdt_tup) == len(context_cdt_tup)
            assert len(set(context_cdt_tup + corpus_cdt_tup)) == len(context_cdt_tup + corpus_cdt_tup)
            branch_cdt_tokens = []
            for i in branch_cdt_tup:
                branch_cdt_tokens += list(i)

            branch_cdt_count = len(branch_cdt_tup)

        elif self.draft_config.use_corpus:
            branch_cdt_tokens = []
            branch_cdt_count = 0
        elif self.draft_config.use_context:
            corpus_cdt_count = 0
            corpus_cdt_tokens = []
            branch_cdt_tup = set(context_cdt_tup)
            branch_cdt_tokens = []
            for i in branch_cdt_tup:
                branch_cdt_tokens += list(i)
            branch_cdt_count = len(branch_cdt_tup)

        else:
            branch_cdt_count = 0
            corpus_cdt_count = 0
            branch_cdt_tokens = []
            corpus_cdt_tokens = []
        profile.incremental_update('overlap_time', perf_counter() - t_after_retrieve)
        profile.incremental_update('total_candidate_count', branch_cdt_count + corpus_cdt_count)
        profile.incremental_update('after_retrieve_time', perf_counter() - t_after_retrieve)
        profile.incremental_update('before_forward_time', perf_counter() - ti)

        if lst_token is not None:
            assert lst_token == all_old_tokens[-1]

        assert return_dict_in_generate == False

        t0 = perf_counter()
        cdt_content = (branch_cdt_tokens, BRANCH_GRAM_N, corpus_cdt_tokens, CORPUS_GRAM_N)

        profile.incremental_update('aux_tokens_num',
                                   sum(draft_branches.widths))
        profile.incremental_update('cdt_tokens_num', len(branch_cdt_tokens) + len(corpus_cdt_tokens))
        outputs, forward_profile = self.SDforward(draft_branches=draft_branches, cdt_content=cdt_content,
                                                  output_attentions=output_attentions,
                                                  output_hidden_states=output_hidden_states, return_dict=True,
                                                  **model_inputs)

        torch.cuda.synchronize()
        profile.incremental_updates(forward_profile)
        profile.incremental_update('forward_time', perf_counter() - t0)

        ta = perf_counter()
        is_prefill = False
        steps += 1

        if synced_gpus and this_peer_finished:
            continue

        next_token_logits = outputs.out_logits

        next_tokens_scores = logits_processor(input_ids, next_token_logits)

        next_tokens = torch.argmax(next_tokens_scores, dim=-1)
        if eos_token_id is not None:
            if pad_token_id is None:
                raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)
        t0 = perf_counter()
        next_token = next_tokens.item()
        branch_hits = [next_token] + [0] * (BRANCH_GRAM_N - 1)
        corpus_hits = [next_token] + [0] * (CORPUS_GRAM_N - 1)

        profile.incremental_update('decode_item_time', perf_counter() - t0)

        t0 = perf_counter()

        if draft_branches.filled_depth >= draft_branches.branch_len - 2:

            draft_results = torch.argmax(outputs.draft_logits, dim=-1)[0].tolist()
            self.context_cache.update_cache(all_old_tokens, next_token, draft_branches, draft_results)

        else:
            pass

        profile.incremental_update('cache_update_time', perf_counter() - t0)

        t1 = perf_counter()
        inp_logits = torch.argmax(outputs.inp_logits, dim=-1)[0].tolist()
        if hasattr(outputs, 'verify_logits'):
            verify_results = torch.argmax(outputs.verify_logits, dim=-1)[0].tolist()
        else:
            verify_results = []
        new_results = torch.argmax(outputs.inp_logits, dim=-1)[0].tolist()

        max_hit, hit_point, pre_len, hits, g_size, candidate_len \
            = draft_branches.update_draft_branches_inter_logits(inp_logits, verify_results, new_results, cdt_content,
                                                                branch_hits, corpus_hits)

        profile.incremental_update('update_draft_branches_time', perf_counter() - t1)

        tau = perf_counter()
        if max_hit > 0:
            attention_mask = model_kwargs["attention_mask"]
            model_kwargs["attention_mask"] = torch.cat(
                (attention_mask, torch.ones(1, max_hit, device=attention_mask.device, dtype=attention_mask.dtype)),
                dim=1)

        t0 = perf_counter()
        past_key_values = []

        if pre_len == 0:
            for idx, kv in enumerate(outputs.past_key_values):
                assert outputs.step_len == kv[0].size(2)
                offset = outputs.step_len - candidate_len + hit_point * g_size
                if max_hit > 0:
                    kv[0][:, :, outputs.kvcache_len:outputs.kvcache_len + max_hit, :] = kv[0][:, :,
                                                                                        offset: offset + max_hit, :]
                    kv[1][:, :, outputs.kvcache_len:outputs.kvcache_len + max_hit, :] = kv[1][:, :,
                                                                                        offset: offset + max_hit, :]
                past_key_values.append(
                    (kv[0][:, :, :outputs.kvcache_len + max_hit, :], kv[1][:, :, :outputs.kvcache_len + max_hit, :]))
        else:
            for idx, kv in enumerate(outputs.past_key_values):
                # for hh in range(max_hit):
                assert outputs.step_len == kv[0].size(2)
                offset = outputs.step_len - candidate_len + len(branch_cdt_tokens) + hit_point * g_size
                if max_hit > 0:
                    kv[0][:, :, outputs.kvcache_len:outputs.kvcache_len + max_hit, :] = kv[0][:, :,
                                                                                        offset:offset + max_hit, :]
                    kv[1][:, :, outputs.kvcache_len:outputs.kvcache_len + max_hit, :] = kv[1][:, :,
                                                                                        offset:offset + max_hit, :]
                past_key_values.append(
                    (kv[0][:, :, :outputs.kvcache_len + max_hit, :], kv[1][:, :, :outputs.kvcache_len + max_hit, :]))
        outputs.past_key_values = past_key_values
        # torch.cuda.synchronize(input_ids.device)
        profile.incremental_update('update_kv_time', perf_counter() - t0)
        lst_token = hits[max_hit]
        # t0 = perf_counter()
        t0 = perf_counter()
        for hh in range(max_hit + 1):
            if eos_token_id is not None and hits[hh] == eos_token_id[0]:
                all_old_tokens.append(hits[hh])
                next_tokens = eos_token_id_tensor
                max_hit = hh
                # break
            else:
                all_old_tokens.append(hits[hh])

        step_count += 1

        if chat:
            all_str = self.tokenizer.decode(all_old_tokens, skip_special_tokens=True, \
                                            spaces_between_special_tokens=False, clean_up_tokenization_spaces=True, )
            if chat:
                from termcolor import colored
                if max_hit > 1:
                    not_hit = self.tokenizer.decode(all_old_tokens[:-max_hit + 1], skip_special_tokens=True, \
                                                    spaces_between_special_tokens=False,
                                                    clean_up_tokenization_spaces=True, )
                    pt = colored(not_hit[prev:], "blue") + colored(all_str[len(not_hit):], "blue")
                else:
                    pt = all_str[prev:]
                print(pt, flush=True, end="")
            prev = len(all_str)

        input_ids = torch.cat([input_ids, torch.tensor(hits[:max_hit + 1], device=next_tokens.device,
                                                       dtype=next_tokens.dtype).unsqueeze(0)], dim=-1)
        if streamer is not None:
            streamer.put(next_tokens.cpu())

        profile.incremental_update('hit_time', perf_counter() - t0)

        t0 = perf_counter()
        model_kwargs = self._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
        )
        profile.incremental_update('model_kwargs_update_time', perf_counter() - t0)

        if eos_token_id_tensor is not None:
            unfinished_sequences = unfinished_sequences.mul(
                next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
            )

            # stop when each sentence is finished
            if unfinished_sequences.max() == 0:
                this_peer_finished = True

        if stopping_criteria(input_ids, scores):
            this_peer_finished = True

        if this_peer_finished and not synced_gpus:
            break

        profile.incremental_update('after_update_time', perf_counter() - tau)
        profile.incremental_update('iter_time', perf_counter() - ti)
        profile.incremental_update('after_forward_time', perf_counter() - ta)

    t0 = perf_counter()
    for criteria in stopping_criteria:
        if hasattr(criteria, "max_length"):
            all_old_tokens = all_old_tokens[:criteria.max_length]
            input_ids = input_ids[:, :criteria.max_length]
    if max_length is not None:
        # print("max : ", max_length, init_len)
        all_old_tokens = all_old_tokens[:init_len + max_length]
        input_ids = input_ids[:][:init_len + max_length]

    logger.debug("\n" + profile.__str__())

    info = (f"\n==========================ACCELERATION===SUMMARY======================================\n"
            f"Generated tokens \t\t {len(all_old_tokens) - init_len}\n"
            f"Total steps \t\t {steps}\n"
            f"Decoding Efficiency \t\t {(len(all_old_tokens) - init_len) / steps}\n")
    info += (f"Corpus candidate count \t\t {profile.get_profile('corpus_cdt_count')}\n"
             f"Context candidate count \t\t {profile.get_profile('context_cdt_count')}\n"
             f"Average aux tokens for each forward step \t\t {profile.get_profile('aux_tokens_num') / steps:.4f}\n"
             f"Average candidate tokens for each forward step \t\t {profile.get_profile('cdt_tokens_num') / steps:.4f}\n"
             f"==========================ACCELERATION===SUMMARY======================================")
    logger.debug(info)
    if streamer is not None:
        streamer.end()

    if return_dict_in_generate:
        if self.config.is_encoder_decoder:
            return GreedySearchEncoderDecoderOutput(
                sequences=input_ids,
                scores=scores,
                encoder_attentions=encoder_attentions,
                encoder_hidden_states=encoder_hidden_states,
                decoder_attentions=decoder_attentions,
                cross_attentions=cross_attentions,
                decoder_hidden_states=decoder_hidden_states,
            )
        else:
            return GreedySearchDecoderOnlyOutput(
                sequences=input_ids,
                scores=scores,
                attentions=decoder_attentions,
                hidden_states=decoder_hidden_states,
            )
    else:
        profile.set_profile('post_process_time', perf_counter() - t0)

        return (input_ids, steps, profile)


def greedy_search(
        self,
        input_ids: torch.LongTensor,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        synced_gpus: bool = False,
        streamer: Optional["BaseStreamer"] = None,
        **model_kwargs,
) -> Union[GreedySearchOutput, torch.LongTensor]:
    r"""
    copy methods from transformers
    ```"""
    # init
    profile = InferProfile()

    t0 = perf_counter()
    logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
    stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
    if max_length is not None:
        warnings.warn(
            "`max_length` is deprecated in this function, use"
            " `stopping_criteria=StoppingCriteriaList([MaxLengthCriteria(max_length=max_length)])` instead.",
            UserWarning,
        )
        stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
    pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
    eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id
    if isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]
    eos_token_id_tensor = torch.tensor(eos_token_id).to(input_ids.device) if eos_token_id is not None else None
    output_scores = output_scores if output_scores is not None else self.generation_config.output_scores
    output_attentions = (
        output_attentions if output_attentions is not None else self.generation_config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.generation_config.output_hidden_states
    )
    return_dict_in_generate = (
        return_dict_in_generate
        if return_dict_in_generate is not None
        else self.generation_config.return_dict_in_generate
    )

    # init attention / hidden states / scores tuples
    scores = () if (return_dict_in_generate and output_scores) else None
    decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
    cross_attentions = () if (return_dict_in_generate and output_attentions) else None
    decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

    # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
    if return_dict_in_generate and self.config.is_encoder_decoder:
        encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
        encoder_hidden_states = (
            model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
        )

    # keep track of which sequences are already finished
    unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)

    this_peer_finished = False  # used by synced_gpus only
    profile.incremental_update('prepare_time', perf_counter() - t0)
    s = 0
    while True:
        profile.incremental_update('forward_count', 1)
        s += 1
        ti = perf_counter()
        if synced_gpus:
            # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
            # The following logic allows an early break if all peers finished generating their sequence
            this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
            # send 0.0 if we finished, 1.0 otherwise
            dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
            # did all peers finish? the reduced sum will be 0.0 then
            if this_peer_finished_flag.item() == 0.0:
                break

        # prepare model inputs
        model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

        profile.incremental_update('before_forward_time', perf_counter() - ti)
        # forward pass to get next token
        tf = perf_counter()
        outputs = self(
            **model_inputs,
            return_dict=True,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        profile.incremental_update('forward_time', perf_counter() - tf)
        ta = perf_counter()
        if synced_gpus and this_peer_finished:
            continue  # don't waste resources running the code we don't need

        next_token_logits = outputs.logits[:, -1, :]

        # pre-process distribution
        next_tokens_scores = logits_processor(input_ids, next_token_logits)

        # Store scores, attentions and hidden_states when required
        if return_dict_in_generate:
            if output_scores:
                scores += (next_tokens_scores,)
            if output_attentions:
                decoder_attentions += (
                    (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                )
                if self.config.is_encoder_decoder:
                    cross_attentions += (outputs.cross_attentions,)

            if output_hidden_states:
                decoder_hidden_states += (
                    (outputs.decoder_hidden_states,)
                    if self.config.is_encoder_decoder
                    else (outputs.hidden_states,)
                )

        # argmax
        next_tokens = torch.argmax(next_tokens_scores, dim=-1)
        # if input_ids.shape[-1] > 108:
        #     print('breakpoint')
        # finished sentences should have their next token be a padding token
        if eos_token_id is not None:
            if pad_token_id is None:
                raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

        # update generated ids, model inputs, and length for next step
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        if streamer is not None:
            streamer.put(next_tokens.cpu())
        model_kwargs = self._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
        )

        # if eos_token was found in one sentence, set sentence to finished
        if eos_token_id_tensor is not None:
            unfinished_sequences = unfinished_sequences.mul(
                next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
            )

            # stop when each sentence is finished
            if unfinished_sequences.max() == 0:
                this_peer_finished = True

        # stop if we exceed the maximum length
        if stopping_criteria(input_ids, scores):
            this_peer_finished = True

        if this_peer_finished and not synced_gpus:
            break

        profile.incremental_update('after_forward_time', perf_counter() - ta)
        profile.incremental_update('iter_time', perf_counter() - ti)

    if streamer is not None:
        streamer.end()

    if return_dict_in_generate:
        if self.config.is_encoder_decoder:
            return GenerateEncoderDecoderOutput(
                sequences=input_ids,
                scores=scores,
                encoder_attentions=encoder_attentions,
                encoder_hidden_states=encoder_hidden_states,
                decoder_attentions=decoder_attentions,
                cross_attentions=cross_attentions,
                decoder_hidden_states=decoder_hidden_states,
                past_key_values=model_kwargs.get("past_key_values"),
            )
        else:
            return GenerateDecoderOnlyOutput(
                sequences=input_ids,
                scores=scores,
                attentions=decoder_attentions,
                hidden_states=decoder_hidden_states,
                past_key_values=model_kwargs.get("past_key_values"),
            )
    else:
        return (input_ids, s, profile)
