import os
import time
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask as _expand_mask
from transformers.models.llama.modeling_llama import BaseModelOutputWithPast, CausalLMOutputWithPast

from ...inference_profile import InferProfile
from ...draft_branch import *
from time import perf_counter

def generate_hm_with_aux(widths, device, aux_idxs):
    aux_idxs.insert(0, [0, 1])
    aux_idxs_ = torch.tensor(aux_idxs)
    start_indices = aux_idxs_[:, 0]
    end_indices = aux_idxs_[:, 1]

    pre_fill_width = widths[0] + 1

    col_indices = torch.arange(0, pre_fill_width)
    mask = (col_indices >= start_indices.unsqueeze(1)) & (col_indices < end_indices.unsqueeze(1))
    mask[:, 0] = True
    return mask.to(device)


def generate_hm_with_dia(width, device):
    return torch.eye(width, dtype=torch.bool, device=device)


def set_dia_zero(matrix: torch.Tensor, start_row, start_col, end_row, end_col):
    assert end_row - start_row == end_col - start_col
    matrix[start_row:end_row, start_col:end_col] = torch.finfo(torch.float16).min
    matrix[start_row:end_row, start_col:end_col] = matrix[start_row:end_row, start_col:end_col].fill_diagonal_(0)

    return matrix


def set_lower_triangular_true_efficient(matrix, start_row, start_col, end_row, end_col, set_value=0):
    matrix_shape = matrix.shape
    assert 0 <= start_row < matrix_shape[0]
    assert 0 <= end_row <= matrix_shape[0]
    assert 0 <= start_col < matrix_shape[1]
    assert 0 <= end_col <= matrix_shape[1]
    rows, cols = torch.arange(matrix_shape[0]).unsqueeze(1), torch.arange(matrix_shape[1])

    mask = ((cols <= rows) & (rows >= start_row) & (rows < end_row) & (cols >= start_col) & (cols < end_col)).to(
        matrix.device)

    matrix.masked_fill_(mask, set_value)

    return matrix



def prepare_verify_mask_(start, end, mask, gram_n, candidate_len, dtype):
    candidate_num= candidate_len//gram_n
    if gram_n == 2:
        small_m = torch.tensor([0, torch.finfo(dtype).min]).repeat(candidate_num)[:-1]
        mask[start:end, start:end] = mask[start:end, start:end].fill_diagonal_(0).diagonal_scatter(small_m, -1)
    elif gram_n == 3:
        small_m1 = torch.tensor([0, 0, torch.finfo(dtype).min]).repeat(candidate_num)[:-1]
        small_m2 = torch.tensor([0, torch.finfo(dtype).min, torch.finfo(dtype).min]).repeat(candidate_num)[:-2]
        mask[start:end, start:end] = mask[start:end, start:end].fill_diagonal_(0).diagonal_scatter(small_m1,
                                                                                                   -1).diagonal_scatter(
            small_m2, -2)
    elif gram_n == 4:
        small_m1 = torch.tensor([0, 0, 0, torch.finfo(dtype).min]).repeat(candidate_num)[:-1]
        small_m2 = torch.tensor([0, 0, torch.finfo(dtype).min, torch.finfo(dtype).min]).repeat(candidate_num)[:-2]
        small_m3 = torch.tensor(
            [0, torch.finfo(dtype).min, torch.finfo(dtype).min, torch.finfo(dtype).min]).repeat(
            candidate_num)[:-3]
        mask[start:end, start:end] = mask[start:end, start:end].fill_diagonal_(0).diagonal_scatter(small_m1, -1). \
            diagonal_scatter(small_m2, -2). \
            diagonal_scatter(small_m3, -3)
    elif gram_n == 5:
        small_m1 = torch.tensor([0, 0, 0, 0, torch.finfo(dtype).min]).repeat(candidate_num)[:-1]
        small_m2 = torch.tensor([0, 0, 0, torch.finfo(dtype).min, torch.finfo(dtype).min]).repeat(candidate_num)[
                   :-2]
        small_m3 = torch.tensor(
            [0, 0, torch.finfo(dtype).min, torch.finfo(dtype).min, torch.finfo(dtype).min]).repeat(candidate_num)[
                   :-3]
        small_m4 = torch.tensor([0, torch.finfo(dtype).min, torch.finfo(dtype).min, torch.finfo(dtype).min,
                                 torch.finfo(dtype).min]).repeat(candidate_num)[:-4]
        mask[start:end, start:end] = mask[start:end, start:end].fill_diagonal_(0).diagonal_scatter(small_m1,
                                                                                                   -1).diagonal_scatter(
            small_m2, -2).diagonal_scatter(small_m3, -3).diagonal_scatter(small_m4, -4)
    elif gram_n == 6:
        small_m1 = torch.tensor([0, 0, 0, 0, 0, torch.finfo(dtype).min]).repeat(candidate_num)[:-1]
        small_m2 = torch.tensor([0, 0, 0, 0, torch.finfo(dtype).min, torch.finfo(dtype).min]).repeat(
            candidate_num)[
                   :-2]
        small_m3 = torch.tensor(
            [0, 0, 0, torch.finfo(dtype).min, torch.finfo(dtype).min, torch.finfo(dtype).min]).repeat(
            candidate_num)[
                   :-3]
        small_m4 = torch.tensor([0, 0, torch.finfo(dtype).min, torch.finfo(dtype).min, torch.finfo(dtype).min,
                                 torch.finfo(dtype).min]).repeat(candidate_num)[:-4]
        small_m5 = torch.tensor(
            [0, torch.finfo(dtype).min, torch.finfo(dtype).min, torch.finfo(dtype).min, torch.finfo(dtype).min,
             torch.finfo(dtype).min]).repeat(candidate_num)[:-5]
        mask[start:end, start:end] = mask[start:end, start:end].fill_diagonal_(0).diagonal_scatter(small_m1,
                                                                                                   -1).diagonal_scatter(
            small_m2, -2).diagonal_scatter(small_m3, -3).diagonal_scatter(small_m4, -4).diagonal_scatter(
            small_m5,
            -5)
    elif gram_n == 7:
        small_m1 = torch.tensor([0, 0, 0, 0, 0, 0, torch.finfo(dtype).min]).repeat(candidate_num)[:-1]
        small_m2 = torch.tensor([0, 0, 0, 0, 0, torch.finfo(dtype).min, torch.finfo(dtype).min]).repeat(
            candidate_num)[:-2]
        small_m3 = torch.tensor(
            [0, 0, 0, 0, torch.finfo(dtype).min, torch.finfo(dtype).min, torch.finfo(dtype).min]).repeat(
            candidate_num)[:-3]
        small_m4 = torch.tensor(
            [0, 0, 0, torch.finfo(dtype).min, torch.finfo(dtype).min, torch.finfo(dtype).min,
             torch.finfo(dtype).min]).repeat(candidate_num)[:-4]
        small_m5 = torch.tensor(
            [0, 0, torch.finfo(dtype).min, torch.finfo(dtype).min, torch.finfo(dtype).min,
             torch.finfo(dtype).min,
             torch.finfo(dtype).min]).repeat(candidate_num)[:-5]
        small_m6 = torch.tensor(
            [0, torch.finfo(dtype).min, torch.finfo(dtype).min, torch.finfo(dtype).min, torch.finfo(dtype).min,
             torch.finfo(dtype).min, torch.finfo(dtype).min]).repeat(candidate_num)[:-6]
        mask[start:end, start:end] = mask[start:end, start:end].fill_diagonal_(0).diagonal_scatter(small_m1,
                                                                                                   -1).diagonal_scatter(
            small_m2, -2).diagonal_scatter(small_m3, -3).diagonal_scatter(small_m4, -4).diagonal_scatter(
            small_m5,
            -5).diagonal_scatter(
            small_m6, -6)
    else:
        mask[start:end, start:end] = mask[start:end, start:end].fill_diagonal_(0)
        for i in range(gram_n - 1):  # 7 : 0 - 5
            small_l = [0] * (gram_n - i - 1) + [torch.finfo(dtype).min] * (i + 1)
            small_m = torch.tensor(small_l).repeat(candidate_num)[:-1 - i]
            mask[start:end, start:end] = mask[start:end, start:end].diagonal_scatter(small_m, -1 - i)
    return mask


def prepare_verify_mask(branch_cdt_tokens, branch_gram_n, corpus_cdt_tokens, corpus_gram_n, mask, dtype):
    mask[:, 0] = 0
    cad_len = len(branch_cdt_tokens) + len(corpus_cdt_tokens)
    l_corpus = len(corpus_cdt_tokens)
    l_branch = len(branch_cdt_tokens)
    if cad_len > 0:
        start = -cad_len
        if l_corpus > 0 and l_branch > 0:
            end = -l_corpus
            mask = prepare_verify_mask_(start, end, mask, branch_gram_n, len(branch_cdt_tokens), dtype)
            start = -l_corpus
            end = None
            mask = prepare_verify_mask_(start, end, mask, corpus_gram_n, len(corpus_cdt_tokens), dtype)

        elif l_corpus > 0 and l_branch == 0:
            end = None
            mask = prepare_verify_mask_(start, end, mask, corpus_gram_n, len(corpus_cdt_tokens), dtype)
        elif l_branch > 0 and l_corpus == 0:
            end = None
            mask = prepare_verify_mask_(start, end, mask, branch_gram_n, len(branch_cdt_tokens), dtype)

        else:
            raise RuntimeError('wrong candidate tokens')

    return mask



def make_SD_causal_mask_multi_branch(
        draft_branches: DraftBranch,
        is_prefill: bool, branch_cdt_tokens, branch_gram_n,
        corpus_cdt_tokens, corpus_gram_n, input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device,
        past_key_values_length: int = 0, prompt_size=-1,
):
    profile = InferProfile()
    t0 = perf_counter()
    bsz, tgt_len = input_ids_shape

    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device, dtype=dtype)


    aux_len = sum(draft_branches.aux_branch.aux_sizes)
    profile.incremental_update('mask_init_time', perf_counter() - t0)
    if aux_len > 0:
        aux_all_len = sum(draft_branches.aux_branch.aux_sizes)
        aux_attn = draft_branches.aux_branch.attn_mask.to(device)
        if is_prefill:
            t0 = perf_counter()
            if prompt_size == -1:
                prompt_size = tgt_len - sum(draft_branches.widths)
            mask = set_lower_triangular_true_efficient(mask, 0, 0, prompt_size, prompt_size)
            pre_fill_idx = prompt_size
            mask[prompt_size:tgt_len, 0:prompt_size] = 0
            mask[pre_fill_idx:pre_fill_idx + aux_all_len, pre_fill_idx:pre_fill_idx + aux_all_len] = aux_attn

            profile.incremental_update('mask_prefill_time', perf_counter() - t0)
            del t0
        else:
            t0 = perf_counter()
            pre_idx = 1
            mask[:, 0] = 0

            mask[pre_idx:pre_idx + aux_all_len, pre_idx:pre_idx + aux_all_len] = draft_branches.aux_branch.attn_mask

            profile.incremental_update('mask_aux_time', perf_counter() - t0)
            t1 = perf_counter()
            hm = draft_branches.aux_branch.hm_mask.to(device)
            s0, s1 = hm.shape
            all_l_size = sum(draft_branches.widths[1:])
            assert all_l_size % s0 == 0
            mask[aux_len + pre_idx:aux_len + pre_idx + all_l_size, :s1].masked_fill_(
                hm.repeat(len(draft_branches.widths[1:]), 1),
                                                                         0)

            profile.incremental_update('mask_hm_time', perf_counter() - t1)
            t2 = perf_counter()

            mask[aux_len + pre_idx:aux_len + pre_idx + all_l_size,
            aux_len + pre_idx:aux_len + pre_idx + all_l_size] = draft_branches.attn_mask.to(device)

            profile.incremental_update('mask_fill_time', perf_counter() - t2)
            t3 = perf_counter()
            if len(branch_cdt_tokens) > 0 or len(corpus_cdt_tokens) > 0:
                mask = prepare_verify_mask(branch_cdt_tokens, branch_gram_n, corpus_cdt_tokens, corpus_gram_n, mask,
                                           dtype)
            else:
                assert tgt_len == sum(draft_branches.widths) + 1
            profile.incremental_update('mask_cdt_time', perf_counter() - t3)

    assert mask.dtype == torch.float16
    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length), profile


def prepare_SD_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length, others):
    # create causal mask
    # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
    is_prefill, branch_cdt_tokens, branch_gram_n, corpus_cdt_tokens, corpus_gram_n, draft_branches = others
    combined_attention_mask = None
    # print("size: ", input_shape, past_key_values_length)
    mask_profile = InferProfile()
    if input_shape[-1] > 1:
        combined_attention_mask, mask_profile = make_SD_causal_mask_multi_branch(draft_branches, is_prefill,
                                                                                 branch_cdt_tokens,
                                                                   branch_gram_n, corpus_cdt_tokens, corpus_gram_n,
                                                                   input_shape, torch.float16, inputs_embeds.device,
                                                                   past_key_values_length=past_key_values_length)

    if attention_mask is not None:
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        # expanded_attn_mask = AttentionMaskConverter._expand_mask(mask=attention_mask, dtype=torch.float16,
        #                                                          tgt_len=input_shape[-1]).to(inputs_embeds.device)
        expanded_attn_mask = _expand_mask(attention_mask, torch.float16, tgt_len=input_shape[-1]).to(
            inputs_embeds.device)
        combined_attention_mask = (
            expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
        )

    return combined_attention_mask, mask_profile


def LlamaModelSDforward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        is_prefill: bool = False,
        branch_cdt_tokens=None,
        branch_gram_n=4,
        corpus_cdt_tokens=None,
        corpus_gram_n=4,
        draft_branches=None,
) -> Union[Tuple, BaseModelOutputWithPast]:
    profile_LlamaModelSD_forward = InferProfile()

    if branch_cdt_tokens is None:
        branch_cdt_tokens = []
    if corpus_cdt_tokens is None:
        corpus_cdt_tokens = []
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache

    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    # retrieve input_ids and inputs_embeds
    if input_ids is not None and inputs_embeds is not None:
        raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
    elif input_ids is not None:
        batch_size, seq_length = input_ids.shape
    elif inputs_embeds is not None:
        batch_size, seq_length, _ = inputs_embeds.shape
    else:
        raise ValueError("You have to specify either input_ids or inputs_embeds")
    seq_length_with_past = seq_length
    past_key_values_length = 0

    if use_cache:
        use_legacy_cache = not isinstance(past_key_values, Cache)
        if use_legacy_cache:
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)
        past_key_values_length = past_key_values.get_usable_length(seq_length)

    if position_ids is None:
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        position_ids = torch.arange(
            past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
        )
        position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
    else:
        # try:
        position_ids = position_ids.view(-1, seq_length).long()
        # except:
        #     raise ('error')
        #

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)
    if attention_mask is None:
        attention_mask = torch.ones(
            (batch_size, seq_length_with_past), dtype=torch.bool, device=inputs_embeds.device
        )

    t0 = time.perf_counter()
    attention_mask, mask_profile = self.prepare_SD_decoder_attention_mask(attention_mask, (batch_size, seq_length),
                                                                          inputs_embeds,
                                                            past_key_values_length,
                                                            (is_prefill, branch_cdt_tokens, branch_gram_n,
                                                             corpus_cdt_tokens, corpus_gram_n,
                                                             draft_branches), )
    # torch.cuda.synchronize()

    profile_LlamaModelSD_forward.incremental_update('attention_mask_time', time.perf_counter() - t0)
    profile_LlamaModelSD_forward.incremental_updates(mask_profile)

    hidden_states = inputs_embeds

    if self.gradient_checkpointing and self.training:
        if use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
            )
            use_cache = False

    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    next_decoder_cache = None
    t0 = time.perf_counter()
    for decoder_layer in self.layers:
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if self.gradient_checkpointing and self.training:
            layer_outputs = self._gradient_checkpointing_func(
                decoder_layer.__call__,
                hidden_states,
                attention_mask,
                position_ids,
                past_key_values,
                output_attentions,
                use_cache,
            )
        else:
            layer_outputs = decoder_layer.forward(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

        hidden_states = layer_outputs[0]

        if use_cache:
            next_decoder_cache = layer_outputs[2 if output_attentions else 1]

        if output_attentions:
            all_self_attns += (layer_outputs[1],)
    torch.cuda.synchronize()
    profile_LlamaModelSD_forward.incremental_update('layer_forward_time', time.perf_counter() - t0)
    hidden_states = self.norm(hidden_states)

    # add hidden states from the last decoder layer
    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    next_cache = None
    if use_cache:
        next_cache = next_decoder_cache.to_legacy_cache() if use_legacy_cache else next_decoder_cache
    if not return_dict:
        return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
    return (BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    ), profile_LlamaModelSD_forward)



def SDforward(
        self,
        input_ids: torch.LongTensor = None,
        draft_branches: Optional[DraftBranch] = None,
        cdt_content: Tuple = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
) -> Union[Tuple, CausalLMOutputWithPast]:
    branch_cdt_tokens, branch_gram_n, corpus_cdt_tokens, corpus_gram_n = cdt_content

    branch_cdt_tokens = [] if branch_cdt_tokens is None else branch_cdt_tokens
    corpus_cdt_tokens = [] if corpus_cdt_tokens is None else corpus_cdt_tokens
    forward_profile = InferProfile()
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    assert labels is None, " Inference Mode "
    assert input_ids.size(0) == 1, " single batch only "
    assert corpus_gram_n == 0 or len(corpus_cdt_tokens) % corpus_gram_n == 0
    assert branch_gram_n == 0 or len(branch_cdt_tokens) % branch_gram_n == 0
    
    if past_key_values is not None:
        past_size = past_key_values[0][0].size(2)
    else:
        past_size = 0

    prefill_size = input_ids.size(1)
    for layer in self.model.layers:
        layer.self_attn.cur_len = prefill_size

    lst_id = position_ids[0][-1].item()

    t0 = time.perf_counter()
    ids_list, all_past, attn_size = draft_branches.prepare_ids(lst_id)

    assert (len(all_past) == sum(draft_branches.widths))
    forward_profile.incremental_update('prepare_ids_time', time.perf_counter() - t0)
    if len(branch_cdt_tokens) > 0 or len(corpus_cdt_tokens) > 0:
        input_ids = torch.cat((input_ids, torch.tensor(all_past + branch_cdt_tokens + corpus_cdt_tokens,
                                                       device=input_ids.device,
                                                       dtype=input_ids.dtype).unsqueeze(0)), dim=1)

        branch_cdt_ids = list(range(lst_id + 1, lst_id + 1 + branch_gram_n)) * (
                len(branch_cdt_tokens) // branch_gram_n)
        corpus_cdt_ids = list(range(lst_id + 1, lst_id + 1 + corpus_gram_n)) * (
                len(corpus_cdt_tokens) // corpus_gram_n)

        position_ids = torch.cat(
            (position_ids, torch.tensor(ids_list + branch_cdt_ids + corpus_cdt_ids, device=input_ids.device,
                                        dtype=input_ids.dtype).unsqueeze(0)), dim=1)
        attention_mask = torch.cat(
            (attention_mask, torch.ones(1, attn_size + len(branch_cdt_tokens) + len(corpus_cdt_tokens),
                                        device=input_ids.device, dtype=input_ids.dtype)), dim=1)

    else:

        input_ids = torch.cat(
            (input_ids, torch.tensor(all_past, device=input_ids.device, dtype=input_ids.dtype).unsqueeze(0)), dim=1)
        position_ids = torch.cat(
            (position_ids, torch.tensor(ids_list, device=input_ids.device, dtype=input_ids.dtype).unsqueeze(0)), dim=1)
        attention_mask = torch.cat((attention_mask, torch.ones(1, attn_size,
                                                               device=input_ids.device, dtype=input_ids.dtype)), dim=1)
    step_len = attention_mask.size(1)

    is_prefill = True if past_key_values is None else False
    t0 = time.perf_counter()
    outputs, llama_model_SD_forward_profile = self.model.LlamaModelSDforward(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        is_prefill=is_prefill,
        branch_cdt_tokens=branch_cdt_tokens,
        branch_gram_n=branch_gram_n,
        corpus_cdt_tokens=corpus_cdt_tokens,
        corpus_gram_n=corpus_gram_n,
        draft_branches=draft_branches,
    )
    torch.cuda.synchronize()
    forward_profile.incremental_updates(llama_model_SD_forward_profile)
    forward_profile.incremental_update('LlamaModel_forward_time', time.perf_counter() - t0)
    hidden_states = outputs[0]

    if self.config.pretraining_tp > 1:
        lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
        logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
        logits = torch.cat(logits, dim=-1)
    else:
        logits = self.lm_head(hidden_states)

    logits = logits.float()

    loss = None
    if labels is not None:  # train
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = CrossEntropyLoss()
        shift_logits = shift_logits.view(-1, self.config.vocab_size)
        shift_labels = shift_labels.view(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels)

    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    ret = CausalLMOutputWithPast(
        loss=loss,
        logits=logits.to(input_ids.device),
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )
    ret.kvcache_len = prefill_size + past_size
    ret.step_len = step_len

    if branch_cdt_tokens or corpus_cdt_tokens:
        cdt_len = len(branch_cdt_tokens) + len(corpus_cdt_tokens)
    else:
        cdt_len = 0

    ret.out_logits = ret.logits[:, prefill_size - 1, :].to(input_ids.device)
    assert draft_branches.filled_depth != -1

    if len(draft_branches.aux_branch.aux_list) == 0:
        assert len(all_past) == sum([len(p) for p in draft_branches.branches if p is not None])
    else:
        assert len(all_past) == sum([len(p) for p in draft_branches.branches if p is not None]) + \
               sum(len(a) for a in draft_branches.aux_branch.aux_list) - len(draft_branches.branches[0])
    if draft_branches.filled_depth == 0 and len(draft_branches.aux_branch.aux_list) > 0:
        aux_size = [len(a) for a in draft_branches.aux_branch.aux_list]
        draft_branch_idx = [sum(aux_size[:i]) - 1 for i in range(1, len(draft_branches.aux_branch.aux_list) + 1)]
        assert cdt_len == 0
        if cdt_len > 0:
            ret.inp_logits = ret.logits[:, -sum(aux_size):-cdt_len, :][:, draft_branch_idx, :].to(input_ids.device)
            ret.verify_logits = ret.logits[:, -cdt_len:, :].to(input_ids.device)
            ret.draft_logits = ret.logits[:, -sum(aux_size):-cdt_len, :][:, draft_branch_idx, :].to(input_ids.device)
        else:
            ret.inp_logits = ret.logits[:, -sum(aux_size):, :][:, draft_branch_idx, :].to(input_ids.device)
            ret.draft_logits = ret.logits[:, -sum(aux_size):, :].to(input_ids.device)

    else:
        if cdt_len > 0:
            ret.inp_logits = ret.logits[:, -len(draft_branches.branches[draft_branches.filled_depth]) - cdt_len:-cdt_len,
                             :].to(input_ids.device)
            ret.verify_logits = ret.logits[:, -cdt_len:, :].to(input_ids.device)
            ret.draft_logits = ret.logits[:, -len(all_past) - cdt_len:-cdt_len, :].to(input_ids.device)
        else:
            ret.inp_logits = ret.logits[:, -len(draft_branches.branches[draft_branches.filled_depth]):, :].to(
                input_ids.device)
            ret.draft_logits = ret.logits[:, -len(all_past):, :].to(input_ids.device)

    return ret, forward_profile


