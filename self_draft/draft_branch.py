import random

import torch


def copy_from(all_old_tokens):
    return random.choice(all_old_tokens)


def set_random_two_dim_list(dim_1, max_dim_2, set_token, all_old_tokens):
    res = []
    for _ in range(dim_1):
        inner_length = random.randint(1, max_dim_2)
        inner_list = [set_token(all_old_tokens) for _ in range(inner_length)]
        res.append(inner_list)
    return res


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


def generate_hm_with_aux(pre_fill_width, aux_idxs):
    aux_idxs_ = torch.tensor(aux_idxs)
    start_indices = aux_idxs_[:, 0]
    end_indices = aux_idxs_[:, 1]

    col_indices = torch.arange(0, pre_fill_width)
    mask = (col_indices >= start_indices.unsqueeze(1)) & (col_indices < end_indices.unsqueeze(1))
    mask[:, 0] = True
    return mask


class AuxBranch:
    def __init__(self, num, max_pad_len, all_old_tokens):
        self.aux_list = set_random_two_dim_list(num, max_pad_len, copy_from, all_old_tokens)
        self.update_size()
        self.fix_aux_length = True
        self.base_ids_ = []
        self.attn_mask = None

    def get_all_aux(self):
        res = []
        for a in self.aux_list:
            res += a
        return res

    def update_size(self):
        self.aux_sizes = [len(a) for a in self.aux_list]

    def ini_from_corpus(self, all_old_tokens, corpus_cache):
        candidate_aux, _ = corpus_cache.retrieve_N_gram_for_aux(all_old_tokens[:-5], len(self.aux_list),
                                                                aux_sizes=self.aux_sizes)

        for i, ca in enumerate(candidate_aux):
            self.aux_list[i] = list(ca)

        self.aux_sizes = [len(a) for a in self.aux_list]
        self.update_size()
        self.get_hm_mask()
        self.get_aux_attn_mask()

    def shorter_first_several(self, branch_len):
        for i, _ in enumerate(self.aux_list):
            if i <= branch_len:
                self.aux_list[i] = [self.aux_list[i][0]]
        self.update_size()
        self.get_hm_mask()
        self.get_aux_attn_mask()

    def get_bias_(self, update_flag):
        if self.fix_aux_length:
            if not self.base_ids_ or update_flag:
                self.base_ids_ = []
                for a in self.aux_list:
                    self.base_ids_ += list(range(1, 1 + len(a)))
                return self.base_ids_
            else:
                return self.base_ids_
        else:
            self.base_ids_ = []
            for a in self.aux_list:
                self.base_ids_ += list(range(1, 1 + len(a)))
            return self.base_ids_

    def get_pos(self, base_pos, update_flag):
        bias = self.get_bias_(update_flag)
        return [a + base_pos for a in bias]

    def shorter_aux(self):
        self.aux_list.pop(0)
        self.aux_sizes.pop(0)
        self.get_aux_attn_mask()
        self.get_hm_mask()
        self.get_bias_(True)

    def update_aux(self, draft_res):
        if self.fix_aux_length:
            for i, a in enumerate(self.aux_list):
                self.aux_list[i].append(draft_res[i])
                self.aux_list[i].pop(0)
            # self.aux_sizes = [len(a) for a in self.aux_list]
            assert self.aux_sizes == [len(a) for a in self.aux_list]
        else:
            for i, a in enumerate(self.aux_list):
                self.aux_list[i].pop(0)
            self.aux_sizes = [len(a) for a in self.aux_list]
            self.get_aux_attn_mask()
            self.get_hm_mask()
        assert len(self.aux_sizes) == len(self.aux_list)

    def get_last_level(self):
        return [a[-1] for a in self.aux_list]

    def get_aux_attn_mask(self, dtype=torch.float16):
        l = sum(self.aux_sizes)
        self.attn_mask = torch.full((l, l), torch.finfo(dtype).min, dtype=dtype)
        pre_idx = 0
        for a in self.aux_sizes:
            self.attn_mask = set_lower_triangular_true_efficient(self.attn_mask, pre_idx, pre_idx,
                                                                 pre_idx + a, pre_idx + a)
            pre_idx += a

    def get_hm_mask(self):
        aux_idxs = [[0, 1]]
        for a in self.aux_sizes:
            aux_idxs.append([aux_idxs[-1][1], aux_idxs[-1][1] + a])
        self.hm_mask = generate_hm_with_aux(sum(self.aux_sizes) + 1, aux_idxs)


class DraftBranch:
    def __init__(self, all_old_tokens, branch_num, branch_len, max_pad_len, ini_method, always_fwd_one):
        max_pad_len = 2 if max_pad_len is None else max_pad_len
        self.branches = [None for _ in range(branch_len - 1)]
        self.branch_num = branch_num
        self.branch_len = branch_len
        self.always_fwd_one = always_fwd_one
        if ini_method in ['RANDOM_WITH_AUX']:
            self.ini_method = copy_from
        self.aux_branch = AuxBranch(branch_num + branch_len - 3, max_pad_len, all_old_tokens)
        self.aux_branch.shorter_first_several(branch_len - 3)
        self.branches[0] = self.aux_branch.get_last_level()
        self.filled_depth = sum([1 for l in self.branches if l is not None]) - 1
        self.widths = []
        self.fix_aux_length = True
        self.attn_mask = None
        self.dtype = torch.float16

    def prepare_ids(self, lst_id):

        all_past = []
        attn_size = 0
        self.widths = []
        random_prefix_len = []
        ids_ = []
        assert len(self.aux_branch.aux_list) == len(self.branches[0])

        for ll in range(self.filled_depth + 1):
            assert self.branches[ll] is not None
            if ll == 0:
                all_past += self.aux_branch.get_all_aux()
                random_prefix_len = self.aux_branch.aux_sizes[:]
                ids_ = self.aux_branch.get_pos(lst_id, False)
                assert random_prefix_len == self.aux_branch.aux_sizes
                self.widths.append(sum(random_prefix_len))
                attn_size += sum(random_prefix_len)
            else:
                all_past += self.branches[ll]
                self.widths.append(len(self.branches[ll]))
                attn_size += len(self.branches[ll])
                if len(random_prefix_len) < len(self.branches[ll]):
                    random_prefix_len.insert(0, 0)
                ids_ += [lst_id + random_prefix_len[i] + ll for i in range(len(self.branches[ll]))]

        return ids_, all_past, attn_size

    def update_attn_mask(self):
        assert self.filled_depth >= 1
        basic_width = self.widths[1]
        all_l_size = sum(self.widths[1:])
        self.attn_mask = torch.full((all_l_size, all_l_size),
                                    fill_value=torch.finfo(self.dtype).min, dtype=self.dtype)

        for l in range(0, self.filled_depth):
            self.attn_mask[l * basic_width:all_l_size, 0:all_l_size - basic_width * l].fill_diagonal_(0)

    def update_draft_branches_inter_logits(self, inp_logits, verify_results, new_results, cdt_contents, branch_hits,
                                           corpus_hits):
        branch_cdt_tokens, branch_cdt_size, corpus_cdt_tokens, corpus_cdt_size = cdt_contents
        max_hit, hit_point, pre_len = 0, 0, 0

        if self.filled_depth == 0:
            self.branches[0] = self.branches[0][1:]
            self.aux_branch.shorter_aux()
            self.branches[1] = inp_logits
            self.widths.append(len(inp_logits))
            self.filled_depth += 1

            self.attn_mask = self.aux_branch.attn_mask
            self.update_attn_mask()

        elif self.filled_depth < self.branch_len - 2:
            self.aux_branch.shorter_aux()

            for bl in range(self.filled_depth + 1):
                self.branches[bl] = self.branches[bl][1:]  # 所有结果左移一位

            self.branches[self.filled_depth + 1] = inp_logits[1:]
            self.widths.append(len(inp_logits[1:]))
            self.filled_depth += 1
            self.widths = []
            self.widths.append(sum(self.aux_branch.aux_sizes))
            for l in range(1, self.filled_depth + 1):
                self.widths.append(len(self.branches[l]))

            self.update_attn_mask()
        else:

            if len(branch_cdt_tokens) or len(corpus_cdt_tokens):
                max_hit, hit_point, pre_len = self.verify_candidates(verify_results, cdt_contents, branch_hits,
                                                                     corpus_hits)

            if self.always_fwd_one:  # update past tokens
                self.aux_branch.update_aux(self.branches[1][1:])
                self.branches[0] = self.branches[1][1:]
                for l in range(1, self.branch_len - 2):
                    self.branches[l] = self.branches[l + 1][:]

                self.branches[self.branch_len - 2] = new_results
            else:
                self.branches[0] = self.branches[1][1 + max_hit:]
                for l in range(1, self.branch_len - 2):
                    self.branches[l] = self.branches[l + 1][max_hit:]

                self.branches[self.branch_len - 2] = new_results[max_hit:]
            assert self.aux_branch.aux_sizes == [len(a) for a in self.aux_branch.aux_list]

        candidate_len = len(branch_cdt_tokens) + len(corpus_cdt_tokens)
        if pre_len == 0:
            g_size = branch_cdt_size
            hits = branch_hits
        else:
            g_size = corpus_cdt_size
            hits = corpus_hits
        return max_hit, hit_point, pre_len, hits, g_size, candidate_len

    def update_draft_branches_inter_sample(self, outputs, cdt_content, input_ids, logits_warper):  # 暂时先不更改这一部分
        branch_cdt_tokens, branch_cdt_size, corpus_cdt_tokens, corpus_cdt_size = cdt_content
        next_token_scores = logits_warper(input_ids, outputs.out_logits)
        max_hit, hit_point, pre_len = 0, 0, 0
        aux_sizes = [len(a) for a in self.aux_list]

        if self.filled_depth == 0:
            probs = torch.nn.functional.softmax(next_token_scores, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            hits = [next_tokens.item()]
            self.branches[0] = self.branches[0][1:]
            self.aux_list = self.aux_list[1:]
            self.branches[1] = torch.argmax(outputs.inp_logits, dim=-1)[0].tolist()
            # shortening the aux_list
            for i in range(len(self.aux_list)):
                if len(self.aux_list[i]) + self.filled_depth > self.branch_len - 2:
                    if len(self.aux_list[i]) > 1:
                        self.aux_list[i] = self.aux_list[i][1:]
            self.filled_depth += 1

        elif self.filled_depth < self.branch_len - 2:
            self.aux_list = self.aux_list[1:]
            probs = torch.nn.functional.softmax(next_token_scores, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            hits = [next_tokens.item()]

            for l in range(self.filled_depth + 1):
                self.branches[l] = self.branches[l][1:]
            self.branches[self.filled_depth + 1] = torch.argmax(outputs.inp_logits, dim=-1)[0].tolist()[1:]

            # shortening the aux_list
            for i in range(len(self.aux_list)):
                if len(self.aux_list[i]) + self.filled_depth >= self.branch_len - 2:
                    if len(self.aux_list[i]) > 1:
                        self.aux_list[i] = self.aux_list[i][1:]
            self.filled_depth += 1

        else:
            if len(branch_cdt_tokens) or len(corpus_cdt_tokens):

                hits, max_hit, hit_point, pre_len = self.verify_candidates_with_sample(outputs, cdt_content, input_ids,
                                                                                       next_token_scores, logits_warper)
                next_tokens = torch.tensor([hits[0]], device=next_token_scores.device)
            else:
                probs_next = torch.nn.functional.softmax(next_token_scores, dim=-1)
                next_tokens = torch.multinomial(probs_next, num_samples=1).squeeze(1)
                hits = [next_tokens.item()]
            new_results = torch.argmax(outputs.inp_logits, dim=-1)[0].tolist()
            assert len(self.branches[self.branch_len - 2]) == self.branch_num and len(new_results) == self.branch_num

            if self.always_fwd_one:
                self.branches[0] = self.branches[1][1:]
                for l in range(1, self.branch_len - 2):
                    self.branches[l] = self.branches[l + 1][:]

                self.branches[self.branch_len - 2] = new_results

            else:
                self.branches[0] = self.branches[1][1 + max_hit:]
                for l in range(1, self.branch_len - 2):
                    self.branches[l] = self.branches[l + 1][max_hit:]

                self.branches[self.branch_len - 2] = new_results[max_hit:]

            assert sum(aux_sizes) == len(self.aux_list)

        if pre_len == 0:
            g_size = branch_cdt_size
        else:
            g_size = corpus_cdt_size

        return next_tokens, max_hit, hit_point, pre_len, hits, g_size

    @staticmethod
    def insert_values(lst, interval, values):
        # 将列表按照固定间隔划分为二维列表
        sublists = [lst[i:i + interval] for i in range(0, len(lst), interval)]

        # 在每个子列表后面添加值
        for sublist in sublists:
            sublist.extend(values)

        # 将二维列表转换回一维列表
        new_lst = [item for sublist in sublists for item in sublist]

        return new_lst

    def verify_candidates_with_sample(self, outputs, cdt_contents,
                                      input_ids, next_token_scores, logits_warper):
        branch_cdt_tokens, branch_cdt_size, corpus_cdt_tokens, corpus_cdt_size = cdt_contents
        probs_next = torch.nn.functional.softmax(next_token_scores, dim=-1)[0]
        hits = []
        verify_logits = logits_warper(input_ids, outputs.verify_logits[0])
        verify_probs = torch.nn.functional.softmax(verify_logits, dim=-1)
        next_tokens = None
        pad_token_id = 2
        pivot = len(branch_cdt_tokens) // branch_cdt_size
        if branch_cdt_size > corpus_cdt_size:
            pad_len = branch_cdt_size - corpus_cdt_size
            pad_list = [pad_token_id] * pad_len
            draft_tokens = branch_cdt_tokens + self.insert_values(corpus_cdt_tokens, corpus_cdt_size, pad_list)
        else:
            pad_len = corpus_cdt_size - branch_cdt_size
            pad_list = [pad_token_id] * pad_len
            draft_tokens = self.insert_values(branch_cdt_tokens, branch_cdt_size, pad_list) + corpus_cdt_tokens
        if branch_cdt_tokens and corpus_cdt_tokens:
            draft_size = max(branch_cdt_size, corpus_cdt_size)
        elif branch_cdt_tokens:
            draft_size = branch_cdt_size
        elif corpus_cdt_tokens:
            draft_size = corpus_cdt_size
        else:
            raise RuntimeError('No draft tokens')

        draft_indices = list(range(len(draft_tokens) // draft_size))
        hit_point = -1

        for idx_in_ngram in range(draft_size):  # iterate over the n-gram
            g_idx = 0
            is_accept = False

            while g_idx < len(draft_indices):
                draft_idx = draft_indices[g_idx]
                draft_offset = draft_idx * draft_size
                draft = draft_tokens[draft_offset + idx_in_ngram]
                prob_accept = min(1, probs_next[draft].item())
                sample_prob = random.random()

                if sample_prob < prob_accept:
                    hits.append(draft)
                    is_accept = True
                    max_hit_idx = draft_idx
                    new_draft_indices = []
                    for draft_idx_n in draft_indices:
                        draft_offset_n = draft_idx_n * draft_size
                        new_draft = draft_tokens[draft_offset_n + idx_in_ngram]
                        if new_draft == draft:
                            new_draft_indices.append(draft_idx_n)
                    draft_indices = new_draft_indices
                    break
                else:
                    probs_next[draft] = 0
                    probs_next = probs_next / probs_next.sum()
                    g_idx += 1
            if is_accept:
                probs_next = verify_probs[draft_offset + idx_in_ngram]
                hit_point = draft_indices[0]
                continue
            else:
                new_token_gen = torch.multinomial(probs_next, num_samples=1).item()
                hits.append(new_token_gen)
                break

        if hit_point < pivot:
            pre_len = 0
            if len(hits) > branch_cdt_size:
                hits = hits[:branch_cdt_size]
        else:
            pre_len = 1
            hit_point -= pivot
            if len(hits) > corpus_cdt_size:
                hits = hits[:corpus_cdt_size]

        max_hit = len(hits) - 1

        return hits, max_hit, hit_point, pre_len

    @staticmethod
    def verify_candidates_(verify_results, next_token, cdt_tokens, cdt_size, hits):
        hit_point = 0
        max_hit = 0
        for eg in range(len(verify_results) // cdt_size):
            egx = eg * cdt_size
            correct = [next_token] + verify_results[egx:egx + cdt_size]  # verification result
            cdt = cdt_tokens[egx:egx + cdt_size]  # draft sequence
            gg = 0
            for gg in range(len(cdt)):
                if cdt[gg] != correct[gg]:
                    break
            if gg > max_hit:
                max_hit = gg
                hit_point = eg
                hits[:max_hit + 1] = correct[:max_hit + 1]

        return max_hit, hit_point

    def verify_candidates(self, verify_results, cdt_contents, branch_hits, corpus_hits):
        branch_cdt_tokens, branch_cdt_size, corpus_cdt_tokens, corpus_cdt_size = cdt_contents
        next_token = branch_hits[0]

        assert len(branch_cdt_tokens) + len(corpus_cdt_tokens) == len(verify_results)
        if len(branch_cdt_tokens) > 0:
            branch_cdt_results = verify_results[:len(branch_cdt_tokens)]
        else:
            branch_cdt_results = []
        if len(corpus_cdt_tokens) > 0:
            corpus_cdt_results = verify_results[-len(corpus_cdt_tokens):]
        else:
            corpus_cdt_results = []

        branch_max_hit, branch_hit_point = self.verify_candidates_(branch_cdt_results, next_token, branch_cdt_tokens,
                                                                   branch_cdt_size, branch_hits)

        corpus_max_hit, corpus_hit_point = self.verify_candidates_(corpus_cdt_results, next_token, corpus_cdt_tokens,
                                                                   corpus_cdt_size, corpus_hits)
        if branch_max_hit >= corpus_max_hit or len(corpus_cdt_tokens) == 0:
            pre_len = 0
            max_hit = branch_max_hit
            hit_point = branch_hit_point
        else:
            pre_len = 1
            max_hit = corpus_max_hit
            hit_point = corpus_hit_point

        return max_hit, hit_point, pre_len
