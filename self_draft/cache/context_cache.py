import numpy as np
from loguru import logger

from ..draft_branch import *


class ContextCache:
    def __init__(self, context_cache=None, max_val_len=None, gram_n=None):
        self.max_val_len = 10 if max_val_len is None else max_val_len
        self.gram_n = 4 if gram_n is None else gram_n
        self.cache = {}
        self.c = 0
        if context_cache is not None:
            assert isinstance(context_cache, ContextCache)
            self.cache = context_cache.cache.copy()

    def cache_insert(self, key, tup):
        # insert a tuple into the cache
        key = self.convert_key(key)
        if key not in self.cache:
            self.cache[key] = []
        if tup in self.cache[key]:  # first column of the past tokens
            self.cache[key].remove(tup)
            self.cache[key].append(tup)
        elif len(self.cache[key]) < self.max_val_len:
            self.cache[key].append(tup)
        else:
            assert len(self.cache[key]) == self.max_val_len
            self.cache[key] = self.cache[key][1:] + [tup]

    def update_cache(self, other_cache):
        if other_cache is None:
            return
        for key, val in other_cache.items():
            for tup in val:
                self.cache_insert(key, tup)

    def get_cache_from_prompt(self, all_old_tokens, tuple_size, key_len=1):
        # get the cache from the prompt
        for i in range(len(all_old_tokens) - tuple_size - key_len):
            t_ = all_old_tokens[i:i + key_len]
            if len(t_) == 1:
                t = t_[0]
            else:
                t = tuple(t_)
            self.cache_insert(t, tuple(all_old_tokens[i + key_len:i + key_len + tuple_size]))

    def retrieve(self, all_old_tokens, max_n=None, pre_cdts=None):
        pre_cdts = [] if pre_cdts is None else pre_cdts
        max_n = self.max_val_len if max_n is None else max_n
        cdt_tokens, cdt_tokens_tup = [], []
        key = all_old_tokens[-1:]
        cdt_tokens_tup_, cdt_count_ = self.retrieve_by_key(key, max_n)
        for tok in cdt_tokens_tup_:
            if tok not in cdt_tokens_tup and tok not in pre_cdts:
                cdt_tokens_tup.append(tok)
                cdt_tokens += list(tok)

        max_n -= len(cdt_tokens_tup_)
        return cdt_tokens, cdt_tokens_tup, len(cdt_tokens_tup)

    def retrieve_by_key(self, key_, max_n):
        cdt_count = 0
        cdt_tokens_ = []

        if len(key_) == 1:
            key = key_[0]
        else:
            key = tuple(key_)

        if (key in self.cache) and max_n > 0:
            cdt_tokens_ = self.cache[key][-max_n:]
            cdt_count = len(cdt_tokens_)

        return cdt_tokens_, cdt_count

    def update_cache(self, all_old_tokens, next_token, draft_branches: DraftBranch, draft_results):

        gram_len = min(self.gram_n, draft_branches.branch_len - 1)
        branch_len_size = [len(p) for p in draft_branches.branches]
        assert sum(draft_branches.aux_branch.aux_sizes) + sum(branch_len_size[1:]) == len(draft_results)

        # make the aux branch and draft branch have the same length
        max_aux_size = max(draft_branches.aux_branch.aux_sizes)
        p = 0
        new_aux_list = []
        new_draft_results = []
        for i in range(len(draft_branches.aux_branch.aux_sizes)):
            aux_tokens = draft_branches.aux_branch.aux_list[i]
            if len(aux_tokens) < max_aux_size:
                new_aux_list.append([-1] * (max_aux_size - len(aux_tokens)) + aux_tokens)
                new_draft_results.append([-1] * (max_aux_size - len(aux_tokens)) + draft_results[p:p + len(aux_tokens)])
            else:
                new_aux_list.append(aux_tokens)
                new_draft_results.append(draft_results[p:p + len(aux_tokens)])
            p += len(aux_tokens)

        draft_results = self.slice_list(draft_results, branch_len_size, draft_branches.aux_branch.aux_sizes)
        assert draft_branches.branch_len > gram_len
        pt_arr = np.concatenate((np.array(new_aux_list).T, np.array([p[1:] for p in draft_branches.branches[1:]])),
                                axis=0)
        draft_arr = np.concatenate((np.array(new_draft_results).T, np.array([d[1:] for d in draft_results[1:]])),
                                   axis=0)

        for l in range(0, draft_branches.branch_len - gram_len + max_aux_size - 1):
            ks = pt_arr[l, :].tolist()
            vals = [tuple(col) for col in
                    np.concatenate((pt_arr[l + 1:l + gram_len, :], draft_arr[l + gram_len - 1:l + gram_len, :])).T]
            assert len(vals[:][-1]) == gram_len

            for i, k in enumerate(ks):
                if k == -1:
                    pass
                else:
                    self.cache_insert(k, vals[i])

        for i in range(max(0,self.c-gram_len-1),len(all_old_tokens)):
            k = all_old_tokens[i]
            t1 = tuple(all_old_tokens[i + 1:i+1+gram_len])
            t2,t3 = (),()

            if len(t1) < gram_len:
                t2 = tuple(draft_branches.branches[ll][0] for ll in range(1, gram_len - len(t1)))
                t3 = (draft_results[gram_len-len(t1) - 1][0],)
            tup = t1 + t2 + t3
            assert len(tup) == gram_len
            self.cache_insert(k, tup)
        self.c = len(all_old_tokens)

        for l in range(1, draft_branches.branch_len - gram_len):
            k = draft_branches.branches[l][0]
            val = tuple(draft_branches.branches[ll][0] for ll in range(l + 1, l + gram_len)) + (
                draft_results[gram_len + l - 1][0],)
            assert len(val) == gram_len
            self.cache_insert(k, val)

        pre_tokens = all_old_tokens[-1]
        tup = tuple(draft_branches.branches[ll][0] for ll in range(1, gram_len)) + (draft_results[gram_len - 1][0],)
        assert len(tup) == gram_len
        self.cache_insert(pre_tokens, tup)

        k = all_old_tokens[-gram_len]
        val = tuple(all_old_tokens[-gram_len + 1:]) + (next_token,)
        assert len(val) == gram_len
        self.cache_insert(k, val)

    def reduce_cache(self, keep_num):
        if keep_num is None:
            return
        else:
            res = {}
            if keep_num is not None:
                if keep_num == 0:
                    self.cache = {}
                    return
                i = -keep_num
            else:
                i = keep_num
            for key, val in self.cache.items():
                res[key] = val[i:]
            self.cache = res.copy()

    def flush_cache(self):
        self.cache = {}

    def stat_info(self):
        all_len = 0
        max_len = 0
        for key, val in self.cache.items():
            all_len += len(val)
            if max_len < len(val):
                max_len = len(val)

        stat_str = f'average length:{all_len / len(self.cache)} max length:{max_len}'
        logger.success(stat_str)

    @staticmethod
    def convert_key(key):
        # convert the key to a hashable type
        if isinstance(key, int):
            key = key
        elif isinstance(key, list):
            key = tuple(key)
            if len(key) == 1:
                key = key[0]
            else:
                key = key
        elif isinstance(key, tuple):
            if len(key) == 1:
                key = key[0]
                key = key
        elif isinstance(key, torch.Tensor):
            key = key.item()
        else:
            raise RuntimeError(f'Unsupported data type: {type(key)}')
        return key

    @staticmethod
    def slice_list(l, lengths, aux_sizes=[]):
        res = []
        p_size = sum(aux_sizes)
        for i in range(len(lengths)):
            if i == 0:
                t = []
                for j, aux_s in enumerate(aux_sizes):
                    t.append(l[sum(aux_sizes[:j + 1]) - 1])
                res.append(t)

            else:
                res.append(l[p_size + sum(lengths[1:i]):p_size + sum(lengths[1:i + 1])])
        for i, r in enumerate(res):
            assert len(r) == lengths[i]

        assert res[-1] == l[p_size + sum(lengths[1:-1]):]
        return res

    @staticmethod
    def extract_ngrams_fast(data, n):
        results = []
        for row in data:
            row_dict = {row[i]: tuple(row[i + 1:i + 1 + n]) for i in range(len(row) - n)}
            results.append(row_dict)
        return results
