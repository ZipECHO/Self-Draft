import os
import pickle
import warnings

import numpy as np
from datasets import load_dataset
from loguru import logger
from tqdm import tqdm
from transformers import AutoTokenizer


class CorpusCache:
    def __init__(self, N, max_key_len=4, tokenizer_path=None, cache_save_path=None, text_file_path=None, rebuild=False,
                 clean=False,search_with_dif_key_len=1):
        self.tokenizer_name = os.path.dirname(tokenizer_path).split('/')[-1]
        self.text_file_path = text_file_path
        if text_file_path is not None:
            if not os.path.isfile(text_file_path):
                self.all_content = self.load_text_from_dir(
                    os.path.join(os.path.dirname(self.text_file_path), 'OANC-GrAF'))

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=False)
        logger.info(f'load tokenizer from {tokenizer_path} done!')
        self.max_key_len = max_key_len
        self.n_gram = N
        self.search_with_dif_key_len = search_with_dif_key_len
        if os.path.isfile(cache_save_path):
            self.n_gram_cache = pickle.load(open(cache_save_path, 'rb'))
        else:
            self.build_cache(text_file_path, cache_save_path)
        if rebuild:
            self.build_cache(text_file_path, cache_save_path)
        if clean:
            self.clean(6)

    def clean(self, k):
        self.clean_cache = {}
        for key_len in range(1, self.max_key_len + 1):
            if key_len not in self.clean_cache:
                self.clean_cache[key_len] = {}
            for key, gram_freq in tqdm(self.n_gram_cache[key_len].items()):
                if key not in self.clean_cache[key_len]:
                    self.clean_cache[key_len][key] = {}
                for gram, freq in gram_freq.items():
                    if freq > k:
                        self.clean_cache[key_len][key][gram] = freq
                if len(self.clean_cache[key_len][key]) == 0:
                    self.clean_cache[key_len].__delitem__(key)
            if len(self.clean_cache[key_len]) == 0:
                self.clean_cache.__delitem__(key_len)
        with open(f'data/OANC/clean-{k}-OANC-tmp-cache.pickle', 'wb') as file:
            pickle.dump(self.clean_cache, file)
            file.close()

    def build_cache(self, text_file_path, cache_save_path):

        self.n_gram_cache = {key_len: None for key_len in range(1, self.max_key_len + 1)}
        for key_len in self.n_gram_cache.keys():
            self.n_gram_cache[key_len] = self.build_ngram_from_corpus(key_len, self.n_gram)

        self.sort_cache_by_freq(cache_save_path)

        with open(cache_save_path, 'wb') as file:
            logger.info(f'saving {self.n_gram} gram cache at: {cache_save_path}')
            pickle.dump(self.n_gram_cache, file)
            logger.info('done!')
            file.close()

    @staticmethod
    def load_mbpp(path):
        return load_dataset(path)['train']['code']

    @staticmethod
    def load_pickle(path):
        with open(path, 'rb') as file:
            res = pickle.load(file)
        return res

    @staticmethod
    def extract_n_gram_from_sentence(all_sentences, key_len, N, tokenizer):
        res = {}
        all_tokens = []
        for sentence in tqdm(all_sentences):
            tokens = tokenizer.encode(sentence)
            all_tokens.extend(tokens)
            for i, _ in enumerate(tokens[:-N - key_len]):
                if key_len == 1:
                    key = tokens[i]
                else:
                    key = tuple(tokens[i:i + key_len])
                n_gram = tuple(tokens[i + key_len:i + key_len + N])
                if key not in res:
                    res[key] = {n_gram: 1}
                elif n_gram not in res[key]:
                    res[key][n_gram] = 1
                else:
                    res[key][n_gram] += 1
        return res, all_tokens

    def build_ngram_from_corpus(self, key_len, N):

        n_gram_cache, all_tokens = self.extract_n_gram_from_sentence(self.all_content, key_len, N, self.tokenizer)
        logger.info(f'total tokens: {len(all_tokens)}')
        return n_gram_cache

    @staticmethod
    def normalization(freqs):
        freq_arr = np.array(list(freqs))
        return freq_arr / freq_arr.sum()

    def retrieval_n_gram(self, prefix, n):
        assert isinstance(prefix, int)
        if prefix not in self.n_gram_cache[1].keys():
            return []
        grams, freqs = list(self.n_gram_cache[1][prefix].keys()), list(self.n_gram_cache[1][prefix].values())
        n = min(len(grams), n)
        chose_index = np.random.choice(range(len(grams)), n, replace=False, p=self.normalization(freqs))

        return [grams[i] for i in chose_index]

    def retrieve_N_gram_with_different_key_length(self, prefix, n, pre_cdt=[]):
        assert isinstance(prefix, list)
        candidates = []
        overlap_c = 0
        prefix_len = len(prefix)
        query_len = min(self.max_key_len, prefix_len)
        if self.max_key_len < prefix_len:
            warnings.warn(f"query length {prefix_len} is larger than cache max key length {self.max_key_len}")

        while query_len > 0 and n > 0:
            if query_len == 1:
                query = prefix[-1]
            else:
                query = tuple(prefix[-query_len:])
            if query in self.n_gram_cache[query_len]:
                grams, freqs = list(self.n_gram_cache[query_len][query].keys()), \
                    list(self.n_gram_cache[query_len][query].values())
                sample_count = min(len(grams), n)
                # assert freqs
                g_ = grams[:sample_count]
                append_count = 0
                for i in range(len(g_)):
                    if g_[i] not in candidates and g_[i] not in pre_cdt:
                        candidates.append(g_[i])
                        append_count += 1
                    if g_[i] in pre_cdt and g_[i] not in candidates:
                        overlap_c += 1

                n = n - append_count

            query_len -= 1
        return candidates, overlap_c

    def retrieve_N_gram_for_aux(self, prefix, n, aux_sizes=[], pre_cdt=[]):
        assert isinstance(prefix, list)
        candidates = []
        overlap_c = 0
        prefix_len = len(prefix)
        k = 0
        query_len = min(self.max_key_len, prefix_len)
        if self.max_key_len < prefix_len:
            warnings.warn(f"query length {prefix_len} is larger than cache max key length {self.max_key_len}")

        while query_len > 0 and n > 0:
            if query_len == 1:
                query = prefix[-1]
            else:
                query = tuple(prefix[-query_len:])
            if query in self.n_gram_cache[query_len]:
                grams, freqs = list(self.n_gram_cache[query_len][query].keys()), \
                    list(self.n_gram_cache[query_len][query].values())
                sample_count = min(len(grams), n)
                # assert freqs
                g_ = grams[:sample_count]
                append_count = 0
                for i in range(len(g_)):
                    if g_[i] not in candidates and g_[i] not in pre_cdt:
                        candidates.append(g_[i][:aux_sizes[k]])
                        k += 1
                        append_count += 1
                    if g_[i] in pre_cdt and g_[i] not in candidates:
                        overlap_c += 1

                n = n - append_count

            query_len -= 1
        return candidates, overlap_c

    def retrieve_with_sample(self, grams, sample_count, freqs):
        chose_idx = np.random.choice(range(len(grams)), sample_count, replace=False, p=self.normalization(freqs))
        chose_grams = [grams[i] for i in chose_idx]
        return chose_grams

    def retrieve_sorted_with_top_k(self, grams, sample_count, freqs):
        assert all(freqs[i] >= freqs[i + 1] for i in range(len(freqs) - 1))
        return grams[:sample_count]
        # return list(range(min(sample_count,len(grams))))

    def retrieve_unsorted_with_top_k(self, grams, sample_count, freqs):
        sorted_freq_grams = sorted(zip(freqs, grams), reverse=True)
        sorted_freq, sort_grams = zip(*sorted_freq_grams)
        return sort_grams[:sample_count]

    def is_increase(self, l):
        for i in range(len(l) - 1):
            if l[i] > l[i + 1]:
                return False
        return True

    def sort_cache_by_freq(self, dump_path, topk=20):
        for key_len in range(1, self.max_key_len + 1):
            for key, grams_freq in tqdm(self.n_gram_cache[key_len].items()):
                self.n_gram_cache[key_len][key] = dict(
                    sorted(self.n_gram_cache[key_len][key].items(), key=lambda item: item[1], reverse=True)[:topk])

    @staticmethod
    def load_text_from_dir(base_folder):
        content = []
        count = 0
        text_save_path = os.path.join(base_folder, 'all_text.pkl')
        print(f'extracting all text data from {base_folder}')
        for root, dirs, files in os.walk(base_folder):
            for file in files:
                if file.endswith(".txt"):
                    file_path = os.path.join(root, file)
                    with open(file_path, 'r') as f:
                        text = f.read().replace('\n', '').strip()
                        content.append(text)

                count += 1
                if count % 2000 == 0:
                    print(f'processed {count} text files and saved at:{text_save_path}')
                    with open(text_save_path, 'wb') as file:
                        pickle.dump(content, file)
        print(f'processed {count} text files and saved at:{text_save_path}')
        with open(text_save_path, 'wb') as file:
            pickle.dump(sorted(content), file)
        return sorted(content)

    # def retrieve_from_corpus(self, lst_token, search_with_dif_key_len, all_old_tokens, max_key_len,
    #                          corpus_cdt_max, pre_cdts=None):
    #     corpus_n_gram = []
    #     corpus_cdt_count = 0
    #     overlap_c_ = 0
    #     corpus_n_gram_ = []
    #     pre_cdts = [] if pre_cdts is None else pre_cdts
    #     if lst_token is not None:
    #         if search_with_dif_key_len == 1:
    #             corpus_n_gram_, overlap_c_ = self.retrieve_N_gram_with_different_key_length(
    #                 all_old_tokens[-max_key_len:], corpus_cdt_max, pre_cdts)
    #         else:
    #             corpus_n_gram_ = self.retrieval_n_gram(lst_token, corpus_cdt_max)
    #         corpus_cdt_count = len(corpus_n_gram_)
    #         for tok in list(corpus_n_gram_):
    #             corpus_n_gram += list(tok)
    #
    #     return corpus_n_gram, corpus_n_gram_, corpus_cdt_count, overlap_c_

    def retrieve_from_corpus(self, all_old_tokens,
                             corpus_cdt_max, pre_cdts=None):
        corpus_n_gram = []
        overlap_c_ = 0
        pre_cdts = [] if pre_cdts is None else pre_cdts
        if self.search_with_dif_key_len == 1:
            corpus_n_gram_, overlap_c_ = self.retrieve_N_gram_with_different_key_length(
                all_old_tokens[-self.max_key_len:], corpus_cdt_max, pre_cdts)
        else:
            corpus_n_gram_ = self.retrieval_n_gram(all_old_tokens[-1], corpus_cdt_max)
        corpus_cdt_count = len(corpus_n_gram_)
        for tok in list(corpus_n_gram_):
            corpus_n_gram += list(tok)

        return corpus_n_gram, corpus_n_gram_, corpus_cdt_count, overlap_c_

# if __name__ == "__main__":
#     tokenizer_path = '/path/to/tokenizer'
#     text_file_path = 'data/OANC/OANC-all-text.pickle'
#     cache_save_path = 'data/OANC/OANC-cache.pickle'
#     corpus_name = 'OANC'
#     N = 4
#     max_key_len = 4
#     logger.info(f'Building {N}-gram cache with different key length from for data: {corpus_name}')
#
#     # corpus_cache = CorpusCache(N, max_key_len, tokenizer_path, cache_save_path, text_file_path, clean=True)
#     corpus_cache = CorpusCache(N,max_key_len,tokenizer_path,cache_save_path,text_file_path,clean=True)
#     # corpus_cache = CorpusCache(corpus_name, N, 4, tokenizer_path,)
#
#     # rebuild = True
#     # corpus_cache = CorpusCache(corpus_path, cache_mode='lookahead_gram', max_key_len=4, N=i, max_lookahead_length=5)
#     # corpus_cache = CorpusCache(corpus_name ='OANC',tokenizer_path=tokenizer_path, N=N, max_key_len=4,rebuild=rebuild)
#
#     logger.info('finished!')
