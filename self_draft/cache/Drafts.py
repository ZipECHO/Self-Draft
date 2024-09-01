from .context_cache import *
from .CorpusCache import *

def retrieve_drafts(all_old_tokens,use_context,use_corpus, context_cache:ContextCache, corpus_cache:CorpusCache):

    branch_cdt_tokens, corpus_cdt_tokens = [], []
    branch_cdt_tup, corpus_cdt_tup = [], []
    branch_cdt_count, corpus_cdt_count = 0, 0

    if use_context:
        # Retrieve the context drafts
        branch_cdt_tokens,branch_cdt_tup,branch_cdt_count = context_cache.retrieve(all_old_tokens)

    if use_corpus:
        # Retrieve the corpus drafts
        _ = context_cache.max_val_len - branch_cdt_count
        if _ == 0:
            corpus_cdt_max = 2
        else:
            corpus_cdt_max = min(3, _)

        (corpus_cdt_tokens, corpus_cdt_tup,
         corpus_cdt_count, overlap_c_) = corpus_cache.retrieve_from_corpus(all_old_tokens, corpus_cdt_max,
                                                                            pre_cdts=branch_cdt_tup)


    return branch_cdt_tokens, branch_cdt_tup, branch_cdt_count, corpus_cdt_tokens, corpus_cdt_tup, corpus_cdt_count
