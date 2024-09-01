from .cache import ContextCache, CorpusCache

class Drafts:
    def __init__(self):
        self.draft_tokens = []
        self.gram_len = 4
        self.draft_tup = []


class ALL_Drafts:
    def __init__(self):
        self.drafts = []
        self.context_draft = []
        self.corpus_draft = []
        self.context_gram = 4
        self.corpus_gram = 4
        self.context_draft_tup = []
        self.corpus_draft_tup = []

