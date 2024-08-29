class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False


class Trie:
    def __init__(self):
        self.root = TrieNode()
        self.node_count = 1  # 根节点计数

    def insert(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
                self.node_count += 1  # 新节点计数
            node = node.children[char]
        node.is_end_of_word = True

    def search(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                return False
            node = node.children[char]
        return node.is_end_of_word

    def get_node_count(self):
        return self.node_count

def compress_candidate_tokens(candidates):
    trie = Trie()
    for seq in candidates:
        trie.insert(seq)
    return trie
