class Word:
    """
    Encapsulates all the necessary information for calculating the probabilities
    """

    def __init__(self, text=None):
        self.string = str(text)
        self.count = 0
        self.children = dict()

    def __str__(self):
        return self.string

    def __repr__(self):
        return "[string: {}, count: {}, children: {}]".format(self.string, self.count, self.children)


class PrefixTree:

    def __init__(self, n=None):
        self.root = Word(".")
        self.error = Word()
        self.n = n

    def add_ngram(self, sequence):
        node = self.root
        node.count += 1
        for word in sequence:
            node.children[word] = node.children.setdefault(word, Word(word))
            node = node.children[word]
            node.count += 1

    def get_word(self, sequence):
        node = self.root
        for word in sequence:
            node = node.children.get(word, self.error)
        return node

    def traverse(self, node=None, sequence=None, size=None):
        sequence = sequence if sequence else []
        node = self.root if not node else node

        if not node.children:
            yield sequence

        if size:
            if len(sequence) == size:
                yield sequence

        for word, n in node.children.items():
            sequence.append(word)
            yield from self.traverse(n, sequence, size)
            sequence.pop()

    def vocabulary(self, n=1):
        v = set()
        for ngram in self.traverse():
            v.add("_".join(ngram[:n]))
        v = sorted(list(v))
        return v

    @staticmethod
    def extract_ngrams(sequence, n):
        assert n > 0, "n must be greater than 0, given: {}".format(n)
        ngrams = list()
        for i in range(n, len(sequence) + 1):
            ngrams.append(sequence[i - n:i])
        return ngrams

    @staticmethod
    def store_ngrams(corpus, n, tree=None):
        assert n > 0, "n must be greater than 0, given: {}".format(n)
        tree = PrefixTree() if not tree else tree
        tree.n = n
        for sequence in corpus:
            for ngram in PrefixTree.extract_ngrams(sequence, n):
                tree.add_ngram(ngram)
        return tree

    @staticmethod
    def compute_probabilities(tree, logs=True, smoothing=False):
        from math import log
        tree.logs = logs
        tree.smoothing = smoothing

        v = len(tree.vocabulary(tree.n)) if tree.smoothing else 0
        a = 1 if tree.smoothing else 0

        tree.error.probability = log(a / v) if (smoothing and logs) else 0.0

        for ngram in tree.traverse():
            n = tree.get(ngram)
            p = tree.get(ngram[:-1])  # get parent node
            prob = (n.count + a) / (p.count + v)
            n.probability = log(prob) if logs else prob
        return tree
