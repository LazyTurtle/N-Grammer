def log2prob(log_value):
    from math import exp
    return exp(log_value) if log_value else 0.0


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
        self.probabilities = False

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

    def predict_ngram(self, sequence):
        probability = self.get_word(sequence[-self.n:]).probability
        return probability

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
    def compute_probabilities(tree, logs=False, smoothing=False):
        from math import log
        tree.logs = logs
        tree.smoothing = smoothing

        v = len(tree.vocabulary(tree.n)) if tree.smoothing else 0
        a = 1 if tree.smoothing else 0

        tree.error.probability = log(a / v) if (smoothing and logs) else 0.0

        for ngram in tree.traverse():
            n = tree.get(ngram)
            p = tree.get(ngram[:-1])
            prob = (n.count + a) / (p.count + v)
            n.probability = log(prob) if logs else prob

        tree.probabilities = True
        return tree


class MultiNgramPrefixTree:
    def __init__(self, n):
        assert n > 0, "n must be greater than 0, given: {}".format(n)
        self.n = n
        self.trees = dict()
        for i in range(1, n + 1):
            self.trees[i] = PrefixTree(i)

        self.coefficients = None

    def add_ngram(self, sequence, n=None):
        n = len(sequence) if n is None else n
        self.trees[n].add_ngram(sequence)

    def get_word(self, sequence, n=None):
        n = len(sequence) if n is None else n
        self.trees[n].get_word(sequence)

    def traverse(self, n=None, node=None, sequence=None, size=None):
        n = len(self.trees) if n is None else n
        self.trees[n].traverse(node, sequence, size)

    def vocabulary(self, n=None):
        n = 1 if n is None else n
        v = set()
        for ngram in self.trees[n].traverse():
            v.add("_".join(ngram[:n]))
        v = sorted(list(v))
        return v

    def predict_ngram(self, sequence):
        if self.coefficients is None:
            print("There are no coefficients for interpolation, reverting to greater n prediction")
            n = min(len(sequence), self.n)
            return self.trees[n].predict_ngram(sequence)

        total_coefficients = sum([value for n, value in self.coefficients.items()])
        assert total_coefficients > 1, "The sum of the coefficient is greater than 1"

        if len(self.coefficients) != len(self.trees):
            print("The number of coefficients ({}) and trees ({}) do not match".format(len(self.coefficients), len(self.trees)))

        probability = 0.
        for i in range(1, len(self.trees) + 1):
            probability += self.coefficients[i] * self.trees[i].predict_ngram(sequence)

        return probability

    @staticmethod
    def store_ngrams(corpus, n, tree=None):
        assert n > 0, "n must be greater than 0, given: {}".format(n)
        tree = MultiNgramPrefixTree(n) if not tree else tree
        for sequence in corpus:
            for i in range(1, n+1):
                for ngram in PrefixTree.extract_ngrams(sequence, i):
                    tree.add_ngram(ngram)
        return tree

    @staticmethod
    def compute_probabilities(multi_tree, logs=False, smoothing=False):
        for n, tree in multi_tree.trees.items():
            PrefixTree.compute_probabilities(tree, logs, smoothing)
        return multi_tree
