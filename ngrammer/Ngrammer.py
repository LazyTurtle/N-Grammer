from math import log


def log2prob(log_value):
    from math import exp
    return exp(log_value) if log_value else 0.0


def perplexity(tree, sentence):
    n = len(sentence)
    probability = tree.score(sentence)
    perp = (1 / probability) ** (1 / n)
    return perp


class Node:
    """
    Encapsulates all the necessary information for calculating the probabilities
    """

    def __init__(self, text=None):
        self.string = str(text)
        self.count = 0
        self.children = dict()

    def __set__(self, instance, value):
        self.instance = value

    def __get__(self, instance, owner):
        return self.instance

    def __str__(self):
        return self.string

    def __repr__(self):
        return "[string: {}, count: {}, children: {}]".format(self.string, self.count, self.children)


class PrefixTree:

    def __init__(self, n=None):
        self.root = Node(".")
        self.error = Node("Error")
        self.n = n
        self.interpolation = None

    def add_ngram(self, sequence):
        node = self.root
        node.count += 1
        for word in sequence:
            node.children[word] = node.children.setdefault(word, Node(word))
            node = node.children[word]
            node.count += 1

    def get(self, sequence):
        node = self.root
        sequence = sequence[-self.n:]
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

    def vocabulary(self, n=None):
        n = self.n if n is None else n
        v = set()
        for ngram in self.traverse():
            v.add("_".join(ngram[:n]))
        v = sorted(list(v))
        return v

    def score(self, sentence):
        from math import prod
        probabilities = list()
        for ngram in PrefixTree.extract_ngrams(sentence, self.n):
            probabilities.append(self.get(ngram).probability)
        return sum(probabilities) if self.logs else prod(probabilities)

    def __deleted_interpolation__(self):
        w = [0] * self.n
        for ngram in self.traverse():
            # current ngram count
            v = self.get(ngram).count
            # (n)-gram counts
            n = [self.get(ngram[0:i + 1]).count for i in range(len(ngram))]
            # (n-1)-gram counts -- parent node
            p = [self.get(ngram[0:i]).count for i in range(len(ngram))]
            # -1 from both counts & normalize
            d = [float((n[i] - 1) / (p[i] - 1)) if (p[i] - 1 > 0) else 0.0 for i in range(len(n))]
            # increment weight of the max by raw ngram count
            k = d.index(max(d))
            w[k] += v
        self.interpolation = [float(v) / sum(w) for v in w]

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
        tree = PrefixTree(n) if not tree else tree

        for sequence in corpus:
            for ngram in PrefixTree.extract_ngrams(sequence, tree.n):
                tree.add_ngram(ngram)
        return tree

    @staticmethod
    def compute_probabilities(tree, logs=False, smoothing=False):
        tree.logs = logs
        tree.smoothing = smoothing

        v = len(tree.vocabulary(tree.n - 1)) if tree.smoothing else 0
        a = 1 if tree.smoothing else 0

        tree.error.probability = log(a / v) if (smoothing and logs) else 0.0

        for ngram in tree.traverse():
            n = tree.get(ngram)
            p = tree.get(ngram[:-1])
            prob = (n.count + a) / (p.count + v)
            n.probability = log(prob) if logs else prob

        return tree

    def __set__(self, instance, value):
        self.instance = value

    def __get__(self, instance, owner):
        return self.instance


class MultiNgramPrefixTree:
    """
    A class used almost only to encapsulate the use of multiple prefix trees
    used to smooth out prediction
    """

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

    def get(self, sequence, n=None):
        n = self.n if n is None else n
        self.trees[n].get(sequence)

    def traverse(self, n=None, node=None, sequence=None, size=None):
        n = len(self.trees) if n is None else n
        self.trees[n].traverse(node, sequence, size)

    def vocabulary(self, n=None):
        n = self.n if n is None else n
        return self.trees[n].vocabulary(n)

    def predict_sentence(self, sentence, n=None, use_interpolation=False):
        n = self.n if n is None else n
        if not use_interpolation:
            return self.trees[n].score(sentence)

        if not self.trees[n].interpolation:
            self.trees[n].__deleted_interpolation__()

        from math import prod
        coefficients = self.trees[n].interpolation
        probabilities = list()

        for ngram in PrefixTree.extract_ngrams(sentence, n):
            multigram_probabilities = list()

            for i in range(1, n + 1):
                node = self.trees[i].get(ngram)
                multigram_probabilities.append(node.probability)
            for i in range(len(multigram_probabilities)):
                multigram_probabilities[i] *= coefficients[i]

            interpolated_ngram_probability = sum(multigram_probabilities)

            probabilities.append(interpolated_ngram_probability)
        return sum(probabilities) if self.logs else prod(probabilities)

    def set_interpolations(self):
        for n, tree in self.trees.items():
            tree.__deleted_interpolation__()

    @staticmethod
    def store_ngrams(corpus, n, tree=None):
        assert n > 0, "n must be greater than 0, given: {}".format(n)
        tree = MultiNgramPrefixTree(n) if not tree else tree
        for sequence in corpus:
            for i in range(1, n + 1):
                for ngram in PrefixTree.extract_ngrams(sequence, i):
                    tree.add_ngram(ngram)
        return tree

    @staticmethod
    def compute_probabilities(multi_tree, logs=False, smoothing=False):
        multi_tree.logs = logs
        multi_tree.smoothing = smoothing

        for n, tree in multi_tree.trees.items():
            PrefixTree.compute_probabilities(tree, logs, smoothing)
        return multi_tree
