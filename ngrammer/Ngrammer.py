import collections
from math import log
import spacy
from collections import defaultdict


def log2prob(log_value):
    from math import exp
    return exp(log_value) if log_value else 0.0


def perplexity(tree, sentence):
    n = len(sentence)
    probability = tree.predict_sentence(sentence)
    perp = (1 / probability) ** (1 / n)
    return perp


class Predictor:

    def store(self, data):
        pass

    def predict(self, data):
        pass

    def train(self):
        pass

    def __set__(self, instance, value):
        self.instance = value

    def __get__(self, instance, owner):
        return self.instance


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


class PrefixTree(Predictor):
    """
    Fairly standard implementation of the prefix tree.
    It takes an n variable that indicates the length if the ngrams it computes
    """

    def __init__(self, n=None):
        self.root = Node(".")
        self.error = Node("[Error]")
        assert n > 0, "n must be greater than 0, given: {}".format(n)
        self.n = n
        self.logs = None
        self.smoothing = None

    def store(self, data):
        assert isinstance(data, collections.Iterable), "Data must be a sequence"
        if isinstance(data, str):
            data = data.strip().split()
        self._add_sentence(data)

    def _add_sentence(self, sentence):
        ngrams = self._extract_ngrams(sentence)
        for ngram in ngrams:
            self._add_ngram(ngram)

    def _extract_ngrams(self, sequence):
        ngrams = list()
        for i in range(self.n, len(sequence) + 1):
            ngrams.append(sequence[i - self.n:i])
        return ngrams

    def _add_ngram(self, sequence):
        node = self.root
        node.count += 1
        for word in sequence:
            word = str(word)
            node.children[word] = node.children.setdefault(word, Node(word))
            node = node.children[word]
            node.count += 1

    def predict(self, data):
        assert isinstance(data, collections.Iterable), "Data must be a sequence"
        if isinstance(data, str):
            data = data.strip().split()
        prediction = self._predict_sentence(data)
        return prediction

    def _predict_sentence(self, sentence):
        from math import prod
        probabilities = list()
        for ngram in self._extract_ngrams(sentence):
            probabilities.append(self._predict_ngram(ngram))
        return sum(probabilities) if self.logs else prod(probabilities)

    def _predict_ngram(self, ngram):
        node = self.get(ngram)
        probability = node.probability
        return probability

    def train(self, logs=False, smoothing=False):
        self.logs = logs
        self.smoothing = smoothing

        v = len(self.vocabulary(self.n - 1)) if self.smoothing else 0
        a = 1 if self.smoothing else 0

        self.error.probability = log(a / v) if (smoothing and logs) else 0.0

        for ngram in self.traverse():
            n = self.get(ngram)
            p = self.get(ngram[:-1])
            prob = (n.count + a) / (p.count + v)
            n.probability = log(prob) if logs else prob

    def get(self, sequence):
        node = self.root
        sequence = sequence[-self.n:]
        for word in sequence:
            word = str(word)
            node = node.children.get(word, self.error)
        return node

    def traverse(self, node=None, sequence=None, size=None):
        sequence = sequence if sequence else []
        node = self.root if not node else node

        if not node.children:
            yield sequence
            return

        if size:
            if len(sequence) == size:
                yield sequence
                return

        for word, n in node.children.items():
            sequence.append(word)
            yield from self.traverse(n, sequence, size)
            sequence.pop()

    def vocabulary(self, n=None):
        n = self.n if n is None else n
        v = set()
        for ngram in self.traverse(size=n):
            v.add("_".join(ngram[:n]))
        v = sorted(list(v))
        return v


class Cache(Predictor):

    def __init__(self, maximum=200, minimum=5):
        from collections import deque
        self.minimum = minimum
        self.maximum = maximum
        self.cache = deque()
        self.active = False

    def store(self, data):
        assert isinstance(data, collections.Iterable), "Data must be a sequence"
        if isinstance(data, str):
            data = data.strip().split()

        removed_words = self._store_sentence(data)
        return removed_words

    def _store_sentence(self, sentence):
        removed_words = list()
        for word in sentence:
            removed_word = self._store_word(word)
            if removed_word is not None:
                removed_words.append(removed_word)
        return removed_words

    def _store_word(self, word):
        removed_word = None
        if len(self.cache) == self.maximum:
            removed_word = self.cache.popleft()
        self.cache.append(word)
        if len(self.cache) >= self.minimum:
            self.active = True
        return removed_word

    def predict(self, data):
        return self._frequency(data)

    def train(self):
        print("No need to train {}.".format(self))

    def _frequency(self, word):
        if self.active:
            return self.cache.count(word) / len(self.cache)
        print("Cache is not active, {} out of {} necessary items are present.".format(len(self.cache), self.minimum))
        return None


class CachedPrefixTree(PrefixTree):
    """
    A version of the prefix tree that uses a series of caches for interpolation as a unigram model.
    If more than one cache is present, it will be used the mean among al caches.
    """

    def __init__(self, n=None, caches_lengths=200):
        super(CachedPrefixTree, self).__init__(n)
        self.caches = self._setup_cache(caches_lengths)
        self.interpolation_coefficients = None

    def _setup_cache(self, caches_lengths):
        caches = list()
        if caches_lengths:
            if isinstance(caches_lengths, int):
                caches.append(Cache(caches_lengths))
            elif isinstance(caches_lengths, collections.Iterable):
                for limit in caches_lengths:
                    caches.append(Cache(limit))
            else:
                print("Error, expected int or a series of int, given:", caches_lengths)
        else:
            print("Error, no cache length has ben provided.")
        return caches

    def predict(self, data):
        for cache in self.caches:
            cache.store(data)
        probability = super().predict(data)
        return probability

    def _predict_ngram(self, ngram):
        tree_probability = super()._predict_ngram(ngram)
        if self.interpolation_coefficients is None or len(self.caches) == 0 or not all([cache.active for cache in self.caches]):
            return tree_probability

        cache_probability = 0.
        for cache in self.caches:
            cache_probability += cache.predict(ngram[-1])
        cache_probability /= len(self.caches)
        k_c = self.interpolation_coefficients[0]
        k_t = self.interpolation_coefficients[1]
        ngram_probability = k_c * cache_probability + k_t * tree_probability
        return ngram_probability

    def train(self, logs=False, smoothing=False):
        super().train(logs, smoothing)
        self.interpolation_coefficients = self._deleted_interpolation()

    def _deleted_interpolation(self):
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
        return [float(v) / sum(w) for v in w]


class MultiPredictor(Predictor):
    def __init__(self):
        self.predictors = self._construct_predictors()
        self.coefficients = self._construct_coefficients()

    def store(self, data):
        for predictor in self.predictors:
            predictor.store(data)

    def predict(self, data):
        probabilities = list()
        for predictor in self.predictors:
            prediction = predictor.predict(data)
            probabilities.append(prediction)
        for i in range(len(probabilities)):
            probabilities[i] *= self.coefficients[i]
        result = sum(probabilities)
        return result

    def train(self):
        for predictor in self.predictors:
            predictor.train()

    def _construct_predictors(self):
        return None

    def _construct_coefficients(self):
        return None


class MultiNgramPrefixTree(MultiPredictor):
    """
    A class used almost only to encapsulate the use of multiple prefix trees
    used to smooth out prediction
    """

    def __init__(self, n, coefficients=None):
        assert n > 0, "n must be greater than 0, given: {}".format(n)
        self.n = n
        super(MultiNgramPrefixTree, self).__init__()

    def _construct_predictors(self):
        super()._construct_predictors()
        trees = list()
        for i in range(1, self.n + 1):
            trees.append(PrefixTree(i))
        return trees

    def train(self, logs=False, smoothing=False):
        super(MultiNgramPrefixTree, self).train()
        self.coefficients = self._deleted_interpolation()

    def get(self, sequence, n=None):
        n = self.n-1 if n is None else n-1
        return self.predictors[n].get(sequence)

    def traverse(self, n=None, node=None, sequence=None, size=None):
        n = self.n-1 if n is None else n-1
        return self.predictors[n].traverse(node, sequence, size)

    def get_vocabulary(self, n=None):
        n = self.n-1 if n is None else n-1
        return self.predictors[n].vocabulary(n)

    def _deleted_interpolation(self):
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
        return [float(v) / sum(w) for v in w]


class Storage(Predictor):
    def __init__(self, pos=None):
        self.pos = pos
        self.words = defaultdict(int)
        self.total = 0

    def store(self, data):
        self.words[data] += 1
        self.total += 1

    def predict(self, data):
        prediction = self._probability(data)
        return prediction

    def train(self):
        print("No need for training {}".format(self))

    def _probability(self, word):
        return self.words[word] / self.total


class PosTree(MultiNgramPrefixTree):
    spacy_model_path = "en_core_web_sm"

    class PosCollector:

        def __init__(self, caches_lengths=200, alpha=0.5):
            self.collectors = dict()
            self.caches = None
            self.caches_lengths = caches_lengths
            assert 0 <= alpha <= 1, "Alpha must be a number between 0 and 1, given: {}".format(alpha)
            self.alpha = alpha

            if self.caches_lengths:
                if isinstance(self.caches_lengths, int):
                    self.caches_lengths = [self.caches_lengths]
                    self.caches = [dict()]
                elif isinstance(self.caches_lengths, collections.Iterable):
                    self.caches = [dict()] * len(self.caches_lengths)
                else:
                    print("Error, expected int or an iterable object, given:", self.caches_lengths)

        def store(self, pos, word):
            if pos not in self.collectors.keys():
                self.collectors[pos] = Storage(pos)

            if self.caches:
                for i in range(len(self.caches)):
                    if pos not in self.caches[i].keys():
                        self.caches[i][pos] = Cache(self.caches_lengths[i])

            self.collectors[pos].store(word)

            if self.caches:
                for cache in self.caches:
                    cache[pos].store(word)

        def frequency(self, word):
            frequencies = dict()
            for pos, storage in self.collectors.items():
                frequencies[pos] = storage.probability(word)
            return frequencies

        def word_probability(self, word):
            if not self.caches:
                return self.frequency(word)

            probabilities = dict()
            for pos, frequency in self.collectors.items():
                probabilities[pos] = self.alpha * frequency

                beta = (1 - self.alpha) / len(self.caches)

                for i in range(len(self.caches)):
                    probabilities[pos] += beta * self.caches[i][pos].probability(word)

            return probabilities

    def __init__(self, n, caches_lengths=None, alpha=0.5):
        super().__init__(n)
        self.nlp = spacy.load(PosTree.spacy_model_path)
        self.collector = PosTree.PosCollector(caches_lengths, alpha)
        self.vocabulary = defaultdict(int)
        self.unique_words_rate = 0.

    def predict_sentence(self, sentence, n=None, annotations=True, use_interpolation=False):
        n = self.n if n is None else n
        from math import prod

        sentence, pos_sentence = self.get_sentence_pos(sentence, annotations)

        if not use_interpolation:
            tree = self.trees[n]
            probabilities = list()
            ngrams = PrefixTree.extract_ngrams(sentence, n)
            pos_ngrams = PrefixTree.extract_ngrams(pos_sentence, n)

            for i in range(len(ngrams)):
                ngram = ngrams[i]
                pos_gram = pos_ngrams[i]
                probability = self.__extract_probability_pos_gram__(n, ngram, pos_gram, tree)
                probabilities.append(probability)

            return sum(probabilities) if tree.logs else prod(probabilities)

        if not self.trees[n].interpolation:
            self.trees[n].__deleted_interpolation__()

        coefficients = self.trees[n].interpolation
        probabilities = list()

        ngrams = PrefixTree.extract_ngrams(sentence, n)
        pos_ngrams = PrefixTree.extract_ngrams(pos_sentence, n)

        for i in range(len(ngrams)):
            ngram = ngrams[i]
            pos_gram = pos_ngrams[i]
            multigram_probabilities = list()
            for j in range(1, n + 1):
                probability = self.__extract_probability_pos_gram__(j, ngram[-j:], pos_gram[-j:], self.trees[j])
                multigram_probabilities.append(probability)

            for j in range(len(multigram_probabilities)):
                multigram_probabilities[j] *= coefficients[j]

            interpolated_probability = sum(multigram_probabilities)
            probabilities.append(interpolated_probability)

        return sum(probabilities) if self.logs else prod(probabilities)

    def __extract_probability_pos_gram__(self, n, ngram, pos_gram, tree):
        if ngram[-1] not in self.vocabulary.keys():
            return self.unique_words_rate

        ngram_probability = 0.
        probabilities = self.collector.word_probability(ngram[-1])
        pos_node = tree.get(pos_gram[:-1])
        for possible_pos_gram in tree.traverse(pos_node, pos_gram[:-1], n):
            p_word = probabilities[possible_pos_gram[-1]]
            # the float number is arbitrary
            p_pos = tree.get(possible_pos_gram).probability + 0.000001
            ngram_probability += p_word * p_pos
        return ngram_probability

    def get_sentence_pos(self, corpus_sentence, annotation=True):

        if annotation:
            # remove "[Start]" and "[End]" for spacy
            corpus_sentence = corpus_sentence[1:-1]

        sentence = " ".join(corpus_sentence)
        doc = self.nlp(sentence)

        # I'll treat "[Start]" and "[End]" as special characters
        sentence = ["[Start]"] + [token.text for token in doc] + ["[End]"]
        pos_sentence = ["X"] + [token.pos_ for token in doc] + ["X"]

        return sentence, pos_sentence

    @staticmethod
    def store_ngrams(corpus, n, tree=None):
        assert n > 0, "n must be greater than 0, given: {}".format(n)
        if tree is None:
            tree = PosTree(n)

        corpus = [" ".join(sentence) for sentence in corpus]
        for doc in tree.nlp.pipe(corpus):

            # I'll treat "[Start]" and "[End]" as special characters
            sentence = ["[Start]"] + [token.text for token in doc] + ["[End]"]
            pos_sentence = ["X"] + [token.pos_ for token in doc] + ["X"]

            for i in range(1, n + 1):
                for ngram in PrefixTree.extract_ngrams(pos_sentence, i):
                    tree.add_ngram(ngram, i)

            for i in range(len(sentence)):
                tree.collector.store(pos_sentence[i], sentence[i])
                tree.vocabulary[sentence[i]] += 1

        n_unique_words = 0
        for word, n in tree.vocabulary.items():
            if n == 1:
                n_unique_words += 1
        tree.unique_words_rate = n_unique_words / len(tree.vocabulary.keys())

        return tree
