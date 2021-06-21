import collections
import math
from math import log
import spacy
from collections import defaultdict


def log2prob(log_value):
    from math import exp
    return exp(log_value) if log_value else 0.0


def calc_perplexity(tree, sentence):
    n = len(sentence)
    probability = tree.predict(sentence)
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
        print("Start training {}".format(self))
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

    def __init__(self, maximum=200, minimum=5, logs=False):
        from collections import deque
        self.minimum = minimum
        self.maximum = maximum
        self.cache = deque()
        self.active = False
        self.logs = logs

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
        prob = self._frequency(data)
        if self.logs:
            prob = math.log(prob)
        return prob

    def train(self):
        print("No need to train {}.".format(self))

    def _frequency(self, word):
        if self.active:
            return self.cache.count(word) / len(self.cache)
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
        if self.interpolation_coefficients is None or len(self.caches) == 0 or not all(
                [cache.active for cache in self.caches]):
            return tree_probability

        cache_probability = 0.
        for cache in self.caches:
            cache_probability += cache.predict(ngram[-1])
        cache_probability /= len(self.caches)  # in case we have many caches we use their mean
        k_c = self.interpolation_coefficients[0]
        k_t = self.interpolation_coefficients[self.n - 1]
        coef_sum = sum(self.interpolation_coefficients[1:self.n - 1])
        c = coef_sum * (k_c / (k_c + k_t))
        t = coef_sum * (k_t / (k_c + k_t))
        k_c += c
        k_t += t
        ngram_probability = k_c * cache_probability + k_t * tree_probability
        return ngram_probability

    def train(self, logs=False, smoothing=False):
        super().train(logs, smoothing)
        for cache in self.caches:
            cache.logs = logs
        self.interpolation_coefficients = self._deleted_interpolation()

    def set_coefficients(self,new_coefficients):
        self.interpolation_coefficients = new_coefficients

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

    def set_coefficients(self,new_coefficients):
        self.coefficients = new_coefficients

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
        n = self.n - 1 if n is None else n - 1
        return self.predictors[n].get(sequence)

    def traverse(self, n=None, node=None, sequence=None, size=None):
        n = self.n - 1 if n is None else n - 1
        return self.predictors[n].traverse(node, sequence, size)

    def get_vocabulary(self, n=None):
        n = self.n - 1 if n is None else n - 1
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
    """
    Store the frequencies of singular words among an entire set of words
    """

    def __init__(self, pos=None, logs=False):
        self.pos = pos
        self.words = defaultdict(int)
        self.total = 0
        self.logs = logs

    def store(self, data):
        self.words[data] += 1
        self.total += 1

    def predict(self, data):
        prediction = self._probability(data)
        if self.logs:
            prediction = math.log(prediction)
        return prediction

    def train(self):
        print("No need to train {}".format(self))

    def _probability(self, word):
        return self.words[word] / self.total


class CachedStorage(Storage):
    """
    Grant the storage a cache
    """

    def __init__(self, pos=None, coefficients=None, cache_size=200, logs=False):
        super().__init__(pos, logs)
        self.cache = Cache(cache_size, logs=logs)
        self.coefficients = coefficients

    def store(self, data):
        super().store(data)

    def predict(self, data):
        self.cache.store(data)
        storage_part = super().predict(data)
        cache_part = self.cache.predict(data)
        if cache_part is not None:
            f_c = self.coefficients[0]
            f_s = self.coefficients[1]
            probability = f_c * cache_part + f_s * storage_part
            return probability
        return storage_part

    def set_logs(self, logs=False):
        self.logs = logs
        self.cache.logs = logs


class Collector(Predictor):
    """
    Collect the frequencies of single words over their part of speech in different cached storages
    """

    def __init__(self, caches_lengths=200, logs=False):
        self.collectors = dict()
        self.caches_lengths = caches_lengths
        self.logs = logs

    def store(self, data):
        pos, word = data
        if pos not in self.collectors.keys():
            self.collectors[pos] = CachedStorage(pos, self.caches_lengths, logs=self.logs)
        self.collectors[pos].store(word)

    def predict(self, data):
        frequencies = dict()
        for pos, storage in self.collectors.items():
            frequencies[pos] = storage.predict(data)
        return frequencies

    def set_coefficients(self, coefficients):
        for pos, storage in self.collectors.items():
            storage.coefficients = coefficients

    def set_logs(self, logs=False):
        for pos, collector in self.collectors.items():
            collector.set_logs(logs)


class PosTree(PrefixTree):
    """
    Uses the POS in the prefix tree and the collectors in order to compute the probabilities of sentences
    The predict data should be already formatted properly
    """
    spacy_model_path = "en_core_web_sm"

    def __init__(self, n, caches_length=200):
        super(PosTree, self).__init__(n)
        self.nlp = self._setup_spacy()
        self.word_collector = Collector(caches_length)
        self.tree_coefficient = None
        self.collector_coefficient = None

    def store(self, data):
        sentence, pos_sentence = self._convert_to_pos(data)
        super(PosTree, self).store(pos_sentence)
        for i in range(len(sentence)):
            word = sentence[i]
            pos = pos_sentence[i]
            self.word_collector.store((pos, word))

    def train(self, logs=False, smoothing=False):
        super().train(logs, smoothing)
        coefficients = self._deleted_interpolation()
        self.collector_coefficient = sum(coefficients[:2])
        self.tree_coefficient = sum(coefficients[2:])
        cache_coef = coefficients[0] / self.collector_coefficient
        storage_coef = coefficients[1] / self.collector_coefficient
        self.word_collector.set_coefficients((cache_coef, storage_coef))
        self.word_collector.set_logs(logs)

    def _predict_sentence(self, sentence):
        from math import prod

        sentence, pos_sentence = self._convert_to_pos(sentence)
        ngrams = self._extract_ngrams(sentence)
        posgrams = self._extract_ngrams(pos_sentence)

        probabilities = list()
        for i in range(len(ngrams)):
            ngram = ngrams[i]
            posgram = posgrams[i]
            probabilities.append(self._predict_pos_ngram(ngram, posgram))

        result = sum(probabilities) if self.logs else prod(probabilities)
        return result

    def _predict_pos_ngram(self, ngram, posgram):
        parent = self.get(posgram[:-1])
        probabilities = list()
        word_probabilities = self.word_collector.predict(ngram[-1])
        for possible_pos_gram in self.traverse(parent, posgram[:-1], self.n):
            current_pos = possible_pos_gram[-1]
            pos_probability = super()._predict_ngram(possible_pos_gram)
            word_probability = word_probabilities[current_pos]
            joint_probability = word_probability * self.collector_coefficient + pos_probability * self.tree_coefficient
            probabilities.append(joint_probability)
        probability = sum(probabilities)
        return probability

    def _setup_spacy(self, unknown_placeholder="UNKNOWN"):
        # I'll treat the placeholder for OOV words as a special character
        nlp = spacy.load(PosTree.spacy_model_path)
        ruler = nlp.get_pipe("attribute_ruler")
        patterns = [[{"ORTH": unknown_placeholder}]]
        attrs = {"TAG": "X", "POS": "X"}
        ruler.add(patterns=patterns, attrs=attrs)
        return nlp

    def _convert_to_pos(self, sentence, annotations=("[Start]", "[End]"), unknown_placeholder="UNKNOWN"):

        if annotations is not None:
            # remove "[Start]" and "[End]" for spacy
            sentence = sentence[1:-1]

        for i in range(len(sentence)):
            if sentence[i] == "<unk>":
                sentence[i] = unknown_placeholder

        sentence = " ".join(sentence)
        doc = self.nlp(sentence)

        # I'll treat "[Start]" and "[End]" as special characters
        sentence = [annotations[0]] + [token.text for token in doc] + [annotations[1]]
        pos_sentence = ["X"] + [token.pos_ for token in doc] + ["X"]

        return sentence, pos_sentence

    def set_coefficients(self, new_coefficients):
        self.collector_coefficient = sum(new_coefficients[:2])
        self.tree_coefficient = sum(new_coefficients[2:])

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
