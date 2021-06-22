import collections


def load_corpus(corpus_file, annotations=True, filter_unknown=False, cutoff_line=1, placeholder="[Unknown]"):
    corpus = read_corpus(corpus_file)
    if annotations:
        corpus = annotate_corpus(corpus, add_start_end_annotations)
    if filter_unknown:
        frequency_table = frequencies(corpus)
        lexicon = cutoff(frequency_table, cutoff_line)
        corpus = filter_unknowns(corpus, lexicon, placeholder)
        return corpus, lexicon
    return corpus


def read_corpus(corpus_file):
    structured_corpus = list()
    with open(corpus_file) as corpus:
        for sentence in corpus:
            sentence = sentence.strip()
            sentence_list = sentence.split()
            structured_corpus.append(sentence_list)
    return structured_corpus


def add_start_end_annotations(sentence_list, start="[Start]", end="[End]"):
    return [start] + sentence_list + [end]


def annotate_corpus(corpus, annotation_function):
    for i in range(len(corpus)):
        corpus[i] = annotation_function(corpus[i])
    return corpus


def frequencies(corpus):
    frequency_table = dict()
    for sentence in corpus:
        for word in sentence:
            frequency_table[word] = frequency_table.setdefault(word, 0) + 1
    return frequency_table


def cutoff(frequency_table, f_min=1, f_max=float("inf")):
    words = list()
    for word, frequency in frequency_table.items():
        if f_min <= frequency <= f_max:
            words.append(word)
    return sorted(words)


def remove_words(lexicon, stopwords):
    filtered = list(set(lexicon) - set(stopwords))
    return sorted(filtered)


def load_lexicon_file(lexicon_file):
    lexicon = list()
    with open(lexicon_file) as file:
        for line in file:
            lexicon.append(line.strip())
    return sorted(lexicon)


def load_lexicon_corpus(corpus):
    lexicon = set()
    for sentence in corpus:
        for word in sentence:
            lexicon.add(word)
    lexicon = list(lexicon)
    return sorted(lexicon)


def filter_unknowns(corpus, lexicon, unknown="[Unknown]"):
    for i in range(len(corpus)):
        for j in range(len(corpus[i])):
            word = corpus[i][j]
            corpus[i][j] = word if word in lexicon else unknown
    return corpus


def partition_corpus(corpus, parts=None):
    if parts is None:
        parts = [0.5, 0.5]
    assert isinstance(parts, collections.Sequence), "Parts bust be a sequence of numbers summing to 1, given:{}".format(parts)
    assert sum(parts) == 1, "Parts must sum to 1, given:{}".format(parts)
    n = len(corpus)
    k = 0
    partitions = list()
    for i in partitions:
        if not i == len(parts)-1:
            k_u = round(k + n * parts[i])
            partitions.append(corpus[k:k_u])
            k = k_u
        else:
            partitions.append(corpus[k:])

    return partitions


def word_length(corpus):
    words = 0
    for sentence in corpus:
        words += len(sentence)
    return words


def good_turing_estimate(corpus):
    from collections import defaultdict
    words = defaultdict(int)

    for sentence in corpus:
        for word in sentence:
            words[word] += 1

    unique_words = 0
    for word, count in words.items():
        if count == 1:
            unique_words += 1

    n_words = len(words)
    return unique_words / n_words
