def read_corpus(corpus_file, annotations=False):
    structured_corpus = list()
    with open(corpus_file) as corpus:
        for sentence in corpus:
            sentence = sentence.strip()
            sentence_list = sentence.split()
            if annotations:
                sentence_list = add_start_end_annotations(sentence_list)
            structured_corpus += sentence_list

    return structured_corpus


def add_start_end_annotations(sentence_list):
    start = "[Start]"
    end = "[End]"
    return [start] + sentence_list + [end]


def annotate_corpus(corpus, annotation_function):
    for i in range(corpus):
        corpus[i] = annotation_function(corpus[i])
    return corpus
