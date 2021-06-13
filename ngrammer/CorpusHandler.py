def read_corpus(corpus_file):
    structured_corpus = list()
    with open(corpus_file) as corpus:
        for sentence in corpus:
            sentence = sentence.strip()
            sentence_list = sentence.split()
            structured_corpus += sentence_list

    return structured_corpus
