from ngrammer.Ngrammer import *
from ngrammer import CorpusHandler


def generate(tree, sos="[Start]", eos="[End]"):
    import random
    word = sos
    sentence = [sos]
    while word != eos:
        temp = sentence[-(tree.n - 1):]
        node = tree.get(temp)
        children = list(node.children.keys())
        word = random.choice(children)
        sentence.append(word)
    return sentence


def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█', print_end="\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=print_end)
    # Print New Line on Complete
    if iteration == total:
        print()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print("Start")
    annotations = True
    train = "dataset/ptbdataset/ptb.train.txt"
    validation = "dataset/ptbdataset/ptb.valid.txt"
    test = "dataset/ptbdataset/ptb.test.txt"

    print("Loading dataset")
    train_corpus = CorpusHandler.load_corpus(train, annotations)
    validation_corpus = CorpusHandler.load_corpus(validation, annotations)
    test_corpus = CorpusHandler.load_corpus(test, annotations)

    N = 3

    print("Instantiating trees")
    standard_prefix_tree = PrefixTree(N)
    single_cache_tree = CachedPrefixTree(N)
    multi_cache_tree = CachedPrefixTree(N, [200, 100, 25])
    cached_pos_tree = PosTree(N)

    print("Loading train data")
    print("")

    n_samples = len(train_corpus)

    print("Loading standard tree:")
    """
    In the case one uses pycharm to run this program, 
    it is necessary to enable the terminal simulation
    in the setting for the run window to correctly view the progress bar
    """
    print_progress_bar(0, n_samples, prefix='Progress:', suffix='Complete', length=50)
    for i in range(n_samples):
        standard_prefix_tree.store(train_corpus[i])
        print_progress_bar(i + 1, n_samples, prefix='Progress:', suffix='Complete', length=50)

    print("Loading single cache tree:")
    print_progress_bar(0, n_samples, prefix='Progress:', suffix='Complete', length=50)
    for i in range(n_samples):
        single_cache_tree.store(train_corpus[i])
        print_progress_bar(i + 1, n_samples, prefix='Progress:', suffix='Complete', length=50)

    print("Loading multi cache tree:")
    print_progress_bar(0, n_samples, prefix='Progress:', suffix='Complete', length=50)
    for i in range(n_samples):
        multi_cache_tree.store(train_corpus[i])
        print_progress_bar(i + 1, n_samples, prefix='Progress:', suffix='Complete', length=50)

    """
    The pos tree takes an enormous amount of time to load
    this is because it uses spacy to parse every single sentence one by one singularly
    """
    print("Loading cached pos tree:")
    print_progress_bar(0, n_samples, prefix='Progress:', suffix='Complete', length=50)
    for i in range(n_samples):
        cached_pos_tree.store(train_corpus[i])
        print_progress_bar(i + 1, n_samples, prefix='Progress:', suffix='Complete', length=50)

    print(standard_prefix_tree)
    print(single_cache_tree)
    print(multi_cache_tree)
    print(cached_pos_tree)
