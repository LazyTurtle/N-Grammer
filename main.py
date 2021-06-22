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


def extract_coefficients(n, corpus):
    tree = CachedPrefixTree(n)
    for sentence in corpus:
        tree.store(sentence)
    tree.train()
    interpolation = tree._deleted_interpolation()
    return interpolation


def test_trees(trees, test_data):
    print("Start testing")
    print_progress_bar(0, n_samples, prefix='Progress:', suffix='Complete', length=50)
    n_test_data = len(test_data)
    perplexities = defaultdict(int)
    for i in range(n_test_data):

        for name, tree in trees.items():
            perplexities[tree] += calc_perplexity(tree, test_data[i])
        print_progress_bar(i + 1, n_samples, prefix='Progress:', suffix='Complete', length=50)


    for tree, perplexity in perplexities.items():
        perplexities[tree] /= n_test_data

    for name, tree in trees.items():
        print("Mean Perplexity {}: {}".format(name, perplexities[tree]))


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print("Start")
    annotations = True
    train = "dataset/ptbdataset/ptb.train.txt"
    validation = "dataset/ptbdataset/ptb.valid.txt"
    test = "dataset/ptbdataset/ptb.test.txt"

    print("Loading datasets")
    train_corpus = CorpusHandler.load_corpus(train, annotations)
    validation_corpus = CorpusHandler.load_corpus(validation, annotations)
    test_corpus = CorpusHandler.load_corpus(test, annotations)

    N = 3
    print("Calculating interpolation coefficients")
    coefficients = extract_coefficients(N, validation_corpus)
    turing = CorpusHandler.good_turing_estimate(test_corpus)

    print("Instantiating trees")
    trees = dict()
    trees["Standard prefix tree"] = PrefixTree(N)
    trees["Single cache(400) prefix tree"] = CachedPrefixTree(N, 400)
    trees["Single cache(200) prefix tree"] = CachedPrefixTree(N, 200)
    trees["Single cache(100) prefix tree"] = CachedPrefixTree(N, 100)
    trees["Single cache(50) prefix tree"] = CachedPrefixTree(N, 50)
    trees["Single cache(25) prefix tree"] = CachedPrefixTree(N, 25)
    trees["Multi ngram cached(400) prefix tree"] = CachedMultiNgramPrefixTree(N, 400)
    trees["Multi ngram cached(200) prefix tree"] = CachedMultiNgramPrefixTree(N, 200)
    trees["Multi ngram cached(100) prefix tree"] = CachedMultiNgramPrefixTree(N, 100)
    trees["Multi ngram cached(50) prefix tree"] = CachedMultiNgramPrefixTree(N, 50)
    trees["Multi ngram cached(25) prefix tree"] = CachedMultiNgramPrefixTree(N, 25)
    trees["Multi cache(200,100,50) prefix tree"] = CachedPrefixTree(N, [200, 100, 50])
    trees["Multi ngram cached(200,100,50) prefix tree"] = CachedMultiNgramPrefixTree(N, [200, 100, 50])

    print("Loading train data")
    print("")

    n_samples = len(train_corpus)

    """
    The pos tree takes a lot of time to store all the data, even more than 5 minutes
    this is because it uses spacy to parse every single sentence one by one singularly
    The batch loading doesn't seem to work properly on my machine
    
    In the case one uses pycharm to run this program, 
    it is necessary to enable the terminal simulation for the debug console
    in the setting for the run window to correctly view the progress bar
    """
    for name, tree in trees.items():
        print("Loading data into: {}".format(name))
        print_progress_bar(0, n_samples, prefix='Progress:', suffix='Complete', length=50)
        for i in range(n_samples):
            tree.store(train_corpus[i])
            print_progress_bar(i + 1, n_samples, prefix='Progress:', suffix='Complete', length=50)
        tree.construct_vocabulary(train_corpus)
        print("")

    logs = True
    smoothing = True
    for name, tree in trees.items():
        print("Training: ", name)
        tree.train(logs, smoothing)
        tree.turing = turing

    for name, tree in trees.items():
        if hasattr(tree.__class__, 'set_coefficients') and callable(getattr(tree.__class__, 'set_coefficients')):
            tree.set_coefficients(coefficients)

    test_trees(trees, test_corpus)
