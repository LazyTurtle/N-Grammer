import ngrammer
from ngrammer.ngrammer import *
from ngrammer import CorpusHandler

def print_hi(name):

    print(f'Hi, {name}')
    tree = PrefixTree()
    train = "dataset/NL2SparQL4NLU.train.utterances.txt"
    corpus = CorpusHandler.read_corpus(train,True)
    tree = PrefixTree.store_ngrams(corpus,2,tree)
    tree = PrefixTree.compute_probabilities(tree)
    print(tree.get_word(["the", "play"]).probability)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
