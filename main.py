import ngrammer
from ngrammer.ngrammer import *


def print_hi(name):

    print(f'Hi, {name}')
    tree = PrefixTree()
    ngrams = list()
    ngrams += PrefixTree.extract_ngrams(['a', 'b', 'c', 'd'], 3)
    ngrams += PrefixTree.extract_ngrams(['a', 'e', 'c', 'f'], 3)
    ngrams += PrefixTree.extract_ngrams(['a', 'b', 'g', 'h'], 3)
    ngrams += PrefixTree.extract_ngrams(['x', 'y', 'z', 'a'], 3)
    for ngram in ngrams:
        tree.add_ngram(ngram)
    tests = [['a'], ['a', 'b'], ['a', 'x'], ['e', 'c', 'f'], ['a', 'e', 'c', 'f']]

    word = tree.get_word(["y", "z", "a"])
    print(repr(word))
    for i in range(4):
        for v in tree.vocabulary(i):
            print(i,v)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
