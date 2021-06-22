N-Grammer

This repository is dedicated to the implementation of a simple cache based natural language model and testing its components with different parameters.

Although, ultimately, it was not used, its necessary to install [spaCy](https://spacy.io/) (which is an open-source software library for Natural Language Processing) in order to use PosTree. But this class is not complete at the moment contains several undocumented bugs that makes it impossible to properly use.

[CorpusHelper](ngrammer/CorpusHelper.py) contains functions useful for testing and parsing the dataset files into a format adapted to the models.

[Ngrammer](ngrammer/Ngrammer.py) is where most of the classes are declared and defined.

[Main](main.py) is instead where the training and testing is executed.

The most important classes in this repository are PrefixTree and Cache. The first is the Parent class of almost all the prefix trees tested in the main file. The Cache class is used encapsulated inside other classes transparently and is not usually accessed from outside. Almost all classes are child of a Predictor class, in order to ease the interoperability between them and build more easily more complex classes.

As said before, PosTree is not complete and should not be touched. It was an attempt to implement the same 3g-gram model from R. Kuhn and R. De Mori, "A cache-basednatural language model for speechrecognition" but I was not able to complete it in time at the end.
