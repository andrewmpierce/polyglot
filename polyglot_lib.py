import glob
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
import itertools
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import make_pipeline, make_union
from sklearn.base import TransformerMixin
import re
from textblob import TextBlob
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from polyglot_lib import *
from sklearn.cross_validation import train_test_split


def percentage_of_bracks(text):
    total_length = len(text)
    text = re.sub(r'[^[]]', '', text)
    punc_length = len(text)
    return punc_length/total_length


def percentage_of_curlies(text):
    total_length = len(text)
    text = re.sub(r'[^{}]', '', text)
    punc_length = len(text)
    return punc_length/total_length


def presence_of_at(text):
    total_length = len(text)
    text = re.sub(r'[^@]', '', text)
    punc_length = len(text)
    return punc_length/total_length


def percentage_of_dollar(text):
    total_length = len(text)
    text = re.sub(r'[^$]', '', text)
    punc_length = len(text)
    return punc_length/total_length


def percentage_of_semi(text):
    total_length = len(text)
    text = re.sub(r'[^;]', '', text)
    punc_length = len(text)
    return punc_length/total_length


def percentage_of_hyphen(text):
    total_length = len(text)
    text = re.sub(r'[^-]', '', text)
    punc_length = len(text)
    return punc_length/total_length


def presence_of_end(text):
    total_length = len(text)
    text = re.findall(r'end', text)
    times = len(text)
    return times*3/(total_length)


def presence_of_def(text):
    total_length = len(text)
    text = re.findall(r'def', text)
    times = len(text)
    return times*3/(total_length)


def presence_of_elif(text):
    total_length = len(text)
    text = re.findall(r'elif', text)
    times = len(text)
    return times*3/(total_length)


def presence_of_return(text):
    total_length = len(text)
    text = re.findall(r'return', text)
    times = len(text)
    return times*4/(total_length)


def presence_of_elsif(text):
    total_length = len(text)
    text = re.findall(r'elsif', text)
    times = len(text)
    return times*4/(total_length)


def presence_of_defun(text):
    total_length = len(text)
    text = re.findall(r'defun', text)
    times = len(text)
    return times*4/(total_length)


def presence_of_object(text):
    total_length = len(text)
    text = re.findall(r'object', text)
    times = len(text)
    return times*5/(total_length)


def presence_of_public(text):
    total_length = len(text)
    text = re.findall(r'public static final', text)
    times = len(text)
    return times*4/(total_length)


def presence_of_func(text):
    total_length = len(text)
    text = re.findall(r'func', text)
    times = len(text)
    return times*4/(total_length)


def presence_of_fun(text):
    total_length = len(text)
    text = re.findall(r'fun', text)
    times = len(text)
    return times*3/(total_length)


def presence_of_static(text):
    total_length = len(text)
    text = re.findall(r'static', text)
    times = len(text)
    return times*5/(total_length)


def presence_of_struct(text):
    total_length = len(text)
    text = re.findall(r'struct', text)
    times = len(text)
    return times*6/(total_length)


def percentage_of_hash(text):
    total_length = len(text)
    text = re.sub(r'[^#]', '', text)
    punc_length = len(text)
    return punc_length/total_length


def percentage_of_percent(text):
    total_length = len(text)
    text = re.sub(r'[^%]', '', text)
    punc_length = len(text)
    return punc_length/total_length


def percentage_of_ast(text):
    total_length = len(text)
    text = re.sub(r'[^*]', '', text)
    punc_length = len(text)
    return punc_length/total_length


def percentage_of_arrow(text):
    total_length = len(text)
    text = re.sub(r'[^<>]', '', text)
    punc_length = len(text)
    return punc_length/total_length

def presence_of_let(text):
    total_length = len(text)
    text = re.findall(r'let', text)
    times = len(text)
    return times*3/(total_length)


def percentage_of_parens(text):
    total_length = len(text)
    text = re.sub(r'[^()]', '', text)
    punc_length = len(text)
    return punc_length/total_length


def read_code(directory, lang):
    text = []
    files = glob.glob('benchmarks/benchmarksgame/bench/{}/*{}'.format(directory, lang))
    for file in files:
        with open(file,) as f:
            text.append((f.read(), lang))
    return text


languages = ['.gcc', '.csharp', '.sbcl',
             '.clojure', '.ats',
            '.go', '.hack', '.hs'
            '.java', '.javascript',
            '.jruby', '.ocaml', '.perl', '.tcl'
            '.php', '.python3', '.racket', '.rust',
            '.scala', '.scm', '.vw']

all_langs = [read_code('fasta', lang) for lang in languages]
all_langs  = list(itertools.chain(*all_langs))
langs = [x[0] for x in all_langs]
exts = [x[1] for x in all_langs]

all_langs_fr = [read_code('fastaredux', lang) for lang in languages]
all_langs_fr  = list(itertools.chain(*all_langs_fr))
langs_fr = [x[0] for x in all_langs_fr]
exts_fr = [x[1] for x in all_langs_fr]

all_langs_b = [read_code('binarytrees', lang) for lang in languages]
all_langs_b  = list(itertools.chain(*all_langs_b))
langs_b = [x[0] for x in all_langs_b]
exts_b = [x[1] for x in all_langs_b]

all_langs_m = [read_code('meteor', lang) for lang in languages]
all_langs_m  = list(itertools.chain(*all_langs_m))
langs_m = [x[0] for x in all_langs_m]
exts_m = [x[1] for x in all_langs_m]

all_langs_n = [read_code('knucleotide', lang) for lang in languages]
all_langs_n  = list(itertools.chain(*all_langs_n))
langs_n = [x[0] for x in all_langs_n]
exts_n = [x[1] for x in all_langs_n]

all_langs_r = [read_code('revcomp', lang) for lang in languages]
all_langs_r  = list(itertools.chain(*all_langs_r))
langs_r = [x[0] for x in all_langs_r]
exts_r = [x[1] for x in all_langs_r]

all_langs_rd = [read_code('regexdna', lang) for lang in languages]
all_langs_rd  = list(itertools.chain(*all_langs_rd))
langs_rd = [x[0] for x in all_langs_rd]
exts_rd = [x[1] for x in all_langs_rd]

all_langs_md = [read_code('mandelbrot', lang) for lang in languages]
all_langs_md  = list(itertools.chain(*all_langs_md))
langs_md = [x[0] for x in all_langs_md]
exts_md = [x[1] for x in all_langs_md]

all_langs_s = [read_code('spectralnorm', lang) for lang in languages]
all_langs_s  = list(itertools.chain(*all_langs_s))
langs_s = [x[0] for x in all_langs_s]
exts_s = [x[1] for x in all_langs_s]

all_langs_body = [read_code('nbody', lang) for lang in languages]
all_langs_body  = list(itertools.chain(*all_langs_body))
langs_body = [x[0] for x in all_langs_body]
exts_body = [x[1] for x in all_langs_body]


all_langs_t = [read_code('threadring', lang) for lang in languages]
all_langs_t  = list(itertools.chain(*all_langs_t))
langs_t = [x[0] for x in all_langs_t]
exts_t = [x[1] for x in all_langs_t]



x = langs+langs_fr+langs_b+langs_m+langs_n+langs_r+langs_rd+langs_md+langs_s
x = x+langs_body+langs_t
y = exts+exts_fr+exts_b+exts_m+exts_n+exts_r+exts_rd+exts_md+exts_s
y = y+exts_body+exts_t

class FunctionFeaturizer(TransformerMixin):
    def __init__(self, *featurizers):
        self.featurizers = featurizers

    def fit(self, X, y=None):
        '''All SciKit-learn compatible transformers and classifiers have the same
        interface. `fit` should always return the same object (self)'''
        return self

    def transform(self, X):
        '''Given a list of original data, return a list of feature vectors'''
        feature_vectors = []
        for x in X:
            feature_vector = [f(x) for f in self.featurizers]
            feature_vectors.append(feature_vector)

        return np.array(feature_vectors)

class BagOfWordsFeaturizer(TransformerMixin):
    def __init__(self, num_words=None):
        self.num_words = num_words

    def fit(self, X, y=None):
        words = []
        for x in X:
            x = TextBlob(x.lower())
            words += [word.lemmatize() for word in x.words]
        if self.num_words:
            words = Counter(words)
            self._vocab = [word for word, _ in words.most_common(self.num_words)]
        else:
            self._vocab = list(set(words))
        return self

    def transform(self, X):
        vectors = []
        for x in X:
            x = TextBlob(x.lower())
            word_count = Counter(x.words)
            vector = [0] * len(self._vocab)
            for word, count in word_count.items():
                try:
                    idx = self._vocab.index(word)
                    vector[idx] = count
                except ValueError:
                    pass
            vectors.append(vector)
        return vectors

f = FunctionFeaturizer(percentage_of_parens,
                      percentage_of_bracks,
                      percentage_of_semi,
                      percentage_of_dollar,
                      percentage_of_hyphen,
                      percentage_of_arrow,
                      presence_of_end,
                      presence_of_def,
                      presence_of_elif,
                      presence_of_elsif,
                      presence_of_return,
                      presence_of_defun,
                      presence_of_object,
                      presence_of_public,
                      presence_of_func,
                      presence_of_fun,
                      presence_of_static,
                      percentage_of_ast,
                      presence_of_struct,
                      presence_of_let,
                      presence_of_at,
                      )

code_featurizer = make_union(
    BagOfWordsFeaturizer(70),
    f
)

random_tree = make_pipeline(code_featurizer, RandomForestClassifier())
random_tree.fit(x, y)
