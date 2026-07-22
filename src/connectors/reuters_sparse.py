'''
Created on 4 août 2022
@author: vankomme

'''
import logging
from fuzzywuzzy import fuzz
from nltk.corpus import reuters
from nltk import word_tokenize, download
from nltk.stem.snowball import PorterStemmer
from sklearn.datasets import load_svmlight_file, dump_svmlight_file

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn.feature_extraction import _stop_words
import re
import operator
import random
import sys
import spacy
# (base) robert@land:~/workspace/MatchBench$ python -m spacy download en_core_web_lg
# https://stackoverflow.com/questions/56470403/spacy-nlp-spacy-loaden-core-web-lg
# python -m spacy download en_core_web_lg
nlp = spacy.load("en_core_web_lg")
# stop_words = stopwords.words("english")
#
#  Resource [93mpunkt_tab[0m not found.
#   Please use the NLTK Downloader to obtain the resource:

#   [31m>>> import nltk
#   >>> nltk.download('punkt_tab')
download('punkt_tab')
# create logger
log = logging.getLogger('main.'+__name__)
SYSTEM_ERROR = -1

'''

The class enable to process Reuters dataset to generate 
SVMlight training and testing data formats

'''


class ReutersSparse:
    max_features = 10000
    max_training = 1596
    max_testing = 696
    shuffling_seed = 1564

    def __init__(self):
        # get the Reuters dataset ressource from NLTK
        download('reuters')
        self.documents = reuters.fileids()
        # the stop_words are taken from sklearn instead of nltk to avoid sklearn warning
        self.cached_stop_words = list(_stop_words.ENGLISH_STOP_WORDS)
        self.tfidf_fit()
        log.info("Reuters is initialized")


    '''
    Shuffle the item lists and select a number of articles

    input:  articles
            articles_number

    return articles_number from the shuffled list of articles
    '''

    def shuffles(self, articles, articles_number):
        random.seed(self.shuffling_seed)
        random.shuffle(articles)
        return articles[0:articles_number]

    '''
    Pseudo-random shuffling and selection of articles

    input: list_length
           quantity_samples
           selection_seed

    return list of indexis        

    '''

    def random_selected(self, list_length, quantity_samples, selection_seed):
        indexis = []
        random.seed(self.shuffling_seed)
        all_indexis = []
        all_indexis = [i for i in range(list_length)]
        random.shuffle(all_indexis)

        random.seed(selection_seed)
        for i in range(quantity_samples):
            index = random.randint(0, len(all_indexis)-1)
            indexis.append(all_indexis[index])
            del all_indexis[index]

        return indexis

    '''
    Tokenizer 
    
    Warning:
    The point here is that your stop word list needs to be normalised
    (lemmatised, etc) too if you want the entries in that list to be stopped
    from your normalised tokens.

    input: text, the text to be tokenized
    return: the list of tokens
    '''

    def tokenize(self, text):
        min_length = 3  # this doesn't work
        words = map(lambda word: word.lower(), word_tokenize(text))
        words = [word for word in words if word not in self.cached_stop_words]
        tokens = (list(map(lambda token: PorterStemmer().stem(token), words)))
        p = re.compile('[a-zA-Z]+')
        filtered_tokens = list(filter(lambda token: p.match(token) and
                                      len(token) >= min_length, tokens))
        return filtered_tokens

    '''
    Get the vocabulary used for the tfidf vectors
    Returns the list of vocabulary used
    '''

    def get_vocabulary_list(self):
        return self.cv.get_feature_names_out()

    '''
    Get the vocabulary dictionary used for the tfidf vectors
    Returns the dictionary of vocabulary used
    '''

    def get_vocabulary_dictionary(self):
        return self.cv.vocabulary_

    '''
    Fit the TFIDF on the training data

    '''

    def tfidf_fit(self):
        # read all training texts in a list
        train_docs_id = list(
            filter(lambda doc: doc.startswith("train"), self.documents))
        texts = [reuters.raw(id) for id in train_docs_id]
        self.cv = CountVectorizer(max_df=0.85, stop_words=self.cached_stop_words,
                                  tokenizer=self.tokenize, max_features=self.max_features)
        self.wcv = self.cv.fit_transform(texts)

    '''
    Write the inductive training and test files with the 'acq' (positive) and not 'acq' (negative) categaries
    input: num_training is the number of couple of training positive and negative articles
           num_testing is the number of test couple positive and negative
    '''

    def write_svm_inductive_files(self, num_training, num_testing, path):

        if num_training > self.max_training:
            log.error("The number of training articles is out of reuters range: %s > %s",
                      num_training, self.max_training)
            sys.exit(SYSTEM_ERROR)
        if num_testing > self.max_testing:
            log.error("The number of test articles is out of reuters range: %s > %s",
                      num_testing, self.max_testing)
            sys.exit(SYSTEM_ERROR)

        train_docs_id = list(
            filter(lambda doc: doc.startswith("train"), self.documents))
        test_docs_id = list(
            filter(lambda doc: doc.startswith("test"), self.documents))

        train_pos = self.shuffles(
            [a for a in train_docs_id if reuters.categories(a) == ['acq']], num_training)
        train_neg = self.shuffles(
            [a for a in train_docs_id if 'acq' not in reuters.categories(a)], num_training)
        test_pos = self.shuffles(
            [a for a in test_docs_id if reuters.categories(a) == ['acq']], num_testing)
        test_neg = self.shuffles(
            [a for a in test_docs_id if 'acq' not in reuters.categories(a)], num_testing)

        train_docs = [reuters.raw(doc_id) for doc_id in train_pos] + \
            [reuters.raw(doc_id) for doc_id in train_neg]
        test_docs = [reuters.raw(doc_id) for doc_id in test_pos] + \
            [reuters.raw(doc_id) for doc_id in test_neg]
        train_targets = [1 for i in range(
            0, len(train_pos))]+[-1 for i in range(0, len(train_neg))]
        test_targets = [1 for i in range(
            0, len(test_pos))]+[-1 for i in range(0, len(test_neg))]

        tfidf_transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
        transformer = tfidf_transformer.fit(self.wcv)

        train_vectors = transformer.transform(self.cv.transform(train_docs))
        test_vectors = transformer.transform(self.cv.transform(test_docs))

        # write the training and testing files in the svmlight format
        dump_svmlight_file(train_vectors, train_targets,
                           path+"/inductive/"+"inductive_train.txt", zero_based=False)
        dump_svmlight_file(test_vectors, test_targets, path +
                           "/inductive/"+"inductive_test.txt", zero_based=False)

    '''
    Write the transductive training and test files with the 'acq' (positive) and not 'acq' (negative) categaries
    input:  num_label_couples, the number of label that svmlight has at disposal for the training
            num_training_couples is the number of couple of training positive and negative articles
            num_testing_couples is the number of test couple positive and negative
    '''

    def write_svm_transductive_files(self, num_label_couples, num_training_couples, num_testing_couples, selection_seed, path):

        if num_training_couples > self.max_training:
            log.error("The number of training articles is out of reuters range: %s > %s",
                      num_training_couples, self.max_training)
            sys.exit(SYSTEM_ERROR)
        if num_label_couples > num_training_couples:
            log.error("The number of labeled articles is out of the range of training samples: %s > %s",
                      num_label_couples, num_training_couples)
            sys.exit(SYSTEM_ERROR)
        if num_testing_couples > self.max_testing:
            log.error("The number of test articles is out of reuters range: %s > %s",
                      num_testing_couples, self.max_testing)
            sys.exit(SYSTEM_ERROR)

        train_docs_id = list(
            filter(lambda doc: doc.startswith("train"), self.documents))
        test_docs_id = list(
            filter(lambda doc: doc.startswith("test"), self.documents))

        train_pos = self.shuffles([a for a in train_docs_id if reuters.categories(
            a) == ['acq']], num_training_couples)
        train_neg = self.shuffles(
            [a for a in train_docs_id if 'acq' not in reuters.categories(a)], num_training_couples)

        test_pos = self.shuffles([a for a in test_docs_id if reuters.categories(a) == [
                                 'acq']], num_testing_couples)
        test_neg = self.shuffles(
            [a for a in test_docs_id if 'acq' not in reuters.categories(a)], num_testing_couples)

        train_docs = [reuters.raw(doc_id) for doc_id in train_pos] + \
            [reuters.raw(doc_id) for doc_id in train_neg]
        test_docs = [reuters.raw(doc_id) for doc_id in test_pos] + \
            [reuters.raw(doc_id) for doc_id in test_neg]

        # select the labels for the transductive training
        labels = self.random_selected(
            len(train_pos), num_label_couples, selection_seed)
        train_positive_targets = [
            1 if i in labels else 0 for i in range(0, len(train_pos))]
        labels = self.random_selected(
            len(train_neg), num_label_couples, selection_seed)
        train_negative_targets = [
            -1 if i in labels else 0 for i in range(0, len(train_neg))]
        train_targets = train_positive_targets+train_negative_targets

        test_targets = [1 for i in range(
            0, len(test_pos))]+[-1 for i in range(0, len(test_neg))]

        tfidf_transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
        transformer = tfidf_transformer.fit(self.wcv)

        train_vectors = transformer.transform(self.cv.transform(train_docs))
        test_vectors = transformer.transform(self.cv.transform(test_docs))

        # write the training and testing files in the svmlight format
        dump_svmlight_file(train_vectors, train_targets,
                           path+"/transductive/"+"transductive_train.txt", zero_based=False)
        dump_svmlight_file(test_vectors, test_targets, path +
                           "/transductive/"+"transductive_test.txt", zero_based=False)


'''
Tokenizer according to spacy, 
the one of nltk seems to work better for my usage
number are not filtered, & ; commands are not filtered lemmatization seems not to work

'''


def tokenize_spacy(text):
    tokens = nlp(text)
    tokens = [token.lemma_.lower() for token in tokens if (
        token.is_stop == False and token.is_punct == False and token.lemma_.strip() != ''
    )]

    return tokens


'''
Read the words used in SVMLight examples

'''


def vocabulary_svmlight():
    # log.info(flags['reuters_path'])
    file = "/home/robert/datasets/reuters/example1/words"
    log.info(file)

    with open(file, 'rb') as f:
        lines = f.read().splitlines()
        utf8_safe_lines = [line.decode('utf-8', 'ignore') for line in lines]
    f.close()
    return utf8_safe_lines


'''
Read the example1 training dataset
# https://stackoverflow.com/questions/64340648/how-to-get-indices-of-lil-matrix-elements-in-a-for-loop

in voc the vobulary used by exemples datasets

returns the files word content for each file
'''


def load_example(voc):
    data = load_svmlight_file(
        "/home/robert/datasets/reuters/example1/train.dat", n_features=9947)
    log.info(type(data))
    l = data[0].tolil()

    log.info("The number of files %s", len(l.data))
    files = []
    for f in range(len(l.data)):
        log.debug(" the number of words %s", len(l.data[f]))
        words = []
        for x in range(len(l.data[f])):
            words.append(voc[l.rows[f][x]])

        files.append(words)

    log.debug("bag of words %s", files[1001])
    return files


'''
For each Reuters's items

in matches,svmlight_bow,reuters_item,reuters

'''


def get_item_match_value(matches, svmlight_bow, reuters_item, reuters):
    # reuters bag of words
    reuters_bow = set(tokenize_spacy(reuters.raw(reuters_item)))
    # the number of token is the article
    length = len(tokenize_spacy(reuters.raw(reuters_item)))
    match_count = 0
    # for each word in the bow check the match
    for word in reuters_bow:
        if word == 'icx':
            log.info(reuters_item)
        for svmword in svmlight_bow:
            m = fuzz.ratio(word, svmword)
            if m == 100:
                match_count += m

    # normalizes by the number of tokens
    # if length != 0 :
    #    match_count /= length
    # else :
    #    match_count = 0
    matches[reuters_item] = match_count


'''
Rebuild the list of items of Reuters dataset that are used by SVMlight examples

Deprecated: this does not work!

'''


def rebuild():
    log.info('Rebuilt Reuters from source')
    download('reuters')
    documents = reuters.fileids()
    train_docs_id = list(
        filter(lambda doc: doc.startswith("train"), documents))
    test_docs_id = list(filter(lambda doc: doc.startswith("test"), documents))

    # vocabulary and bag of words used by svmlight
    voc = vocabulary_svmlight()
    bows = load_example(voc)

    items = set()
    # for the first 1000 positive svmlight training files
    for svmligth_item in range(1):
        svmlight_bow = set(bows[svmligth_item])
        log.info(svmlight_bow)
        matches = {}
        for reuters_item in train_docs_id:  # search for a match in all Reuters training files
            if 'acq' in reuters.categories(reuters_item):
                # in the categary of M&A
                get_item_match_value(matches, svmlight_bow,
                                     reuters_item, reuters)
            else:
                # never search out of class M&A
                matches[reuters_item] = 0

        # for each svmligth item, which is the one that matches best?
        id = (max(matches.items(), key=operator.itemgetter(1))[0])
        # test duplication
        if id in items:
            log.error("Duplicated item %s for %s", id, svmligth_item)
        else:
            items.add(id)

    # write items' set to train_pos.txt
    #i = list(items)[8]
    log.info(items)
    i = 'training/12309'
    log.info(i)
    log.info(set(tokenize_spacy(reuters.raw(i))))
    log.info(reuters.raw(i))
    log.info(reuters.categories(i))

    count = 0
    total = 0
    neg = 0
    for it in train_docs_id:
        if 'acq' not in reuters.categories(it):
            neg += 1
        if 'acq' in reuters.categories(it):
            total += 1
            if len(reuters.categories(it)) == 1:
                count += 1

    log.info(
        "number of acq %s sur un total de %s and the negative files are %s", count, total, neg)

    count = 0
    total = 0
    neg = 0
    for it in test_docs_id:
        if 'acq' not in reuters.categories(it):
            neg += 1
        if 'acq' in reuters.categories(it):
            total += 1
            if len(reuters.categories(it)) == 1:
                count += 1

    log.info(
        "number of acq %s sur un total de %s and the negative files are %s", count, total, neg)