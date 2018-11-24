"""
This script trains the TrueCase System

There are three options to train the true caser:
1) Use the sentences in NLTK
2) Use a text file. Each line must contain a single sentence. Use a large
   corpus, for example Wikipedia
3) Use Bigrams + Trigrams count from the website
   http://www.ngrams.info/download_coca.asp

The more training data, the better the results.
"""
import argparse
import pickle
import re

import nltk
import nltk.corpus
from nltk.corpus import brown
from nltk.corpus import reuters
from nltk.corpus import gutenberg
import nltk.data
from tqdm import tqdm

from truecaser import Model

def get_casing(word):
    """ Returns the casing of a word"""
    casing = 'other'

    if word.isdigit(): # Is a digit
        casing = 'numeric'
    elif word.islower(): # All lower case
        casing = 'allLower'
    elif word.isupper(): # All upper case
        casing = 'allUpper'
    elif word[0].isupper(): # is a title, initial char upper, then all lower
        casing = 'initialUpper'

    return casing


def check_sen_sanity(sentence):
    """ Checks the sanity of the sentence. If the sentence is for example
    all uppercase, it is rejected"""
    case_dist = nltk.FreqDist()

    for token in sentence:
        if len(token) > 0:
            case_dist[get_casing(token)] += 1

    if case_dist.most_common(1)[0][0] != 'allLower':
        return False

    return True


def update_dists_from_sens(text, model):
    """
    Updates the NLTK Frequency Distributions based on a list of sentences.
    text: Array of sentences.
    Each sentence must be an array of Tokens.
    """
    # :: Create unigram lookup ::
    for sentence in tqdm(text, 'Unigrams: '):
        if not check_sen_sanity(sentence):
            continue

        for idx, word in enumerate(sentence):
            model.uni_dist[word] += 1

            if word.lower() not in model.word_casing_lookup:
                model.word_casing_lookup[word.lower()] = set()

            model.word_casing_lookup[word.lower()].add(word)


    # :: Create backward + forward bigram lookup and trigram lookup ::
    for sentence in tqdm(text, 'Bigrams & Trigrams: '):
        if not check_sen_sanity(sentence):
            continue

        for idx, word in enumerate(sentence[2:]): # Start at 2 to skip first word in sentence
            prev_word = sentence[idx-1]
            lower_word = word.lower()
            lower_next_word = sentence[idx+1].lower()

            # Only if there are multiple options
            if (lower_word in model.word_casing_lookup and len(model.word_casing_lookup[lower_word]) >= 2):

                model.backward_bi_dist[prev_word + "_" + word] += 1

                if idx < len(sentence)-1:
                    next_word = sentence[idx+1].lower()
                    model.forward_bi_dist[word + "_" + next_word] += 1

                model.trigram_dist[prev_word + "_" + word + "_" + lower_next_word] += 1

    return model


def update_dists_from_ngrams(bigram_file, trigram_file, model):
    """
    Updates the FrequencyDistribitions based on an ngram file,
    e.g. the ngram file of http://www.ngrams.info/download_coca.asp
    """
    with open(bigram_file, 'r') as fh:
        for line in fh:
            splits = line.strip().split('\t')
            cnt, word1, word2 = splits
            cnt = int(cnt)

            # Unigrams
            if word1.lower() not in model.word_casing_lookup:
                model.word_casing_lookup[word1.lower()] = set()

            model.word_casing_lookup[word1.lower()].add(word1)

            if word2.lower() not in model.word_casing_lookup:
                model.word_casing_lookup[word2.lower()] = set()

            model.word_casing_lookup[word2.lower()].add(word2)


            model.uni_dist[word1] += cnt
            model.uni_dist[word2] += cnt

            # Bigrams
            model.backward_bi_dist[word1 + "_" + word2] += cnt
            model.forward_bi_dist[word1 + "_" + word2.lower()] += cnt


    # Trigrams
    with open(trigram_file, 'r') as fh:
        for line in fh:
            splits = line.strip().split('\t')
            cnt, word1, word2, word3 = splits
            cnt = int(cnt)

            model.trigram_dist[word1 + "_" + word2 + "_" + word3.lower()] += cnt

    return model


def train_truecaser(option, *args):
    """
    @param option:
        nltk: train truecaser based on NLTK corpus
        *.txt: train truecaser based on custom text file
        ngrams: train truecaser based on bigrams and trigrams
                from http://www.ngrams.info/download_coca.asp
    @param args:
        If using option 'ngrams', args takes the file names for the bigram and trigram
        training files. List the bigram file first, then the trigram file.
    """
    text_file_pattern = re.compile(r'.+\.txt')
    model = Model(nltk.FreqDist(),
                  nltk.FreqDist(),
                  nltk.FreqDist(),
                  nltk.FreqDist(),
                  {})

    if option == 'nltk':
        # :: Option 1: Train it based on NLTK corpus ::
        print("Update from NLTK Corpus")
        nltk_corpus = (brown.sents()
                       + reuters.sents()
                       + gutenberg.sents()
                       + nltk.corpus.semcor.sents()
                       + nltk.corpus.conll2000.sents()
                       + nltk.corpus.state_union.sents())
        update_dists_from_sens(nltk_corpus, model)
    elif text_file_pattern.match(option):
        # :: Option 2: Train it based the train.txt file ::
        print("Update from text file")
        sentences = []
        with open(option, 'r', encoding='utf8') as fh:
            sentences = fh.readlines()

        tokens = [nltk.word_tokenize(sentence) for sentence in sentences]
        update_dists_from_sens(tokens, model)
    elif option == 'ngrams':
        # :: Option 3: Train it based ngrams tables from
        # http://www.ngrams.info/download_coca.asp ::
        print("Update Bigrams / Trigrams")
        update_dists_from_ngrams(args[0], args[1], model)

    return model


def save_model(output_file, model):
    with open(output_file, 'wb') as f:
        pickle.dump(model.uni_dist, f, protocol=3)
        pickle.dump(model.backward_bi_dist, f, protocol=3)
        pickle.dump(model.forward_bi_dist, f, protocol=3)
        pickle.dump(model.trigram_dist, f, protocol=3)
        pickle.dump(model.word_casing_lookup, f, protocol=3)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--option', required=True)
    parser.add_argument('--out_file', required=True)
    parser.add_argument('--bigram_in')
    parser.add_argument('--trigram_in')
    args = parser.parse_args()

    if args.bigram_in:
        model = train_truecaser(args.option, args.bigram_in, args.trigram_in)
    else:
        model = train_truecaser(args.option)

    save_model(args.out_file, model)
