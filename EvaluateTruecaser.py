import os
import argparse
from Truecaser import getTrueCase
import pickle as cPickle
import nltk
from tqdm import tqdm

DEFAULT_SENTENCES_FILE = 'en_test_sentences.txt'

# Create arguemnt parser
def create_parser():
    parser = argparse.ArgumentParser(description='Evaluate True Caser.')
    parser.add_argument('-in_path', type=str,
                        help='The *file* path to the input file contained sentences seperated by newline.')
    parser.add_argument('-out_path', type=str,
                        help='The *folder* path to the output files.')
    parser.add_argument('-out_prefix', type=str,
                        help='The prefix of output file name.')
    parser.add_argument('-out_suffix', type=str,
                        help='The suffix of output file name. Typically "src" or "tgt".')
    return parser


def evaluateTrueCaser(testSentences, wordCasingLookup, uniDist, backwardBiDist, forwardBiDist, trigramDist):
    correctTokens = 0
    totalTokens = 0
    casedTestSentences = []
    
    for sentence in tqdm(testSentences):
        tokensCorrect = nltk.word_tokenize(sentence)
        tokens = [token.lower() for token in tokensCorrect]
        tokensTrueCase = getTrueCase(tokens, 'title', wordCasingLookup, uniDist, backwardBiDist, forwardBiDist, trigramDist)
        casedTestSentences.append(' '.join(tokensTrueCase))
        
        perfectMatch = True
        
        for idx in range(len(tokensCorrect)):
            totalTokens += 1
            if tokensCorrect[idx] == tokensTrueCase[idx]:
                correctTokens += 1
            else:
                perfectMatch = False
        
        if not perfectMatch:
            print(tokensCorrect)
            print(tokensTrueCase)
            print("-------------------")

    print("Accuracy: %.2f%%" % (correctTokens / float(totalTokens)*100))
    
    return casedTestSentences
    

def load_test_sentences(file_name):
    with open(file_name, 'r', encoding='utf8') as fh:
        test_sentences = fh.readlines()
    
    return test_sentences


def main(args):
    f = open('distributions.obj', 'rb')
    uniDist = cPickle.load(f)
    backwardBiDist = cPickle.load(f)
    forwardBiDist = cPickle.load(f)
    trigramDist = cPickle.load(f)
    wordCasingLookup = cPickle.load(f)
    f.close()
    
    testSentences = []
    if args.in_path:
        with open(args.in_path, 'r') as file_in:
            for line in file_in:
                testSentences.append(line.strip())
    else:
        testSentences = load_test_sentences(DEFAULT_SENTENCES_FILE)
    
    print('Evaluating {} test sentences...'.format(len(testSentences)))
    casedTestSentences = evaluateTrueCaser(testSentences, wordCasingLookup, uniDist, backwardBiDist, forwardBiDist, trigramDist)
    
    if args.out_path or args.out_prefix or args.out_suffix:
        # Check output path is exists, otherwise create one.
        if not os.path.exists(args.out_path):
            os.makedirs(args.out_path)
        
        file_out_path = os.path.join(args.out_path, '{}.truecased.{}'.format(args.out_prefix, args.out_suffix))
        with open(file_out_path, 'w') as file_out:
            for sentence in casedTestSentences:
                print(sentence, file=file_out)
            print('Save cased test sentences to "{}"'.format(file_out_path))
    

if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    main(args)
