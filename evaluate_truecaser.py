import argparse
import os

import nltk
from tqdm import tqdm

from truecaser import get_true_case, load_model

DEFAULT_SENTENCES_FILE = 'en_test_sentences.txt'

def evaluate_true_caser(test_sentences, model):
    correct_tokens = 0
    total_tokens = 0
    cased_test_sentences = []

    for sentence in tqdm(test_sentences):
        tokens_correct = nltk.word_tokenize(sentence)
        tokens = [token.lower() for token in tokens_correct]
        tokens_true_case = get_true_case(tokens, 'title', model)
        cased_test_sentences.append(' '.join(tokens_true_case))

        perfect_match = True

        for idx in range(len(tokens_correct)):
            total_tokens += 1
            if tokens_correct[idx] == tokens_true_case[idx]:
                correct_tokens += 1
            else:
                perfect_match = False

        if not perfect_match:
            print(tokens_correct)
            print(tokens_true_case)
            print("-------------------")

    print("Accuracy: %.2f%%" % (correct_tokens / float(total_tokens)*100))

    return cased_test_sentences


def load_test_sentences(file_name):
    with open(file_name, 'r', encoding='utf8') as fh:
        test_sentences = fh.readlines()

    return test_sentences


def main(args):
    model = load_model(args.model)

    test_sentences = []
    if args.in_path:
        with open(args.in_path, 'r') as file_in:
            for line in file_in:
                test_sentences.append(line.strip())
    else:
        test_sentences = load_test_sentences(DEFAULT_SENTENCES_FILE)

    print('Evaluating {} test sentences...'.format(len(test_sentences)))
    cased_test_sentences = evaluate_true_caser(test_sentences, model)

    if args.out_path or args.out_prefix or args.out_suffix:
        # Check output path is exists, otherwise create one.
        if not os.path.exists(args.out_path):
            os.makedirs(args.out_path)

        file_out_path = os.path.join(args.out_path,
                                     '{}.truecased.{}'.format(args.out_prefix,
                                                              args.out_suffix))
        with open(file_out_path, 'w') as file_out:
            for sentence in cased_test_sentences:
                print(sentence, file=file_out)
            print('Save cased test sentences to "{}"'.format(file_out_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate True Caser.')
    parser.add_argument('--in_path', help='The *file* path to the input \
                             file contained sentences seperated by newline.')
    parser.add_argument('--out_path',
                        help='The *folder* path to the output files.')
    parser.add_argument('--out_prefix',
                        help='The prefix of output file name.')
    parser.add_argument('--out_suffix',
                        help='The suffix of output file name. Typically "src" or "tgt".')
    parser.add_argument('--model',
                        help='The distributions file from train_truecaser.py')
    args = parser.parse_args()

    main(args)
