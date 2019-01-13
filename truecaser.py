"""
This file contains the functions to truecase a sentence. Taken from:
https://github.com/wtee/truecaser/blob/master/truecaser.py
"""

from collections import namedtuple
import math
import pickle
import string

Model = namedtuple(
    "Model",
    (
        "uni_dist",
        "backward_bi_dist",
        "forward_bi_dist",
        "trigram_dist",
        "word_casing_lookup",
    ),
)


def get_score(prev_token, pos_token, next_token, model):
    pseudo_count = 5.0

    # Get Unigram Score
    nominator = model.uni_dist[pos_token] + pseudo_count
    denominator = 0
    for alt_token in model.word_casing_lookup[pos_token.lower()]:
        denominator += model.uni_dist[alt_token] + pseudo_count

    unigram_score = nominator / denominator

    # Get Backward Score
    bigram_backward_score = 1
    if prev_token is not None:
        nominator = model.backward_bi_dist[prev_token + "_" + pos_token] + pseudo_count
        denominator = 0
        for alt_token in model.word_casing_lookup[pos_token.lower()]:
            denominator += (
                model.backward_bi_dist[prev_token + "_" + alt_token] + pseudo_count
            )

        bigram_backward_score = nominator / denominator

    # Get Forward Score
    bigram_forward_score = 1
    if next_token is not None:
        next_token = next_token.lower()  # Ensure it is lower case
        nominator = model.forward_bi_dist[pos_token + "_" + next_token] + pseudo_count
        denominator = 0
        for alt_token in model.word_casing_lookup[pos_token.lower()]:
            denominator += (
                model.forward_bi_dist[alt_token + "_" + next_token] + pseudo_count
            )

        bigram_forward_score = nominator / denominator

    # Get Trigram Score
    trigram_score = 1
    if prev_token is not None and next_token is not None:
        next_token = next_token.lower()  # Ensure it is lower case
        nominator = (
            model.trigram_dist[prev_token + "_" + pos_token + "_" + next_token]
            + pseudo_count
        )
        denominator = 0
        for alt_token in model.word_casing_lookup[pos_token.lower()]:
            denominator += (
                model.trigram_dist[prev_token + "_" + alt_token + "_" + next_token]
                + pseudo_count
            )

        trigram_score = nominator / denominator

    result = (
        math.log(unigram_score)
        + math.log(bigram_backward_score)
        + math.log(bigram_forward_score)
        + math.log(trigram_score)
    )
    # print "Scores: %f %f %f %f = %f" % (unigram_score, bigram_backward_score, bigram_forward_score, trigram_score, math.exp(result))

    return result


def load_model(model_file):
    with open(model_file, "rb") as fh:
        model = Model(
            pickle.load(fh),
            pickle.load(fh),
            pickle.load(fh),
            pickle.load(fh),
            pickle.load(fh),
        )

    return model


def get_true_case(tokens, out_of_vocab_token_opt, model):
    """
    Returns the true case for the passed tokens.
    @param tokens: Tokens in a single sentence
    @param out_of_vocab_token_opt:
        title: Returns out of vocabulary (OOV) tokens in 'title' format
        lower: Returns OOV tokens in lower case
        as-is: Returns OOV tokens as is
    """
    tokens_true_case = []
    for token_idx, token in enumerate(tokens):
        if token in string.punctuation or token.isdigit():
            tokens_true_case.append(token)
        else:
            # model.word_casing_lookup expects a lowercase token
            token_lower = token.lower()
            if token_lower in model.word_casing_lookup:
                if len(model.word_casing_lookup[token_lower]) == 1:
                    tokens_true_case.append(list(model.word_casing_lookup[token_lower])[0])
                else:
                    prev_token = (
                        tokens_true_case[token_idx - 1] if token_idx > 0 else None
                    )
                    next_token = (
                        tokens[token_idx + 1] if token_idx < len(tokens) - 1 else None
                    )

                    best_token = None
                    highest_score = float("-inf")

                    for pos_token in model.word_casing_lookup[token_lower]:
                        score = get_score(prev_token, pos_token, next_token, model)

                        if score > highest_score:
                            best_token = pos_token
                            highest_score = score

                    tokens_true_case.append(best_token)

                if token_idx == 0:
                    tokens_true_case[0] = tokens_true_case[0].title()

            else:  # Token out of vocabulary
                print(f"{token} is out of vocabulary.")
                if out_of_vocab_token_opt == "title":
                    tokens_true_case.append(token.title())
                elif out_of_vocab_token_opt == "lower":
                    tokens_true_case.append(token.lower())
                else:
                    tokens_true_case.append(token)

    return tokens_true_case
