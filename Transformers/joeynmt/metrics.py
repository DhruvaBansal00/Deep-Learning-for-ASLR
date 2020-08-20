# coding: utf-8
"""
This module holds various MT evaluation metrics.
"""

import sacrebleu

def get_string_alignment(sent_1, sent_2):
    penalty_space = 1 ##insertion/deletion err weight
    penalty_sub = 1 ##subsititution err weight

    sentence_1_arr = sent_1.split(" ")
    sentence_2_arr = sent_2.split(" ")

    num_sent_1 = len(sentence_1_arr) + 1
    num_sent_2 = len(sentence_2_arr) + 1

    dp_matrix = [[0 for i in range(num_sent_2)] for j in range(num_sent_1)]

    for row in range(0, num_sent_1):
        for col in range(0, num_sent_2):
            if row == 0 or col == 0:
                dp_matrix[row][col] = (row + col) * penalty_space
            else:
                if sentence_1_arr[row-1] ==  sentence_2_arr[col-1]:
                    dp_matrix[row][col] = min(dp_matrix[row-1][col-1], penalty_space + min(dp_matrix[row-1][col], dp_matrix[row][col-1]))
                else:
                    dp_matrix[row][col] = min(penalty_sub + dp_matrix[row-1][col-1], penalty_space + min(dp_matrix[row-1][col], dp_matrix[row][col-1]))

    # display_dp(dp_matrix)

    aligned_sent_1 = []
    aligned_sent_2 = []
    row = num_sent_1 - 1
    col = num_sent_2 - 1

    while row != 0 and col != 0:
        if sentence_1_arr[row - 1] == sentence_2_arr[col - 1] and dp_matrix[row][col] == dp_matrix[row - 1][col - 1]:
            aligned_sent_1.insert(0, sentence_1_arr[row -1])
            aligned_sent_2.insert(0, sentence_2_arr[col - 1])
            row -= 1
            col -= 1
        elif sentence_1_arr[row - 1] != sentence_2_arr[col - 1] and  dp_matrix[row][col] == (dp_matrix[row - 1][col - 1] + penalty_sub):
            aligned_sent_1.insert(0, sentence_1_arr[row - 1])
            aligned_sent_2.insert(0, sentence_2_arr[col - 1])
            row -= 1
            col -= 1
        elif dp_matrix[row][col] == (dp_matrix[row][col - 1] + penalty_space):
            aligned_sent_1.insert(0, "_")
            aligned_sent_2.insert(0, sentence_2_arr[col - 1])
            col -= 1
        elif dp_matrix[row][col] == (dp_matrix[row - 1][col] + penalty_space):
            aligned_sent_1.insert(0, sentence_1_arr[row - 1])
            aligned_sent_2.insert(0, "_")
            row -= 1

    while row > 0 or col > 0:
        if row > 0:
            aligned_sent_1.insert(0, sentence_1_arr[row - 1])
            aligned_sent_2.insert(0, "_")
            row -= 1
        else:
            aligned_sent_1.insert(0, "_")
            aligned_sent_2.insert(0, sentence_2_arr[col - 1])
            col -= 1
    
    return aligned_sent_1, aligned_sent_2


def chrf(hypotheses, references):
    """
    Character F-score from sacrebleu

    :param hypotheses: list of hypotheses (strings)
    :param references: list of references (strings)
    :return:
    """
    return sacrebleu.corpus_chrf(hypotheses=hypotheses, references=references)


def bleu(hypotheses, references):
    """
    Raw corpus BLEU from sacrebleu (without tokenization)

    :param hypotheses: list of hypotheses (strings)
    :param references: list of references (strings)
    :return:
    """
    return sacrebleu.raw_corpus_bleu(sys_stream=hypotheses,
                                     ref_streams=[references]).score


def token_accuracy(hypotheses, references, level="word"):
    """
    Compute the accuracy of hypothesis tokens: correct tokens / all tokens
    Tokens are correct if they appear in the same position in the reference.

    :param hypotheses: list of hypotheses (strings)
    :param references: list of references (strings)
    :param level: segmentation level, either "word", "bpe", or "char"
    :return:
    """
    def split_by_space(string):
        """
        Helper method to split the input based on spaces.
        Follows the same structure as list(inp)
        :param string: string
        :return: list of strings
        """
        return string.split(" ")

    correct_tokens = 0
    all_tokens = 0
    split_func = split_by_space if level in ["word", "bpe"] else list
    assert len(hypotheses) == len(references)
    for hyp, ref in zip(hypotheses, references):
        all_tokens += len(hyp)
        for h_i, r_i in zip(split_func(hyp), split_func(ref)):
            # min(len(h), len(r)) tokens considered
            if h_i == r_i:
                correct_tokens += 1
    return (correct_tokens / all_tokens)*100 if all_tokens > 0 else 0.0


def sequence_accuracy(hypotheses, references):
    """
    Compute the accuracy of hypothesis tokens: correct tokens / all tokens
    Tokens are correct if they appear in the same position in the reference.

    :param hypotheses: list of hypotheses (strings)
    :param references: list of references (strings)
    :return:
    """
    assert len(hypotheses) == len(references)
    correct_sequences = sum([1 for (hyp, ref) in zip(hypotheses, references)
                             if hyp == ref])
    return (correct_sequences / len(hypotheses))*100 if hypotheses else 0.0


def word_error_allignment(hypotheses, references):
    
    assert len(hypotheses) == len(references)
    
    H,D,S,I,N = 0.0,0.0,0.0,0.0,0.0

    for i in range(len(hypotheses)):
        pred_align, gt_align = get_string_alignment(hypotheses[i], references[i])

        for index in range(len(pred_align)):
            N += 1.0
            if pred_align[index] == gt_align[index]:
                H += 1.0
            elif pred_align[index] == "_":
                D += 1.0
            elif gt_align[index] == "_":
                I += 1.0
            else:
                S += 1.0
    
    return [100 * (H/N), 100 * (D/N), 100 * (S/N), 100*(I/N), 100*(H/N - I/N)]