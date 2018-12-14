import pickle
from datetime import datetime
from os import listdir
from os.path import isfile, join
from random import sample, choice
import re

import numpy as np
from matplotlib import pyplot as plt
from numpy.random import choice as npchoice
from numpy.random import randint
from pycrayon import CrayonClient

from global_vars import NUM_CHARS, character_to_index_mapping

# Evolutionary parameters
target_distribution = np.ones((NUM_CHARS,))/NUM_CHARS
original_text_distribution = np.zeros((NUM_CHARS,))/NUM_CHARS
original_sequence_distribution = np.zeros((NUM_CHARS, NUM_CHARS, NUM_CHARS))
solution_size = 5000
mutation_probability = 0.1
crossover_probability = 1
selection_percentage = 0.7
population_size = 1000
max_iterations = 300
uppercase_probability = 0.7

# Crayon setup
cc = CrayonClient('localhost')
cc.remove_all_experiments()
exp = cc.create_experiment('Dataset Content Evolution - {}'.format(datetime.now()))


def clean_text(text: str):
    ret = ''
    char_cnt = 0
    valid_chars = set(character_to_index_mapping.keys())
    text = re.sub(r'-+', '-', text)
    for char in text:
        char_cnt = char_cnt + 1
        if char in valid_chars:
            ret = ret + char
        if char_cnt % 100000 == 0:
            print('Cleaning text... ({:3.2f}%)'.format(char_cnt / len(text) * 100), end='\r')
    print('Cleaning text... (100.00%)')
    return ret


def delete_duplicates(text: str):
    words = text.split()
    return ' '.join(set(words))


def character_distribution(text: str):
    distribution = np.zeros((NUM_CHARS,))
    for char in character_to_index_mapping.keys():
        distribution[character_to_index_mapping[char]] = text.count(char)
    if len(text) > 0:
        distribution = distribution / distribution.sum()
    return distribution


def sequence_distribution(text: str):
    text = ' ' + text + ' '
    distr = np.zeros((NUM_CHARS, NUM_CHARS, NUM_CHARS))
    seq_list = [text[i:i + 3] for i in range(0, len(text) - 2, 1)]
    for seq in seq_list:
        distr[character_to_index_mapping[seq[0]], character_to_index_mapping[seq[1]], character_to_index_mapping[seq[2]]] = \
            distr[character_to_index_mapping[seq[0]], character_to_index_mapping[seq[1]], character_to_index_mapping[seq[2]]] + 1
    distr = distr / np.sum(np.sum(np.sum(distr)))
    return distr


def count_duplicates(text: str):
    words = text.split()
    words_unique = set(words)
    return len(words) - len(words_unique)


"""
============================================= Evolutionary Methods =============================================
"""


def sample_new_words(bag_of_words: set, number_of_words: int):
    new_words = sample(bag_of_words, number_of_words)
    for word in new_words:
        if npchoice([True, False], p=[uppercase_probability, 1 - uppercase_probability]):
            new_words.remove(word)
            new_words.append(word.capitalize())
    return new_words


def generate_random_solution(bag_of_words: set, mean_words_length: float):
    words = sample_new_words(bag_of_words, int(np.ceil(solution_size/mean_words_length)))
    restore_individual_size(words, bag_of_words, mean_words_length)
    return ' '.join(words)


def restore_individual_size(ind: list, bag_of_words: set, mean_words_length: float):
    ind_str = ' '.join(ind)
    ind_list = ind
    num_missing_characters = solution_size - len(ind_str)
    if num_missing_characters > 0:
        ind_list.extend(sample_new_words(bag_of_words, int(np.ceil(num_missing_characters/mean_words_length))))
        ind_list = list(set(ind_list))
    return ind_list


def mean_length(individuals: list):
    lengths = [len(ind) for ind in individuals]
    return np.mean(lengths)


def evaluate_fitness(solution: str):
    distr_char = character_distribution(solution)
    distr_seq = sequence_distribution(solution)
    loss_single_character = np.sum(np.abs(distr_char - target_distribution))
    loss_sequences = np.sum(np.abs(distr_seq - original_sequence_distribution))
    return 1 / (0.85*loss_single_character + 0.15*loss_sequences + 1)


def single_mutation(sol: str, bag_of_words: set, mean_words_length: float):
    l = sol.split()
    l.remove(l[randint(0, len(l))])
    l = restore_individual_size(l, bag_of_words, mean_words_length)
    return ' '.join(l)


def mutation(individuals: list, bag_of_words: set, mean_words_length: float):
    ret = []
    for ind in individuals:
        if npchoice([True, False], p=[mutation_probability, 1-mutation_probability]):
            ret.append(single_mutation(ind, bag_of_words, mean_words_length))
        else:
            ret.append(ind)
    return ret


def single_crossover(sol1: str, sol2: str, bag_of_words: set, mean_words_length: float):
    list1 = np.array(sol1.split())
    list2 = np.array(sol2.split())
    if len(list1) < len(list2):
        child1idx = np.array(sample(list(range(len(list1))), int(len(list1) / 2)))
    else:
        child1idx = np.array(sample(list(range(len(list2))), int(len(list2) / 2)))
    tmp = list1[child1idx]
    list1[child1idx] = list2[child1idx]
    list2[child1idx] = tmp
    child1_list = restore_individual_size(list(set(list1)), bag_of_words, mean_words_length)
    child2_list = restore_individual_size(list(set(list2)), bag_of_words, mean_words_length)
    child1 = ' '.join(child1_list)
    child2 = ' '.join(child2_list)
    return child1, child2


def crossover(parents: list, bag_of_words: set, mean_words_length: float):
    children = []
    while len(children) < population_size:
        p1 = choice(parents)
        p2 = choice(parents)
        c1, c2 = single_crossover(p1, p2, bag_of_words, mean_words_length)
        children.extend([c1, c2])
    return children


def selection(individuals: list, scores: list):
    s = [sol.split() for sol in individuals]
    s = np.array([ind for _, ind in sorted(zip(scores, s), reverse=True)])
    ret = list(s[0:int(np.floor(len(s)*selection_percentage))])
    ret = [' '.join(set(i)) for i in ret]
    return ret


def evolve(bag_of_words: set, mean_words_length, verbose: bool = False, show_intermediate_results: bool = False):
    individuals = [generate_random_solution(bag_of_words, mean_words_length) for _ in range(population_size)]
    for iter in range(max_iterations):
        do_print = False
        if iter % 10 == 0:
            do_print = True
        if show_intermediate_results and iter % 20 == 0:
            show_best_result(individuals)
        if verbose:
            print('Iteration {}: Computing fitness...'.format(iter+1), end='\r')
        scores = [evaluate_fitness(ind) for ind in individuals]
        exp.add_scalar_value("Evol/Min Fitness", np.min(scores))
        exp.add_scalar_value("Evol/Max Fitness", np.max(scores))
        exp.add_scalar_value("Evol/Mean Fitness", np.mean(scores))
        exp.add_scalar_value("Evol/Mean Length of Individuals", mean_length(individuals))
        if do_print:
            print('Iteration {}: min fitness {}, max fitness {}, mean fitness {}'.format(iter+1, np.min(scores),
                                                                                         np.max(scores), np.mean(scores)))
        if verbose:
            print('Iteration {}: Selecting parents...'.format(iter + 1), end='\r')
        parents = selection(individuals, scores)
        if verbose:
            print('Iteration {}: Crossover...'.format(iter + 1), end='\r')
        children = parents
        children.extend(crossover(parents, bag_of_words, mean_words_length))
        if verbose:
            print('Iteration {}: Mutating individuals...'.format(iter + 1), end='\r')
        children = mutation(children, bag_of_words, mean_words_length)
        individuals = children
    return individuals


def show_best_result(results: list):
    scores = [evaluate_fitness(result) for result in results]
    best_result = np.array([ind for _, ind in sorted(zip(scores, results), reverse=False)])[0]
    dist = character_distribution(best_result)
    plt.plot(dist)
    plt.plot(target_distribution)
    plt.plot(original_text_distribution)
    plt.show()


"""
========================================= End of Evolutionary Methods ==========================================
"""


if __name__ == '__main__':
    # Adjust target distribution
    # target_distribution[0:26] = target_distribution[0:26] / 15  # Capital letters
    # target_distribution[52:68] = target_distribution[52:68] / 15  # Symbols and digits
    target_distribution[69] = target_distribution[69] * 3  # Blank space
    target_distribution = target_distribution / np.sum(target_distribution)

    load_files = True
    list_of_words = []
    if load_files:
        folder_path = '../data/text/'
        print('Processing files...')
        files = [f for f in listdir(folder_path) if isfile(join(folder_path, f))]
        for file in files:
            print('File ', file)
            with open(join(folder_path, file)) as f:
                init = f.read()
            text = delete_duplicates(init)
            text = clean_text(text)
            text = delete_duplicates(text)
            list_of_words.extend(text.split())
        bag_of_words = set(list_of_words)
        seq_distr = sequence_distribution(' '.join(list(bag_of_words)))
        with open('../data/original_sequence_distribution.pkl', 'wb') as output:
            pickle.dump(seq_distr, output, pickle.HIGHEST_PROTOCOL)
        bag_of_words = set([s.lower() for s in list(bag_of_words)])
        print('Files processed.')
        with open('./bag_of_words.pkl', 'wb') as output:
            pickle.dump(bag_of_words, output, pickle.HIGHEST_PROTOCOL)
    else:
        with open('./bag_of_words.pkl', 'rb') as input:
            bag_of_words = pickle.load(input)
    print('Total number of words: ', len(bag_of_words))
    # Compute original text distribution
    original_text_distribution = character_distribution(' '.join(list(bag_of_words)))
    with open('../data/original_sequence_distribution.pkl', 'rb') as input:
        original_sequence_distribution = pickle.load(input)

    words_length = np.array([len(x) for x in bag_of_words])
    mean_words_length = np.mean(words_length)
    print('Mean length of a word: ', mean_words_length)
    print('Start evolution')
    res = evolve(bag_of_words, mean_words_length, True, True)
    results_filename = './dataset_content_res.pkl'
    with open(results_filename, 'wb') as output:
        pickle.dump(res, output, pickle.HIGHEST_PROTOCOL)
    show_best_result(res)
