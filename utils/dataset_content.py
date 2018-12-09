import numpy as np
from numpy.random import choice as npchoice
from numpy.random import randint
from global_vars import NUM_CHARS, character_to_index_mapping
from random import sample, shuffle, choice
import pickle
from matplotlib import pyplot as plt


# Evolutionary parameters
target_distribution = np.ones((26,))/26
solution_size = 100
mutation_probability = 0.1
crossover_probability = 0.75
selection_percentage = 0.8
population_size = 1000
max_iterations = 100
eps = 1e-4


def clean_text(text: str):
    ret = ''
    char_cnt = 0
    valid_chars = set(character_to_index_mapping.keys())
    for char in text:
        char_cnt = char_cnt+ 1
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


def simplify_distribution(original_distribution: np.array):
    ret = original_distribution[0:26]
    return ret + original_distribution[26:52]


def count_duplicates(text: str):
    words = text.split()
    words_unique = set(words)
    return len(words) - len(words_unique)



"""
============================================= Evolutionary Methods =============================================
"""


def generate_random_solution(bag_of_words: set):
    words = sample(bag_of_words, solution_size)
    unique_words = list(set(words))
    restore_individual_size(unique_words, bag_of_words)
    return ' '.join(unique_words)  # TODO: express size as characters number, not word count


def restore_individual_size(ind: list, bag_of_words: set):
    while len(ind) != solution_size:
        additional_words = sample(bag_of_words, solution_size - len(ind))
        ind.extend(additional_words)
        ind = list(set(ind))
    return ind


def evaluate_fitness(solution: str):
    assert (len(solution.split()) == solution_size)
    distr = simplify_distribution(character_distribution(solution))
    if len(solution) > 0:
        loss = np.sum(np.abs(distr - target_distribution))
    else:
        loss = 1
    return 1/(loss + eps)


def single_mutation(sol: str, bag_of_words: set):
    l = sol.split()
    l.remove(l[randint(0, solution_size)])
    l = restore_individual_size(l, bag_of_words)
    return ' '.join(l)


def mutation(individuals: list, bag_of_words: set):
    ret = []
    for ind in individuals:
        if npchoice([True, False], p=[mutation_probability, 1-mutation_probability]):
            ret.append(single_mutation(ind, bag_of_words))
        else:
            ret.append(ind)
    return ret


def single_crossover(sol1: str, sol2: str, bag_of_words: set):
    list1 = np.array(sol1.split())
    list2 = np.array(sol2.split())
    shuffle(list1)
    shuffle(list2)
    child1idx = np.array(sample(list(range(solution_size)), int(solution_size/2)))
    tmp = list1[child1idx]
    list1[child1idx] = list2[child1idx]
    list2[child1idx] = tmp
    child1_list = restore_individual_size(list(set(list1)), bag_of_words)
    child2_list = restore_individual_size(list(set(list2)), bag_of_words)
    assert len(child1_list) == solution_size and len(child2_list) == solution_size
    child1 = ' '.join(child1_list)
    child2 = ' '.join(child2_list)
    return child1, child2


def crossover(parents: list, bag_of_words: set):
    num_parents = len(parents)
    children = []
    while len(children) < num_parents:
        p1 = choice(parents)
        p2 = choice(parents)
        if npchoice([True, False], p=[crossover_probability, 1-crossover_probability]):
            c1, c2 = single_crossover(p1, p2, bag_of_words)
        else:
            c1 = p1
            c2 = p2
        children.extend([c1, c2])
    return children


def selection(individuals: list, scores: list, bag_of_words: set):
    s = [sol.split() for sol in individuals]
    s = np.array([ind for _, ind in sorted(zip(scores, s), reverse=True)])
    ret = list(s[0:int(np.floor(len(s)*selection_percentage))])
    new_elements = [np.array(generate_random_solution(bag_of_words).split()) for _ in range(len(s) - len(ret))]
    ret.extend(new_elements)
    ret = [' '.join(set(i)) for i in ret]
    return ret


def evolve(bag_of_words: set, verbose: bool = False):
    individuals = [generate_random_solution(bag_of_words) for _ in range(population_size)]
    for iter in range(max_iterations):
        do_print = False
        if iter % 5 == 0:
            do_print = True
        if verbose:
            print('Iteration {}: Computing fitness...'.format(iter+1), end='\r')
        scores = [evaluate_fitness(ind) for ind in individuals]
        if do_print:
            print('Iteration {}: min fitness {}, max fitness {}, mean fitness {}'.format(iter+1, np.min(scores),
                                                                                         np.max(scores), np.mean(scores)))
        if verbose:
            print('Iteration {}: Selecting parents...'.format(iter + 1), end='\r')
        parents = selection(individuals, scores, bag_of_words)
        if verbose:
            print('Iteration {}: Crossover...'.format(iter + 1), end='\r')
        children = crossover(parents, bag_of_words)
        if verbose:
            print('Iteration {}: Mutating individuals...'.format(iter + 1), end='\r')
        children = mutation(children, bag_of_words)
        individuals = children
    return individuals


def show_best_result(results: list):
    scores = [evaluate_fitness(result) for result in results]
    best_result = np.array([ind for _, ind in sorted(zip(scores, results), reverse=False)])[0]
    dist = character_distribution(best_result)
    plt.plot(dist)
    plt.show()


"""
========================================= End of Evolutionary Methods ==========================================
"""


if __name__ == '__main__':
    filepath = '../data/orig_wikipedia.txt'
    with open(filepath, 'r') as f:
        init = f.read()
    text = delete_duplicates(init)
    text = clean_text(text)
    text = delete_duplicates(text)
    bag_of_words = set(text.split())
    res = evolve(bag_of_words, False)
    results_filename = './dataset_content_res.pkl'
    with open(results_filename, 'wb') as output:
        pickle.dump(res, output, pickle.HIGHEST_PROTOCOL)
    show_best_result(res)
