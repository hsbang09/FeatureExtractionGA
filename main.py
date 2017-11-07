import numpy as np
import random
import array
import math
import pickle

from deap import base, creator, tools

ROOT_PATH = '/Users/bang/workspace/FeatureExtractionGA'

RESULT_SAVE_PATH = ROOT_PATH + '/result/'

FEATURE_INPUT_DATA_PATH = ROOT_PATH + '/data/baseFeatures'
LABEL_INPUT_DATA_PATH = ROOT_PATH + '/data/labels'


NUM_FEATURES = None
NUM_EXAMPLES = None

FEATURES = []
LABELS = []
LABELS_CARDINALITY = None

# GA Parameters
NUM_GENERATION = 100
POPULATION_SIZE = 200
CXPB = 0.5
MUTPB = 0.1



with open(FEATURE_INPUT_DATA_PATH, 'r') as f:
    lines = f.readlines()
    NUM_FEATURES = len(lines)

    for l in lines:
        match = np.zeros((len(l),), dtype=bool)
        for ind, c in enumerate(l):
            if c == "1":
                match[ind] = True
        FEATURES.append(match)

with open(LABEL_INPUT_DATA_PATH, 'r') as f:
    line = f.readline()
    NUM_EXAMPLES = len(line)
    match = np.zeros((len(line),), dtype=bool)

    for ind, c in enumerate(line):
        if c == "1":
            match[ind] = True
    LABELS = match
    LABELS_CARDINALITY = np.sum(LABELS)


def evaluate(individual):
    # Create a boolean array
    matches = np.ones((NUM_EXAMPLES,), dtype=bool)

    # Get indices of the feature that are active in each individual
    features = np.nonzero(individual)[0]

    for i in features:
        matches = [a and b for a, b in zip(FEATURES[i], matches)]

    S = LABELS_CARDINALITY
    F = np.sum(matches)
    SF = np.sum([a and b for a, b in zip(matches, LABELS)])
    total = NUM_EXAMPLES

    if F == 0:
        conf1 = 0
    else:
        conf1 = SF / F  # Consistency (specificity)
    conf2 = SF / S  # Coverage    (Generality)

    return conf1, conf2


def randBool(num_features):
    prob = 1 / num_features
    return random.random() < prob


def distance_to_UP(fitness_values):
    out = 0
    for i in range(len(fitness_values)):
        out += math.pow(1-fitness_values[i],2)
    return math.sqrt(out)


creator.create("FitnessMulti", base.Fitness, weights=(1.0, 1.0))
creator.create("Individual", array.array, typecode='b', fitness=creator.FitnessMulti)

toolbox = base.Toolbox()
toolbox.register("attr_bool", randBool, NUM_FEATURES)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=NUM_FEATURES)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=1 / NUM_FEATURES)
toolbox.register("select", tools.selNSGA2)






def main():
    
    population = toolbox.population(n=POPULATION_SIZE)
    
    best_scores = []

    for g in range(NUM_GENERATION):

        # Select the next generation individuals
        offspring = toolbox.select(population, POPULATION_SIZE)

        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))

        # TODO: Implement binary tournament selection

        # Apply crossover on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        # Apply mutation on the offspring
        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitness = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitness):
            ind.fitness.values = fit

        # The population is entirely replaced by the offspring
        population[:] = offspring
        
        
        # Save information
        shortest_distance = 10000
        for individual in population:
            dist = distance_to_UP(individual.fitness.values)
            if shortest_distance > dist:
                shortest_distance = dist
        best_scores.append(shortest_distance)
        print("Generation: {0}. Best Score: {1}".format(g, shortest_distance))              
              
        if (g+1) % 20 == 0:
              
            feature_save_path = RESULT_SAVE_PATH + str(g+1) + "_feature"
            metrics_save_path = RESULT_SAVE_PATH + str(g+1) + "_metric"

            with open(metrics_save_path, 'w') as f:

                metrics = []
                for individual in population:
                    row = []
                    for val in individual.fitness.values:
                        row.append(str(val))

                    metrics.append(",".join(row))

                f.write("\n".join(metrics))              


            with open(feature_save_path, 'w') as f:

                feature_indices = []
                for individual in population:
                    indices = []
                    for ind, bit in enumerate(individual):
                        if bit:
                            indices.append(str(ind))
                        else:
                            pass
                    if indices:
                        feature_indices.append(",".join(indices))

                f.write("\n".join(feature_indices))



if __name__ == '__main__':
    main()
