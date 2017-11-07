import numpy as np
import random
import array
import math
import pickle

from deap import base, creator, tools

RESULT_SAVE_PATH = '/Users/bang/workspace/FeatureExtractionGA/result/result'
STATS_SAVE_PATH = '/Users/bang/workspace/FeatureExtractionGA/result/stats'

FEATURE_DATA_PATH = '/Users/bang/workspace/FeatureExtractionGA/data/baseFeatures'
LABEL_DATA_PATH = '/Users/bang/workspace/FeatureExtractionGA/data/labels'


NUM_FEATURES = None
NUM_EXAMPLES = None

FEATURES = []
LABELS = []
LABELS_CARDINALITY = None

# GA Parameters
NUM_GENERATION = 100
POPULATION_SIZE = 100
CXPB = 0.5
MUTPB = 0.1



with open(FEATURE_DATA_PATH, 'r') as f:
    lines = f.readlines()
    NUM_FEATURES = len(lines)

    for l in lines:
        match = np.zeros((len(l),), dtype=bool)
        for ind, c in enumerate(l):
            if c == "1":
                match[ind] = True
        FEATURES.append(match)

with open(LABEL_DATA_PATH, 'r') as f:
    line = f.readline()
    NUM_EXAMPLES = len(line)
    match = np.zeros((len(line),), dtype=bool)

    for ind, c in enumerate(line):
        if c == "1":
            match[ind] = True
    LABELS = match
    LABELS_CARDINALITY = np.sum(LABELS)


def evaluate(individual):
    # TODO: Implement evaluation

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


#stats = tools.Statistics(key=lambda ind: ind.fitness.values)
#stats_size = tools.Statistics(key=len)
#mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)

#stats.register("consistency_avg", np.max, axis=0)
#stats.register("consistency_std", np.std, axis=0)
#stats.register("coverage_avg", np.max, axis=1)
#stats.register("coverage_std", np.std, axis=1)
#stats.register("std", np.std, axis=0)
#stats.register("min", np.min, axis=0)
#stats.register("max", np.max, axis=0)

#logbook = tools.Logbook()



def main():
    
    population = toolbox.population(n=POPULATION_SIZE)

    for g in range(NUM_GENERATION):

        print("Generation: {0}".format(g))

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
        
        
        
        

        # Record statistics
#        record = stats.compile(population)
#        consistency_max = record['consistency_max'][0]
#        coverage_max = record['coverage_max'][0]
#        logbook.record(gen=g, consistency_max=consistency_max, coverage_max=coverage_max)
        
#    print(logbook)

    
    
    

    with open(RESULT_SAVE_PATH, 'w') as resultFile:

        population = toolbox.select(population, POPULATION_SIZE)
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

        resultFile.write("\n".join(feature_indices))

        
        
    # with open(STATS_SAVE_PATH, 'w') as f:
    #     gen, consistency_max, coverage_max = logbook.select("gen", "consistency_max", "coverage_max")
    #     content = []
    #     for i in range(len(gen)):
    #         row = [str(gen[i]), str(consistency_max[i]), str(coverage_max[i])]
    #         content.append(",".join(row))
    #
    #     f.write("\n".join(content))
        
        

if __name__ == '__main__':
    main()
