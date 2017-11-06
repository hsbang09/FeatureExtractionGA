
import numpy as np
import random

from deap import base, creator, tools
from scoop import futures, shared


NUM_FEATURES=None
NUM_EXAMPLES=None

FEATURES = []
LABELS = []
LABELS_CARDINALITY = None

# GA Parameters
NUM_GENERATION = 3
POPULATION_SIZE = 100



with open('/Users/bang/workspace/FeatureGA/data/baseFeatures','r') as f:    
    
    lines = f.readlines()
    NUM_FEATURES = len(lines)
    
    for l in lines:
        match = np.zeros((len(l),), dtype=bool)
        for ind,c in enumerate(l):
            if c=="1":
                match[ind]=True
        FEATURES.append(match)

        
with open('/Users/bang/workspace/FeatureGA/data/labels','r') as f:  
    
    line = f.readline()
    NUM_EXAMPLES = len(line)
    match = np.zeros((len(line),), dtype=bool)
    
    for ind,c in enumerate(line):
        if c=="1":
            match[ind]=True
    LABELS = match
    LABELS_CARDINALITY = np.sum(LABELS)

    
    
def evaluate(individual):
    
    nexample = shared.getConst('nexample')
    features = shared.getConst('features')
    labels = shared.getConst('labels')
    labels_card = shared.getConst('labels_card')

    # Create a boolean array
    matches = np.ones((nexample,), dtype=bool)
    
    # Get indices of the feature that are active in each individual
    featureIndices = np.nonzero(individual)
    
    for i in featureIndices:
        matches = [a and b for a, b in zip(features[i], matches)]
        
    S = labels_card
    F = np.sum(matches)
    SF = [a and b for a, b in zip(matches, labels)]
    total = nexample
    
    conf1 = SF / F  # Consistency (specificity)
    conf2 = SF / S  # Coverage    (Generality)
    
    return conf1, conf2    
    
    



def main():
    
    creator.create("FitnessMulti", base.Fitness, weights=(1.0,1.0))
    creator.create("Individual", array.array, typecode='b', fitness=creator.FitnessMulti)

    toolbox = base.Toolbox()
    toolbox.register("attr_bool", random.randint, 0, 1)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=NUM_FEATURES)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=1/NUM_FEATURES)
    toolbox.register("select", tools.selNSGA2)
    toolbox.register("map", futures.map)    
    

    shared.setConst(nfeat = NUM_FEATURES)
    shared.setConst(nexample = NUM_EXAMPLES)
    shared.setConst(features = FEATURES)
    shared.setConst(labels = LABELS)
    shared.setConst(labels_card = LABELS_CARDINALITY)
    

    population = toolbox.population(n=POPULATION_SIZE)
        
    for g in range(NUM_GENERATION):
        
        # Select the next generation individuals
        offspring = toolbox.select(population, POPULATION_SIZE)
        
        # Clone the selected individuals
        offspring = map(toolbox.clone, offspring)
        
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




if __name__ == '__main__':

    main()
