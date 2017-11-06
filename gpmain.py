#numpy already allows the creation of arrays of all ones or all zeros very easily:
#
#e.g. np.ones((2, 2)) or np.zeros((2, 2))
#
#Since True and False are represented in Python as 1 and 0, respectively, we have only to cast this array as boolean using the optional dtype parameter and we are done.
#
#np.ones((2, 2), dtype=bool)
#
#returns:
#
#array([[ True,  True],
#       [ True,  True]], dtype=bool)


#numpy.full((2,2), True, dtype=bool)



import numpy as np
import random

from deap import base, creator, tools

IND_SIZE=1000

creator.create("FitnessMulti", base.Fitness, weights=(1.0,1.0))
creator.create("Individual", array.array, typecode='b', fitness=creator.FitnessMulti)


toolbox = base.Toolbox()
toolbox.register("attr_bool", random.randint, 0, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=IND_SIZE)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)










def arch_to_array(architecture):
    array = [0] * 60
    arch = json.loads(architecture)
    for j in range(0,5):
        for i in range(0, 12):
            if instruments[i] in arch[j]:
                array[i+j*12] = 1
    return array

# [0,1,1,0,0,0...] to ["BC","A","B","C","D"]

def array_to_arch(array):
    arch = '["'
    for j in range(0,5):
        for i in range(0, 12):
            if array[i+j*12] == 1:
                arch += instruments[i]
        if j < 4:
            arch += '","'
    arch += '"]'
    return arch

vassar = VASSAR.VASSAR()

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", array.array, typecode='b', fitness=creator.FitnessMax)

toolbox = base.Toolbox()

toolbox.register("attr_bool", random.randint, 0, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, 60)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def evaluate(individual):
    architecture = array_to_arch(individual)
    result = vassar.evaluateArch(architecture)
    # Maximize scientific benefit and minimize cost
    return result[0],-result[1]

toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=1/60)
toolbox.register("select", tools.selTournament, tournsize=3)

def main():
    print "globalExplorer"
    population = toolbox.population(n=300)

    NGEN=40
    for gen in range(NGEN):
        offspring = algorithms.varAnd(population, toolbox, cxpb=0.5, mutpb=0.1)
        fits = toolbox.map(toolbox.evaluate, offspring)
        for fit, ind in zip(fits, offspring):
            ind.fitness.values = fit
        population = offspring

    return population

if __name__ == '__main__':
    main()
