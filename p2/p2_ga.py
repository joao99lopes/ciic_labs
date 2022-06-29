import random
import pandas as pd
import os

from deap import base
from deap import creator
from deap import tools

filename = "Project2_DistancesMatrix.xlsx"
filepath = os.path.join(os.getcwd(), filename)
distances_df = pd.read_excel(filepath, index_col=[0])

ecopoints_to_visit = [1,2,5,6,8]
number_of_generations = 100

# CXPB  is the probability with which two individuals
#       are crossed
#
# MUTPB is the probability for mutating an individual
CXPB, MUTPB = 0.5, 0.2

def get_ecopoints_to_visit():
    return ecopoints_to_visit

def get_distance(src, dst):
    return distances_df[src][dst]

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
# Attribute generator 
toolbox.register("indices", get_ecopoints_to_visit)
# Structure initializers
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

ind1 = toolbox.individual()
print(ind1)
print(ind1.fitness.valid)

def evaluate(individual):
    summation = 0
    start = "C"
    for i in range(len(individual)):
        end = "E{}".format(individual[i])
        summation += distances_df[start][end]
        start = end
    summation += distances_df[start]["C"]
    print("evaluate", summation, len(individual)+2)
    return summation, len(individual)+2

toolbox.register("evaluate", evaluate)

ind1.fitness.values = evaluate(ind1)
print(ind1.fitness.valid)
print(ind1.fitness)

#toolbox.register("mate", tools.cxOrdered)
#toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.01)
#toolbox.register("select", tools.selTournament, tournsize=10)

"""
mutant = toolbox.clone(ind1)
ind2, = tools.mutGaussian(mutant, mu=0.0, sigma=0.2, indpb=0.2)
del mutant.fitness.values

print(ind2 is mutant)    # True
print(mutant is ind1)    # False

child1, child2 = [toolbox.clone(ind) for ind in (ind1, ind2)]
tools.cxBlend(child1, child2, 0.5)
del child1.fitness.values
del child2.fitness.values

selected = tools.selBest([child1, child2], 2)
print(child1 in selected)	# True

selected = toolbox.select(population, LAMBDA)
offspring = [toolbox.clone(ind) for ind in selected]

"""

toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evaluate)


#if __name__ == "__main__":
#    main()