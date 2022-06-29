from deap import creator, base, tools, algorithms
import os
import pandas as pd
import numpy as np
import sys
import time


filename = "Project2_DistancesMatrix.xlsx"
filepath = os.path.join(os.getcwd(), filename)
input_filename = sys.argv[1]
input_filepath = os.path.join(os.getcwd(), input_filename)

distances_df = pd.read_excel(filepath, index_col=[0])
input_df = pd.read_csv(input_filepath)

ecopoints_to_visit = input_df.columns.values.tolist()

def get_distance(src, dst):
    return distances_df[src][dst]

def get_path_distance(individual):
    path = get_path_from_individual(individual)
    res = get_distance("C", "E{}".format(path[0]))
    for i in range(len(individual) - 1):
        src = "E{}".format(path[i])
        dst = "E{}".format(path[i+1])
        res += get_distance(src,dst)
    res += get_distance("E{}".format(path[-1]), "C")
    return res,

def get_path_from_individual(individual):
    res = []
    for i in individual:
        res.append(ecopoints_to_visit[i])
    return res

def show_hof(hof_list):
    res = ["C"]
    for i in hof_list:
        res.append("E{}".format(ecopoints_to_visit[i]))
    res.append("C")
    return res

start_time = time.time()

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Structure", list, fitness=creator.FitnessMin)

n = len(ecopoints_to_visit)
toolbox = base.Toolbox()
toolbox.register("EcoPoint", np.random.permutation, n)
toolbox.register("Individual", tools.initIterate, creator.Structure, toolbox.EcoPoint)
toolbox.register("Population", tools.initRepeat, list, toolbox.Individual)

pop = toolbox.Population(n=1000)

toolbox.register("mate", tools.cxPartialyMatched)
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", get_path_distance)

hof = tools.HallOfFame(1)
result, log = algorithms.eaSimple(population=pop,
                                toolbox=toolbox,
                                cxpb=0.7,
                                mutpb=0.2,
                                ngen=400,
                                halloffame=hof,
                                verbose=True)


print("Runtime: %s seconds" % round(time.time() - start_time))
print("Distance:",get_path_distance(hof[0])[0],"km")
print("Route:",show_hof(hof[0]))

