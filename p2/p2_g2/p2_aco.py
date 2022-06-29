import pants
import pandas as pd
import os
import sys
import time

filename = "Project2_DistancesMatrix.xlsx"
filepath = os.path.join(os.getcwd(), filename)
input_filename = sys.argv[1]
input_filepath = os.path.join(os.getcwd(), input_filename)

distances_df = pd.read_excel(filepath, index_col=[0])
input_df = pd.read_csv(input_filepath)

input_to_visit = input_df.columns.values.tolist()

def get_distance(src, dst):
    return distances_df[src][dst]

def get_path_from_input(input_list):
    res = ["C"]
    for i in input_list:
        res.append("E{}".format(i))
    return res

ecopoints_to_visit = get_path_from_input(input_to_visit)

start_time = time.time()

world = pants.World(ecopoints_to_visit, get_distance)

ants = []
n_ants = 10
for i in range(n_ants):
	ant = pants.Ant().initialize(world)
	ant.initialize(world, start=world.nodes[0])
	ants.append(ant)

solver = pants.Solver()

sol = solver.aco(ants)
solution = solver.solve(world)
solutions = solver.solutions(world)

print("Runtime: %s seconds" % round(time.time() - start_time))
print("Distance:", sol.distance,"km")
print("Route:", sol.tour)
