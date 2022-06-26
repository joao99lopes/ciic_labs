import pants
import math
import pandas as pd
import numpy as np

df = pd.read_excel(r"C:/Users/Aurora/Desktop/Project2_DistancesMatrix.xlsx")
df = df.to_numpy() #converting to matrix ij = distance
df = np.delete(df,0,1) #deleting row string names to only have the values

def getDistance(ep1, ep2): #get distance between 2 ecopoints from the matrix
    return math.sqrt(pow(df[0][ep1] - df[0][ep2], 2) + pow(df[ep1][0] - df[ep2][0], 2))

starting_node = [0]
#ecopoints_list_test = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,
#                    42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,
#                   81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,0]

ecopoints_list_test = [0,1,2,3]

world = pants.World(ecopoints_list_test, getDistance)
world.data(ecopoints_list_test[0])

solver = pants.Solver()

solution = solver.solve(world)
solutions = solver.solutions(world)

best = float("inf")
for solution in solutions:
  assert solution.distance < best
  best = solution.distance

print("distance:", solution.distance)
print("path:", solution.tour)    # Nodes visited in order