import pandas
import time
from sklearn.neighbors import NearestNeighbors

print("Reading Data")
start_time = time.perf_counter()
input_data = pandas.read_csv("agp3k_data.csv", index_col=0)
input_data = input_data.loc[:, '1':] #Trims the data so the first column isn't included
read_time = time.perf_counter()-start_time

print("Clustering Data")
start_time = time.perf_counter()
neighbors = NearestNeighbors(n_neighbors = 5, radius= 1.0) #default values
neighbors.fit(input_data)
output = neighbors.kneighbors_graph(input_data)
cluster_time = time.perf_counter()-start_time

print("Writing Data")
start_time = time.perf_counter()
df_output = pandas.DataFrame(output.toarray())
df_output.to_csv("clustered_output.csv")
write_time = time.perf_counter() - start_time

print()
print("Runtimes:")
print("Read Data: %f seconds" % read_time)
print("Run Nearest Neighbors Approximation: %f seconds" % cluster_time)
print("Write Data: %f seconds" % write_time)
