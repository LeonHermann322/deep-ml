import copy
import math
import numpy as np

def k_means_clustering(points: list[tuple[float, ...]], k: int, initial_centroids: list[tuple[float, ...]], max_iterations: int) -> list[tuple[float, ...]]:
	centroids = initial_centroids
	points_mapping = {point: -1 for point in points}

	def find_nearest_centroid_idx(point):

		distances = [math.sqrt(sum([(centroid[i] - point[i])**2 for i in range(len(point))])) for centroid in centroids]

		min_distance = min(distances)
		min_idx = distances.index(min_distance)

		return min_idx

	for iteration in range(max_iterations):
		old_mapping = copy.deepcopy(points_mapping)
		# Add point to nearest centroid cluster
		for point in points_mapping.keys():
			points_mapping[point] = find_nearest_centroid_idx(point)

		if points_mapping == old_mapping:
			return centroids

		# Recalculate centroids
		for i in range(k):
			points = [point for point, centroid_idx in points_mapping.items() if centroid_idx == i]
			mean = np.asarray(points).mean(axis=0)
			centroids[i] = mean.tolist()

		centroids = [tuple(centroid) for centroid in centroids]

	return centroids