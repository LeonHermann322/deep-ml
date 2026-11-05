import math

def disorder(apples: list) -> float:
	"""
	Compute the disorder in a basket of apples.
	"""
	counts = {}
	for apple in apples:
		counts.setdefault(apple, 0)
		counts[apple] += 1

	probabilities = {apple: count / len(apples) for apple, count in counts.items()}

	return 1 - sum([probability ** 2 for probability in counts.values()])