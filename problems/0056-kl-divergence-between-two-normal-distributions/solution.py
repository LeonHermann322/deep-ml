import numpy as np

def kl_divergence_normal(mu_p, sigma_p, mu_q, sigma_q):
	return np.log(sigma_q / sigma_p) - 0.5 + (np.square(sigma_p) + np.square(mu_p - mu_q)) / (2 * np.square(sigma_q))
