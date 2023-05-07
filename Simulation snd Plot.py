import numpy as np
import matplotlib.pyplot as plt
from scipy.special import logsumexp
bins = np.array([0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]) # call the eccentricity bins
true = np.array([0.4, 0.22, 0.12, 0.1, 0.08, 0.02, 0.008, 0.004, 0.04, 0.02, 0.01, 0.02, 0.002, 0.009, 0.005, 0.004, 0.005, 0.01, 0.002, 0.001]) # call the true categorical distribution
true /= np.sum(true)
null = np.array([0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]) # call the distribution with equal possibilities
null /= np.sum(null)
n = 400 #number of exoplanets we want to survey
cate = np.random.choice(bins[:-1], size=n, p=true) # take samples from the categorical distribution
plt.hist(cate, bins=bins, density=False, alpha=0.5, edgecolor='black') # Plot the histogram of the samples
plt.xlabel('Eccentricity')
plt.ylabel('Count')
plt.show()
true_likelihood = np.prod([true[bins[:-1] == x][0] for x in cate]) # calculate and print the log likelihood ratio between the true model and the null model
null_likelihood = np.prod([null[bins[:-1] == x][0] for x in cate])
log_likelihood_ratio = logsumexp([np.log(true[bins[:-1] == x][0]) - np.log(null[bins[:-1] == x][0]) for x in cate])
print("Log Likelihood ratio:", log_likelihood_ratio)
fig, ax = plt.subplots() # Plot the measured eccentricities against the true eccentricities
ax.scatter(cate, np.random.normal(cate, 0.02), alpha=0.5)
ax.plot(bins[:-1], bins[:-1], 'k--')
ax.set_xlabel('True Eccentricity')
ax.set_ylabel('Measured Eccentricity')
plt.show()
