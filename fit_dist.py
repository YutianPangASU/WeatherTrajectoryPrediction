import numpy as np
import seaborn as sns
from scipy.stats import norm
import matplotlib.pyplot as plt


data = norm.rvs(5,0.4,size=1000) # you can use a pandas series or a list if you want

sns.distplot(data)

dev = np.load('pred_dev.npy')

sector = 'ZID'

det = np.load('testset_{}_l2.npy'.format(sector))
drop = np.load('drop_{}_l2.npy'.format(sector))
flipout = np.load('det_{}_l2.npy'.format(sector))
# repara = np.load('repara_{}_l2.npy'.format(sector))
ori = np.load('flipout_{}_l2.npy'.format(sector))

data = ori

# Fit a normal distribution to the data:
mu, std = norm.fit(data)

# Plot the histogram.
plt.hist(ori, bins=25, density=True, alpha=0.3, color='green', label='ori')
plt.hist(drop, bins=25, density=True, alpha=0.6, color='red', label='dropout')
plt.hist(det, bins=25, density=True, alpha=0.6, color='black', label='deterministic')
# plt.hist(repara, bins=25, density=True, alpha=0.6, color='blue', label='reparameterization')
plt.hist(flipout, bins=25, density=True, alpha=0.6, color='yellow', label='flipout')

# Plot the PDF.
plt.xlim(0, 1)
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu, std)
#plt.plot(x, p, 'k', linewidth=2)
# title = "Fit results: mu = %.2f,  std = %.2f" % (mu, std)
# plt.title(title)
plt.title('Deviation Reduction')
plt.xlabel('Deviation/Degrees')
plt.ylabel('Density')
plt.legend()
plt.show()
#plt.savefig('dist_plot_3d_{}.png'.format(sector))




