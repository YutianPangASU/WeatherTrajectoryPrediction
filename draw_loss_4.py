import numpy as np
import matplotlib.pyplot as plt


f1 = np.loadtxt(open('Det_20190624_ZTL_epoch_500/loss_logs.csv'), delimiter=",", skiprows=1)
f2 = np.loadtxt(open('Drop_20190624_ZTL_epoch_200/loss_logs.csv'), delimiter=",", skiprows=1)
f3 = np.loadtxt(open('Flipout_20190624_ZTL_epoch_500/loss_logs.csv'), delimiter=",", skiprows=1)
f4 = np.loadtxt(open('Repara_20190624_ZTL_epoch_500/loss_logs.csv'), delimiter=",", skiprows=1)

ns = 0
ne = 50
x = range(ne-ns)
plt.figure()
plt.title('Convergence Comparasion (ZTL)')
plt.plot(x, f1[ns:ne, 1], 'blue', label='Deterministic')
plt.plot(x, f2[ns:ne, 1], 'red', label='Dropout')
plt.plot(x, f3[ns:ne, 1], 'black', label='Flipout')
plt.plot(x, f4[ns:ne, 1], 'green', label='Reparametrization')
plt.legend()
#plt.show()
plt.savefig("convergence_compare_ZTL_{}_epoch.png".format(ne-ns))


f1 = np.loadtxt(open('Det_20190624_ZID_epoch_200/loss_logs.csv'), delimiter=",", skiprows=1)
f2 = np.loadtxt(open('Drop_20190624_ZID_epoch_200/loss_logs.csv'), delimiter=",", skiprows=1)
f3 = np.loadtxt(open('Flipout_20190624_ZID_epoch_200/loss_logs.csv'), delimiter=",", skiprows=1)
f4 = np.loadtxt(open('Repara_20190624_ZID_epoch_200/loss_logs.csv'), delimiter=",", skiprows=1)


x = range(ne-ns)
plt.figure()
plt.title('Convergence Comparasion (ZID)')
plt.plot(x, f1[ns:ne, 1], 'blue', label='Deterministic')
plt.plot(x, f2[ns:ne, 1], 'red', label='Dropout')
plt.plot(x, f3[ns:ne, 1], 'black', label='Flipout')
plt.plot(x, f4[ns:ne, 1], 'green', label='Reparametrization')
plt.legend()
#plt.show()
plt.savefig("convergence_compare_ZID_{}_epoch.png".format(ne-ns))

