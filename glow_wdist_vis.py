import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors as mcolors

ln_parms = pd.read_csv(r"c:\Users\dego\Desktop\lognormal_fits_duckdb2.csv")

ln_parms['mu'] = np.log(ln_parms['scale'])
ln_parms = ln_parms.rename(columns={'shape': 'sigma'})

# build sigma and mu arrays
mus = []
sigmas = []
os = []
qs = []
ns = []
for q in np.linspace(0, 9, 10):
    for o in np.linspace(7, 14, 8):
        # print(q, " ", o)

        mu =    ln_parms.loc[(ln_parms['Q_decile'] == q) & (ln_parms['order'] == o)].reset_index().loc[0, 'mu']
        sigma = ln_parms.loc[(ln_parms['Q_decile'] == q) & (ln_parms['order'] == o)].reset_index().loc[0, 'sigma']
        n =     ln_parms.loc[(ln_parms['Q_decile'] == q) & (ln_parms['order'] == o)].reset_index().loc[0, 'n']
        
        mus.append(mu)
        sigmas.append(sigma)
        os.append(o)
        qs.append(q)
        ns.append(n)

print(mus)
# graph
decile_bins = np.linspace(0, 1, 11)

osarray = np.array(os.copy())

osarray.shape = (10, 8)

os_axis = osarray[1]
os_axis = [int(os_axis[o]) for o in range(len(os_axis))]
qsarray = np.array(qs.copy())
qsarray.shape = (10, 8)
qs_axis = qsarray[:, 1]
qs_axis = [int(decile_bins[d] * 100) for d in range(len(decile_bins))]


musarray = np.array(mus.copy())
musarray.shape = (10, 8)

sigmasarray = np.array(sigmas.copy())
sigmasarray.shape = (10, 8)

fig, (mu_img, sigma_img) = plt.subplots(1, 2, constrained_layout=True, figsize=(8, 8))
mshow = mu_img.imshow(np.flip(musarray, axis=0))
mu_img.set_xticks([0, 1, 2, 3, 4, 5, 6, 7])
mu_img.set_xticklabels(os_axis)

mu_img.set_yticks([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5])
mu_img.set_yticklabels(np.flip(qs_axis))
mu_img.set_title(r'$\mu$')

im_ratio = musarray.shape[0]/musarray.shape[1]

plt.colorbar(mshow, orientation="vertical", fraction=0.05*im_ratio, ax=mu_img, pad=0.05)


sshow = sigma_img.imshow(np.flip(sigmasarray, axis=0))
sigma_img.set_xticks([0, 1, 2, 3, 4, 5, 6, 7])
sigma_img.set_xticklabels(os_axis)

# fig.set_constrained_layout_pads(w_pad=1./72, h_pad=1./72, wspace=0.01, hspace=0.01)

# sigma_img.set_yticks([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5])
sigma_img.yaxis.set_visible(False)
sigma_img.set_title(r'$\sigma$')

plt.colorbar(sshow, orientation="vertical", fraction=0.05*im_ratio, ax=sigma_img, pad=0.05)

fig.supxlabel('Stream order')
fig.supylabel('Flow percentile')
plt.show()