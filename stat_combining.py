import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import linregress
from matplotlib.cm import get_cmap


### multitemporal color image
column_names = ['order', 'Q_decile', 'mu', 'sigma', 'n']

glow_mt = pd.read_csv('glow_mt_parms.csv')[column_names]
glow_mt = glow_mt.loc[glow_mt.order >= 10]

platte_mt = pd.read_csv('platte_mt_parms.csv')[column_names]
platte_mt = platte_mt.loc[platte_mt.order <= 9]

parms = pd.concat([glow_mt, platte_mt])
print(glow_mt.n.sum())
print(platte_mt.n.sum())
mus = []
sigmas = []
qs = []
os = []
for q in range(10):
    for o in [7, 8, 9, 10, 11, 12, 13, 14]:
        mu = parms.loc[(parms.Q_decile == q) & (parms.order == o), 'mu']
        sigma = parms.loc[(parms.Q_decile == q) & (parms.order == o), 'sigma']

        mus.append(mu)
        sigmas.append(sigma)
        qs.append(q)
        os.append(o)





os_axis = [7, 8, 9, 10, 11, 12, 13, 14]
qs_axis = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]


musarray = np.array(mus.copy())
musarray.shape = (10, 8)

sigmasarray = np.array(sigmas.copy())
sigmasarray.shape = (10, 8)

fig, (mu_img, sigma_img) = plt.subplots(1, 2, constrained_layout=True)
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


### scatter plot
cmap = get_cmap('tab10')
fig, (axmu, axsig) = plt.subplots(nrows=2, constrained_layout=False)
for o in [7, 8, 9, 10, 11, 12, 13, 14]:
    percline = parms.loc[parms.order == o, 'mu']

    axmu.scatter(range(10), percline, c=cmap(o - 7), label=o)
    s, i, r, p, _ = linregress(range(10), percline)
    axmu.set_ylabel(r'$\mu$')
    axmu.set_xticks([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5])
    axmu.set_xticklabels([])
    axmu.set_ylim(-1.5, 7.5)
    # axmu.set_xlim(0, 10)
    axmu.legend(title='Stream Order')
    if p < 0.05:
        if s < 0:
            ls = '--'
        else:
            ls = '-'
        axmu.plot(np.linspace(0, 9), np.linspace(0, 9) * s + i, c=cmap(o - 7), linestyle=ls)
    # else: 
    #     print('not significant')
    percline = parms.loc[parms.order == o, 'sigma']

    axsig.scatter(range(10), percline, c=cmap(o - 7))
    axsig.set_xticks([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5])
    axsig.set_xticklabels([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    axsig.set_xlabel('Flow percentile')
    axsig.set_ylabel(r'$\sigma$')
    axsig.set_ylim(0, 1.35)
    s, i, r, p, _ = linregress(range(10), percline)

    if p < 0.05:
        if s < 0:
            ls = '--'
        else:
            ls = '-'
        axsig.plot(np.linspace(0, 9), np.linspace(0, 9) * s + i, c=cmap(o - 7), linestyle=ls)


plt.show()


### mean width scatter

carterstats = pd.DataFrame({
    'order':    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
    'sigma':    [0.784, 0.736, 0.740, 0.759, 0.512, 0.715, 0.426, 0.337, 0.982, 0.595, 0.741, 0.563, 0.494],
    'mu':       [-0.961, -0.415, 0.202, 1.085, 1.628, 1.830, 2.386, 3.046, 3.748, 3.905, 5.335, 6.003, 6.744]})
# carterstats = pd.DataFrame({
#     'order':    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
#     'sigma':    [0.8, 0.7, 0.7, 0.8, 0.5, 0.7, 0.4, 1.0, 1.0, 0.6, 0.7, 0.6, 0.5],
#     'mu':       [-0.961, -0.415, 0.202, 1.085, 1.628, 1.830, 2.386, 3.046, 3.748, 3.905, 5.335, 6.003, 6.744]})
# carterstats = carterstats.loc[(carterstats.order >= 7) & (carterstats.order <= 14)]

cbslope, cbint, cbr, cbp, _ = linregress(carterstats.order, carterstats.mu)
cbsigmean = np.mean(carterstats.sigma)
cbsigstd = np.std(carterstats.sigma)
print(cbslope)
print(cbint)
print(cbr)


glow = pd.read_csv('glow_mean_parms.csv')
glow = glow.loc[glow.order >= 10]

platte = pd.read_csv('platte_mean_parms.csv')
platte = platte.loc[platte.order <= 9]

meanparms = pd.concat([glow, platte])

fig, (axmu, axsig) = plt.subplots(nrows=2, constrained_layout=False)


axmu.scatter(meanparms.order, meanparms.mu, label='This study')
s, i, r, p, _ = linregress(meanparms.order, meanparms.mu)
print(s)
print(i)
print(r)
axmu.set_ylabel(r'$\mu$')
axmu.set_xlim(0, 15)
axmu.set_xticklabels([])
axmu.set_ylim(-1.5, 7.5)
axmu.axline((0, i), slope=s)
axmu.scatter(carterstats.order, carterstats.mu, label='Boyd & Allen, 2025')
axmu.axline((0, cbint), slope=cbslope, c='tab:orange')
axmu.legend()

axsig.scatter(meanparms.order, meanparms.sigma)
axsig.set_xlabel(r'$\omega$')
axsig.set_xlim(0, 15)
axsig.set_ylabel(r'$\sigma$')
axsig.set_ylim(0, 1.35)
axsig.scatter(carterstats.order, carterstats.sigma)
s, i, r, p, _ = linregress(meanparms.order, meanparms.sigma)
axsig.axline((0, i), slope=s, linestyle='--')
print('    ')
print(cbsigstd)
print(np.std(meanparms.sigma))
print('    ')
print(s)
print(i)
print(r)
print(p)

plt.show()