### Linear regresssions of mu and sigma wrt flow decile are uncertain
### this file has functions to randomly sample mu and sigma from distributions
### and functions to generate RSSA estimates for monte carlo simulation to estimate uncertainty

import pandas as pd
import numpy as np
from scipy.stats import linregress, lognorm
import statsmodels.api as sm
from tqdm import tqdm
from matplotlib import pyplot as plt


def compile_width_params():
    barefoot = pd.read_csv('barefoot_order_mt_parms_20260315.csv')
    s2 = pd.read_csv('platte_order_mt_parms_20260315.csv')
    glow = pd.read_csv('glow_dparms_x_20260323.csv')

    # print(np.unique(s2['order']))
    # print(np.unique(glow['order']))

    barefoot = barefoot.rename(columns={'mu': 'barefoot_mu', 'sigma': 'barefoot_sigma'})
    s2 = s2.rename(columns={'mu': 'platte_mu', 'sigma': 'platte_sigma'})
    glow = glow.rename(columns={'mu': 'glow_mu', 'sigma': 'glow_sigma'})
    s2['order'] = s2['order'] - 1
    print(np.unique(s2['order']))
    glow['order'] = glow['order'] + 3
    print(np.unique(glow['order']))
    statdf = barefoot[['Q_decile', 'order', 'barefoot_mu', 'barefoot_sigma']]\
        .merge(s2[['Q_decile', 'order', 'platte_mu', 'platte_sigma']], on=['Q_decile', 'order'], how='outer')\
        .merge(glow[['Q_decile', 'order', 'glow_mu', 'glow_sigma']], on=['Q_decile', 'order'], how='outer')
    
    return statdf


def lr_width_params(statdf):

    platte_orders = [6, 7, 8]
    glow_orders = [9, 10, 11, 12, 13]

    muslopes = []
    muyints = []
    sigmaslopes = []
    sigmayints = []

    orders = np.arange(1, 15, 1)
    pval = None
    ### mu
    for n in range(len(orders)):
        order = orders[n]
        df = statdf.loc[statdf['order'] == order]

        ## Platte S2 widths
        if order in (platte_orders):
            x = df['Q_decile'].to_numpy()
            y = df['platte_mu'].to_numpy()

            X = sm.add_constant(x)
            
            results = sm.OLS(y, X).fit()
            yint, slope = results.params
            pval = results.pvalues[1]
            
            if pval < 0.05:
                muslopes.append(slope)
                muyints.append(yint)
            else:
                muslopes.append(np.nan)
                muyints.append(np.nan)
            
            
        ## GLOW Mississippi basin widths
        elif order in (glow_orders):
            x = df['Q_decile'].to_numpy()
            y = df['glow_mu'].to_numpy()

            X = sm.add_constant(x)
            
            results = sm.OLS(y, X).fit()
            yint, slope = results.params
            pval = results.pvalues[1]
            
            if pval < 0.05:
                muslopes.append(slope)
                muyints.append(yint)
            else:
                muslopes.append(np.nan)
                muyints.append(np.nan)
            
        else:
            muslopes.append(np.nan)
            muyints.append(np.nan)

    pval = None
    ### sigma
    for n in range(len(orders)):
        order = orders[n]
        df = statdf.loc[statdf['order'] == order]

        ## Platte S2 widths
        if order in (platte_orders):
            x = df['Q_decile'].to_numpy()
            y = df['platte_sigma'].to_numpy()

            X = sm.add_constant(x)
            
            results = sm.OLS(y, X).fit()
            yint, slope = results.params
            pval = results.pvalues[1]
            
            if pval < 0.05:
                sigmaslopes.append(slope)
                sigmayints.append(yint)
            else:
                sigmaslopes.append(np.nan)
                sigmayints.append(np.nan)
            
            
        ## GLOW Mississippi basin widths
        elif order in (glow_orders):
            x = df['Q_decile'].to_numpy()
            y = df['glow_sigma'].to_numpy()

            X = sm.add_constant(x)
            
            results = sm.OLS(y, X).fit()
            yint, slope = results.params
            pval = results.pvalues[1]
            
            if pval < 0.05:
                sigmaslopes.append(slope)
                sigmayints.append(yint)
            else:
                sigmaslopes.append(np.nan)
                sigmayints.append(np.nan)
            
        else:
            sigmaslopes.append(np.nan)
            sigmayints.append(np.nan)

    # extrapolate lr relations to headwaters, sample for large rivers
    # orders = np.array([6, 7, 8, 9, 10, 11, 12, 13])
    sigmaslopes = np.array(sigmaslopes)
    muslopes = np.array(muslopes)

    sigmaslope_mean = np.nanmean(sigmaslopes)
    sigmaslope_std = np.nanstd(sigmaslopes)

    muslope_mean = np.nanmean(muslopes)
    muslope_std = np.nanstd(muslopes)


    muyints = np.array(muyints)
    sigmayints = np.array(sigmayints)

    sigmayint_mean = np.nanmean(sigmayints)
    sigmayint_std = np.nanstd(sigmayints)

    lr = linregress(orders[np.isfinite(muyints)], muyints[np.isfinite(muyints)])

    muyint_slope = lr[0]
    muyint_yint = lr[1]
    
    dparm_stats = (orders, sigmaslopes, muslopes, sigmaslope_mean, sigmaslope_std, muslope_mean, muslope_std, muyints, sigmayints, sigmayint_mean, sigmayint_std, muyint_slope, muyint_yint)
    return dparm_stats


def random_extrapolate_width_params(muslope_mean, muslope_std, sigmaslope_mean, sigmaslope_std, muyint_yint, muyint_slope, sigmayint_mean, sigmayint_std,):
    predicted_muslopes = []
    predicted_sigmaslopes = []
    predicted_muyints = []
    predicted_sigmayints = []


    for order in np.arange(1, 15, 1):
        predicted_muslopes.append(np.random.normal(loc=muslope_mean, scale=muslope_std))
        predicted_sigmaslopes.append(np.random.normal(loc=sigmaslope_mean, scale=sigmaslope_std))
        predicted_muyints.append(muyint_yint + (order * muyint_slope))
        predicted_sigmayints.append(np.random.normal(loc=sigmayint_mean, scale=sigmayint_std))

    dparm_predictors = pd.DataFrame({
        'order': np.arange(1, 15, 1),
        'muslope': predicted_muslopes,
        'sigmaslope': predicted_sigmaslopes,
        'muyint': predicted_muyints,
        'sigmayint': predicted_sigmayints,
    })

    orders = []
    Qds = []
    predicted_mus = []
    predicted_sigmas = []
    for order in np.arange(1, 15, 1):
        row = dparm_predictors.loc[dparm_predictors['order'] == order]
        _, muslope, sigmaslope, muyint, sigmayint = row.values[0]

        for Qd in np.arange(0, 10, 1):
            orders.append(order)
            Qds.append(Qd)

            predicted_mu = (muyint + (Qd * muslope))
            predicted_sigma = (sigmayint + (Qd * sigmaslope))
            predicted_mus.append(predicted_mu)
            predicted_sigmas.append(predicted_sigma)

    predicted_dparms = pd.DataFrame({
        'order': orders,
        'Q_decile': Qds,
        'mu': predicted_mus,
        'sigma': predicted_sigmas
    })
    # predicted_dparms.to_csv('C:/Users/dego/Downloads/pdp.csv')

    return predicted_dparms


def import_length():
    # return pd.read_csv('C:/Users/dego/Downloads/demlens_20260520.csv')
    # return pd.read_csv('C:/Users/dego/Documents/GitHub/multitemporal_RSSA/dem_lens_melt.csv')
    # return pd.read_csv('C:/Users/dego/Documents/GitHub/multitemporal_RSSA/arcdemlens.csv')
    # return pd.read_csv(r'C:\Users\dego\Documents\GitHub\multitemporal_RSSA\lengths_approach4_20260615.csv')
    # return pd.read_csv(r'C:\Users\dego\Documents\GitHub\multitemporal_RSSA\merit_lendf_20260616.csv')
    return pd.read_csv(r'C:\Users\dego\Documents\GitHub\multitemporal_RSSA\lengths_approach5_20260722.csv')


def compute_RSSA_df(width_dparms, length_df, fixed_width=False, fixed_length=False):
    lenlist = []
    meanwidth_list = []
    sumwidth_list = []
    area_km2_list = []
    Qdlist = []
    orderlist = []
    mulist = []
    sigmalist = []

    for Qd in np.arange(0, 10, 1):
        if fixed_length:
            active_orders = length_df.loc[(length_df['Q_decile'] == 4), 'order'].to_numpy()
        else:
            active_orders = length_df.loc[(length_df['Q_decile'] == Qd), 'order'].to_numpy()

        for order in active_orders:
            if fixed_width:
                mu = width_dparms.loc[
                    (width_dparms['order'] == order) & 
                    (width_dparms['Q_decile'] == 4),
                    'mu'].values[0]

                sigma = width_dparms.loc[
                    (width_dparms['order'] == order) & 
                    (width_dparms['Q_decile'] == 4),
                    'sigma'].values[0]
                
            else:
                mu = width_dparms.loc[
                    (width_dparms['order'] == order) & 
                    (width_dparms['Q_decile'] == Qd),
                    'mu'].values[0]

                sigma = width_dparms.loc[
                    (width_dparms['order'] == order) & 
                    (width_dparms['Q_decile'] == Qd),
                    'sigma'].values[0]
                

            if fixed_length:
                length_km = length_df.loc[
                    (length_df['order'] == order) & 
                    (length_df['Q_decile'] == 4),
                    'length_km'].values[0]
            else:
                length_km = length_df.loc[
                    (length_df['order'] == order) & 
                    (length_df['Q_decile'] == Qd),
                    'length_km'].values[0]


            meanwidth = np.exp(mu + (sigma ** 2) / 2)
            area_km2 = meanwidth * length_km * 1e-3

            area_km2_list.append(area_km2)
            lenlist.append(length_km)
            meanwidth_list.append(meanwidth)
            Qdlist.append(Qd)
            orderlist.append(order)
            mulist.append(mu)
            sigmalist.append(sigma)

    
    RSSA_df = pd.DataFrame({
        'area_km2': area_km2_list,
        'length_km': lenlist,
        'mean_width': meanwidth_list,
        'Q_decile': Qdlist,
        'order': orderlist,
        'mu': mulist,
        'sigma': sigmalist
    })

    return RSSA_df


def RSSA_df_wrapper(dparm_stats, length_df, fixed_width=False, fixed_length=False):
    muslope_mean = dparm_stats[5]
    muslope_std = dparm_stats[6]
    sigmaslope_mean = dparm_stats[3]
    sigmaslope_std = dparm_stats[4]
    muyint_yint = dparm_stats[12]
    muyint_slope = dparm_stats[11]
    sigmayint_mean = dparm_stats[9]
    sigmayint_std = dparm_stats[10]

    orders = length_df.sort_values('order')['order'].unique()

    predicted_dparms = random_extrapolate_width_params(muslope_mean, 
                                                       muslope_std, 
                                                       sigmaslope_mean, 
                                                       sigmaslope_std, 
                                                       muyint_yint, 
                                                       muyint_slope, 
                                                       sigmayint_mean, 
                                                       sigmayint_std)
            
    RSSA_df = compute_RSSA_df(predicted_dparms, length_df, fixed_width, fixed_length)

    return RSSA_df

if __name__ == "__main__":

    n_iterations = 1000

    length_df = import_length()

    bigriver_dparms = compile_width_params()

    dparm_stats = lr_width_params(bigriver_dparms)

    # length and width
    RSSA_df = RSSA_df_wrapper(dparm_stats, length_df, fixed_length=False)
    # RSSA_df.to_csv('C:/Users/dego/Downloads/test.csv')

    area_df_wide = RSSA_df.pivot(index='Q_decile', columns='order', values='area_km2')
    area_df_wide.columns = [str(o) for o in np.uint8(np.arange(1, 15, 1))]
    area_df_wide = area_df_wide.reset_index()

    color14 = ['#4379AB',
               '#96CCEB',
               '#FF8900',
               '#FFBC71',
               '#3DA443',
               '#76D472',
               '#BA9900',
               '#F7CD4B',
               '#249A95',
               '#77BEB6',
               '#F14A54',
               '#FF9797',
               '#7B706E',
               '#BCB0AB',]

    color13 = ['#4379AB',
               '#96CCEB',
               '#FF8900',
               '#FFBC71',
               '#3DA443',
               '#76D472',
               '#BA9900',
               '#F7CD4B',
               '#249A95',
               '#77BEB6',
               '#F14A54',
               '#FF9797',
               '#7B706E',]
    color14.reverse()
    color13.reverse()


    fig, axs = plt.subplots(figsize=(10.5, 5), constrained_layout=True, ncols=3, sharey=True)
    ax = axs[2]
    labels = [fr'{order}' for order in np.arange(14, 0, -1)]
    area_df_wide[['Q_decile', '14', '13', '12', '11', '10', '9', '8', '7', '6', '5', '4', '3', '2', '1']].plot(x='Q_decile', kind='bar', stacked=True, ax=ax, color=color14, legend=False)
    ax.set_xticklabels([f'{o * 10} - {(o + 1) * 10}' for o in range(10)])
    ax.legend(labels, title='Stream Order', ncols=2, loc='upper left')
    # # ax.set_ylim(0, 29000)
    ax.set_xlabel('Flow percentile')
    # ax.set_ylabel(r'RSSA $(km^{2})$')
    ax.set_title('Longitudinal and lateral expansion')

    ax.axhline(y=17828, c='black')

    # # import os
    # # if os.path.isfile('C:/Users/dego/Desktop/fa.png'):
    # #     os.remove('C:/Users/dego/Desktop/fa.png')
    # # plt.savefig('C:/Users/dego/Desktop/fa.png')


    # just width
    RSSA_df = RSSA_df_wrapper(dparm_stats, length_df, fixed_length=True)
    # RSSA_df.to_csv('C:/Users/dego/Downloads/test.csv')

    area_df_wide = RSSA_df.pivot(index='Q_decile', columns='order', values='area_km2')
    area_df_wide.columns = [str(o) for o in np.uint8(np.arange(1, 14, 1))]
    area_df_wide = area_df_wide.reset_index()

    # color20 = ['#4379AB', '#96CCEB', '#FF8900', '#FFBC71', '#3DA443', '#76D472', '#BA9900', '#F7CD4B', '#249A95', '#FF9797', '#7B706E', '#BCB0AB', '#E16A96',]
            # '#FFBCD3',
            # '#B976A3',
            # '#DCA3CA',
            # '#A3745C',
            # '#DDB3A4']
    # color20.reverse()

    # fig, ax = plt.subplots(figsize=(4, 4), constrained_layout=True)
    ax = axs[0]
    labels = [fr'{order}' for order in np.arange(13, 0, -1)]
    area_df_wide[['Q_decile', '13', '12', '11', '10', '9', '8', '7', '6', '5', '4', '3', '2', '1']].plot(x='Q_decile', kind='bar', stacked=True, ax=ax, color=color13, legend=False)
    ax.set_xticklabels([f'{o * 10} - {(o + 1) * 10}' for o in range(10)])
    # ax.legend(labels, title='Stream Order', ncols=2, loc='upper left')
    # # ax.set_ylim(0, 29000)
    ax.set_xlabel('Flow percentile')
    ax.set_ylabel(r'RSSA $(km^{2})$')
    ax.set_title('Lateral expansion only')

    ax.axhline(y=17828, c='black')
    # just length
    RSSA_df = RSSA_df_wrapper(dparm_stats, length_df, fixed_width=True)
    # RSSA_df.to_csv('C:/Users/dego/Downloads/test.csv')

    area_df_wide = RSSA_df.pivot(index='Q_decile', columns='order', values='area_km2')
    area_df_wide.columns = [str(o) for o in np.uint8(np.arange(1, 15, 1))]
    area_df_wide = area_df_wide.reset_index()

    # color20 = ['#4379AB', '#96CCEB', '#FF8900', '#FFBC71', '#3DA443', '#76D472', '#BA9900', '#F7CD4B', '#249A95', '#77BEB6', '#F14A54']
    # , '#FF9797', '#7B706E', '#BCB0AB', '#E16A96',]
            # '#FFBCD3',
            # '#B976A3',
            # '#DCA3CA',
            # '#A3745C',
            # '#DDB3A4']
    # color20.reverse()

    # fig, ax = plt.subplots(figsize=(4, 4), constrained_layout=True)
    ax = axs[1]
    labels = [fr'{order}' for order in np.arange(11, 0, -1)]
    area_df_wide[['Q_decile', '14', '13', '12', '11', '10', '9', '8', '7', '6', '5', '4', '3', '2', '1']].plot(x='Q_decile', kind='bar', stacked=True, ax=ax, color=color14, legend=False)
    ax.set_xticklabels([f'{o * 10} - {(o + 1) * 10}' for o in range(10)])
    # ax.legend(labels, title='Stream Order', ncols=2, loc='upper left')
    # # ax.set_ylim(0, 29000)
    ax.set_xlabel('Flow percentile')
    # ax.set_ylabel(r'RSSA $(km^{2})$')
    ax.set_title('Longitudinal expansion only')

    ax.axhline(y=17828, c='black')

    plt.show()


    RSSA_df_list = [-1] * n_iterations
    area_factor_list_2var = [-1] * n_iterations
    area_factor_list_wvar = [-1] * n_iterations
    area_factor_list_lvar = [-1] * n_iterations
    meanareas = [-1] * n_iterations

    # how much does network expand with variable length and width?
    for i in tqdm(range(n_iterations)):
        RSSA_df_list[i] = RSSA_df_wrapper(dparm_stats, length_df)
    
    for i in (range(n_iterations)):
        RSSA_df = RSSA_df_list[i]
        low_area = np.sum(RSSA_df.loc[RSSA_df['Q_decile'] == 0, 'area_km2'])
        high_area = np.sum(RSSA_df.loc[RSSA_df['Q_decile'] == 9, 'area_km2'])

        area_factor_list_2var[i] = high_area / low_area

    area_factor_list_2var = np.array(area_factor_list_2var)
    big_RSSA_df = pd.concat(RSSA_df_list)
    # big_RSSA_df.to_csv('C:/Users/dego/Desktop/RSSA_lw_20260603.csv')
    
    # how much does network expand just variable length?
    for i in tqdm(range(n_iterations)):
        RSSA_df_list[i] = RSSA_df_wrapper(dparm_stats, length_df, fixed_width=True)
    # big_RSSA_df = pd.concat(RSSA_df_list)
    # big_RSSA_df.to_csv('C:/Users/dego/Desktop/RSSA_l_20260512.csv')
    
    for i in (range(n_iterations)):
        RSSA_df = RSSA_df_list[i]
        low_area = np.sum(RSSA_df.loc[RSSA_df['Q_decile'] == 0, 'area_km2'])
        high_area = np.sum(RSSA_df.loc[RSSA_df['Q_decile'] == 9, 'area_km2'])

        area_factor_list_lvar[i] = high_area / low_area

    area_factor_list_lvar = np.array(area_factor_list_lvar)


    # how much does network expand just variable width?
    for i in tqdm(range(n_iterations)):
        RSSA_df_list[i] = RSSA_df_wrapper(dparm_stats, length_df, fixed_length=True)
    # big_RSSA_df = pd.concat(RSSA_df_list)
    # big_RSSA_df.to_csv('C:/Users/dego/Desktop/RSSA_w_20260512.csv')
    
    for i in (range(n_iterations)):
        RSSA_df = RSSA_df_list[i]
        low_area = np.sum(RSSA_df.loc[RSSA_df['Q_decile'] == 0, 'area_km2'])
        high_area = np.sum(RSSA_df.loc[RSSA_df['Q_decile'] == 9, 'area_km2'])

        area_factor_list_wvar[i] = high_area / low_area

    area_factor_list_wvar = np.array(area_factor_list_wvar)



    print('Longitudinal and lateral expansion')
    print(f'RSSA 10th decile / RSSA 1st decile = {np.mean(area_factor_list_2var)} +- 1 std {np.std(area_factor_list_2var)}')
    print(' ')
    print('Longitudinal expansion only')
    print(f'RSSA 10th decile / RSSA 1st decile = {np.mean(area_factor_list_lvar)} +- 1 std {np.std(area_factor_list_lvar)}')
    print(' ')
    print('Lateral expansion only')
    print(f'RSSA 10th decile / RSSA 1st decile = {np.mean(area_factor_list_wvar)} +- 1 std {np.std(area_factor_list_wvar)}')
    

    fig, (ax2, axl, axw) = plt.subplots(ncols=3, figsize=(9, 3))
    ax2.hist(area_factor_list_2var)
    axl.hist(area_factor_list_lvar)
    axw.hist(area_factor_list_wvar)
    plt.show()
