# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import quandl
quandl.ApiConfig.api_key = "AkvsAjUV5KnZahNpFyMw"
import numpy as np
import pandas as pd
import seaborn as sns

prices = quandl.get_table('SHARADAR/SEP', ticker=['AAPL', 'KO', 'NKE', 'TSLA', 'XOM'], date={'gte':'2017-01-01', 'lte':'2018-12-31'}, paginate=True)
prices

pivoted = prices.pivot(index = 'date', columns = 'ticker', values = 'close')
pivoted

pivoted.pct_change()

aapl = pivoted['AAPL'].pct_change().apply(lambda x: (np.log(1+x)))
aapl.head()

# ### Variance
# $$s^2 = \frac{\sum_{i=1}^N (x_i - \bar{x})^2}{N-1}$$
# Standard Deviation (Volatility)
# $$s = \sqrt{\frac{\sum_{i=1}^N (x_i - \bar{x})^2}{N-1}}$$

mean_aapl = aapl.sum()/aapl.count()
mean_aapl

var = aapl.apply(lambda x: (x- mean_aapl)**2)
var

var_aapl = var.sum()/(aapl.count() - 1)
var_aapl

aapl.var()

# ### std 

st_aapl_d = np.sqrt(var_aapl)
st_aapl_d

annual = aapl.std() * np.sqrt(250)
annual

pivoted.pct_change().apply(lambda x: np.log(1+x)).std().apply(lambda x: x*np.sqrt(250)).plot(kind = 'bar')

# ### Covariance
# $$\sigma_{ij} = \frac{\sum_{i=1j=1}^N (x_i - \bar{x_i})(x_j - \bar{x_j})}{N-1}$$

df = pivoted.pct_change()
df.head()

mean_AAPL = df['AAPL'].mean()
mean_TSLA = df['TSLA'].mean()
mean_AAPL

df['dif_AAPL'] = df['AAPL'] - mean_AAPL
df['dif_TSLA'] = df['TSLA'] - mean_TSLA
df

df['COV_TSLA_AAPL'] = (df['dif_AAPL'] * df['dif_TSLA'])/(df['AAPL'].count() - 1)
df

df['COV_TSLA_AAPL'].sum()

df['AAPL'].cov(df['TSLA'])

# ### Correlation
# $$p(R_i,R_j) = \frac{COV(R_i, R_j)}{\sigma_i \sigma_j}$$

cov_xom_tsla = df['TSLA'].cov(df['XOM'])
cov_xom_tsla

df['TSLA'].corr(df['XOM'])

# ### corr matrix

prices = quandl.get_table('SHARADAR/SEP', date={'gte':'2017-01-01', 'lte':'2018-12-31'}, paginate=True)

df = prices.pivot(index='date', columns='ticker', values='close').pct_change().apply(lambda x: np.log(1+x))

df.cov().head()

df.corr().head()

df.corr()['AAPL'].plot(kind='bar')

sns.heatmap(df.corr())

# ### portfolio variance

# ### $$\sigma^2(R_p) = w_1w_1\sigma^2(R_1) + w_1w_2COV(R_1,R_2) + w_2w_2\sigma^2(R_2) + w_2w_1COV(R_1,R_2)$$
# $$\sigma^2(R_p) = \sum_{i=1}^n\sum_{j=1}^nw_iw_jCOV(R_i, R_j)$$

prices = quandl.get_table('SHARADAR/SEP', ticker=['AAPL', 'NKE'], date={'gte':'2000-01-01', 'lte':'2018-12-31'}, paginate=True)

pivoted = prices.pivot_table(index = 'date', columns = 'ticker', values = 'close')
pivoted

cov_matrix = pivoted.apply(lambda x: np.log(1+x)).cov()
cov_matrix

w = {'AAPL' : 0.75, 'NKE' : 0.25}


cov_matrix.mul(w, axis = 0)

p_var = cov_matrix.mul(w, axis = 0).mul(w, axis = 1).sum().sum()
p_var

p_sd = np.sqrt(p_var)
p_sd

annual_p_sd = p_sd * np.sqrt(250)
annual_p_sd

# #### portfolio std with corr() function

# ###$$p(R_i,R_j) = \frac{COV(R_i, R_j)}{\sigma_i \sigma_j}$$
# $$COV(R_i,R_j) = p(R_i,R_j)\sigma_i\sigma_j$$
# Portfolio Variance
# $$\sigma^2(R_p) = w_1^2\sigma^2(R_1) + w_2^2\sigma^2(R_2) + 2w_1w_2p(R_i,R_j)\sigma_i \sigma_j$$

stds = df.std()
stds.head()

corr = df.corr()
corr.head()

aapl_nke_corr = df['AAPL'].corr(df['NKE'])
aapl_nke_corr

p_var = w['AAPL']**2*stds['AAPL']**2 + w['NKE']**2*stds['NKE']**2 + 2*w['AAPL']*w['NKE']*aapl_nke_corr*stds['AAPL']*stds['NKE']
p_var

p_sd = np.sqrt(p_var)
p_sd

ann_p_sd = p_sd * np.sqrt(250)
ann_p_sd

# ### expected return

pivoted

exp_r = pivoted.resample('M').last().pct_change().mean()
exp_r

exp_r_p = exp_r['AAPL'] * w['AAPL'] + exp_r['NKE'] * w['NKE']
exp_r_p

# ### Efficient frontier

cov_matrix = pivoted.pct_change().apply(lambda x: np.log(1+x)).cov()
cov_matrix

sd = pivoted.pct_change().apply(lambda x: np.log(1+x)).std()
sd

assets = pd.concat([exp_r,sd],axis = 1)
assets.columns = ['Returns', 'Volatility']
assets

p_ret = []
p_vol = []
p_weights = []
num_portfolios = 1

for i in range(num_portfolios):
    weights = [0.75, 0.25]
    p_weights.append(weights)
    returns = np.dot(weights, exp_r)
    p_ret.append(returns)
    var = cov_matrix.mul(weights, axis = 0).mul(weights, axis = 1).sum().sum()
    sd = np.sqrt(var)
    ann_sd = sd*np.sqrt(30)
    p_vol.append(ann_sd)

data = {'Returns' : p_ret, 'Volatility': p_vol}

portfolios = pd.DataFrame(data)
portfolios.index = ['portfolio 1']
portfolios

op_space = pd.concat([assets,portfolios],axis = 0)
op_space

op_space.plot.scatter(x= 'Volatility', y = 'Returns',grid = True)

# #### efficient frontier for 3 assets and and 1000 op_space

prices = quandl.get_table('SHARADAR/SEP', ticker=['AAPL', 'KO', 'NKE'], date={'gte':'2000-01-01', 'lte':'2018-12-31'}, paginate=True)

pivoted = prices.pivot(index='date', columns='ticker', values='close')
pivoted.head()

cov_matrix = pivoted.pct_change().apply(lambda x: np.log(1+x)).cov()
cov_matrix

e_r = pivoted.pct_change().mean()
e_r

sd = pivoted.pct_change().apply(lambda x: np.log(1+x)).std()
sd

assets = pd.concat([e_r, sd], axis=1)
assets.columns = ['Returns', 'Volatility']
assets

# +
p_ret = []
p_vol = []
p_weights = []

num_assets = len(pivoted.columns)
num_portfolios = 1000
# -

for portfolio in range(num_portfolios):
    #weights = [.25, .75]
    weights = np.random.random(num_assets)
    weights = weights/np.sum(weights)
    p_weights.append(weights)
    returns = np.dot(weights, e_r)
    p_ret.append(returns)
    var = cov_matrix.mul(weights, axis=0).mul(weights, axis=1).sum().sum()
    sd = np.sqrt(var)
    ann_sd = sd
    p_vol.append(ann_sd)


data = {'Returns':p_ret, 'Volatility':p_vol}

for counter, symbol in enumerate(pivoted.columns.tolist()):
    #print(counter, symbol)
    data[symbol+' weight'] = [w[counter] for w in p_weights]

portfolios  = pd.DataFrame(data)
portfolios.head()

portfolios.plot.scatter(x='Volatility', y='Returns', grid=True)


