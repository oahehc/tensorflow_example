'''
Apply Kaggle competition data as example
https://www.kaggle.com/c/house-prices-advanced-regression-techniques
'''
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches

# load data
dataFrame = pd.read_csv('HousePrices_train.csv')
dataFrame.head()


# bar plot
counts = dataFrame['Neighborhood'].value_counts()
fig = plt.figure(figsize=(8, 6))
ax = fig.gca()
counts.plot.bar(ax = ax) # use plot.bar method on the counts data frame
ax.set_xlabel('Physical locations within Ames city limits')
ax.set_ylabel('Number')


# histogram
fig = plt.figure(figsize=(8, 6))
ax = fig.gca()
dataFrame['LotArea'].plot.hist(ax = ax)
ax.set_xlabel('Lot size in square feet')
ax.set_ylabel('Numbers')


# box plot
# OverallQual: Rates the overall material and finish of the house
# OverallCond: Rates the overall condition of the house
fig = plt.figure(figsize=(8, 6))
ax = fig.gca()
dataFrame[['OverallQual', 'OverallCond']].boxplot(by = 'OverallCond', ax = ax)


# kernel density plot
fig = plt.figure(figsize=(8, 6))
ax = fig.gca()
sns.set_style('whitegrid')
sns.kdeplot(dataFrame['YearBuilt'])
ax.set_xlabel('Year')
ax.set_ylabel('Density')


# Violin plot
fig = plt.figure(figsize=(8, 6))
ax = fig.gca()
sns.set_style('whitegrid')
sns.violinplot(x = 'OverallQual', y = 'LotArea', data = dataFrame, ax = ax)
ax.set_xlabel('OverallQual')
ax.set_ylabel('LotArea')


# scatter plot
fig = plt.figure(figsize=(8, 6))
ax = fig.gca()
dataFrame.plot(kind = 'scatter', x = 'YearBuilt', y = 'SalePrice', ax = ax)
ax.set_xlabel('Year Built')
ax.set_ylabel('Sale Price')


# 2-D kernel density plot
fig = plt.figure(figsize=(8, 6))
ax = fig.gca()
sns.set_style('whitegrid')
sns.kdeplot(dataFrame[['YearRemodAdd','SalePrice']], ax = ax, cmap = 'Blues_d')
dataFrame.plot(kind = 'scatter', x = 'YearRemodAdd', y = 'SalePrice', ax = ax)
ax.set_xlabel('Year RemodAdd')
# dataFrame.plot(kind = 'scatter', x = 'YearBuilt', y = 'SalePrice', ax = ax)
# ax.set_xlabel('Year Built')
ax.set_ylabel('Sale Price')


# line plot
df = dataFrame.groupby(['YearBuilt'], as_index = False).mean().sort_values('YearBuilt')
fig = plt.figure(figsize=(8, 6))
ax = fig.gca()
df.plot(x = 'YearBuilt', y = 'SalePrice', ax = ax)
ax.set_xlabel('YearBuilt')
ax.set_ylabel('SalePrice')


# mark different color
sns.lmplot(x = 'YearBuilt', y = 'SalePrice', data = dataFrame, hue = 'YrSold', palette = 'Set2', fit_reg = False)

# mark different color with manaull select color and create legend
def auto_color(df, col_x, col_y):
    fig = plt.figure(figsize=(6, 6))
    ax = fig.gca()
    temp1 = df.ix[df['YrSold'] == 2007]
    temp2 = df.ix[df['YrSold'] == 2008]
    if temp1.shape[0] > 0 :
        temp1.plot(kind = 'scatter', x = col_x, y = col_y, ax = ax, color = 'DarkBlue', alpha = 0.3)
    if temp2.shape[0] > 0 :
        temp2.plot(kind = 'scatter', x = col_x, y = col_y, ax = ax, color = 'Red', alpha = 0.3)
    ax.set_title(col_x + ' vs. ' + col_y)
    red_patch = mpatches.Patch(color = 'Red', label = '2007')
    blue_patch = mpatches.Patch(color = 'DarkBlue', label = '2008')
    plt.legend(handles = [red_patch, blue_patch])
    return 'Done'
auto_color(dataFrame, 'YearBuilt', 'SalePrice')


# marker size
def auto_size(df, col_x, col_y, col_size):
    fig = plt.figure(figsize=(6, 6))
    ax = fig.gca()
    temp1 = df.ix[df['YrSold'] == 2007]
    temp2 = df.ix[df['YrSold'] == 2008]
    if temp1.shape[0] > 0 :
        temp1.plot(kind = 'scatter', x = col_x, y = col_y, ax = ax, color = 'DarkBlue', s = .01*df[col_size], alpha = 0.3)
    if temp2.shape[0] > 0 :
        temp2.plot(kind = 'scatter', x = col_x, y = col_y, ax = ax, color = 'Red', s = .01*df[col_size], alpha = 0.3)
    red_patch = mpatches.Patch(color = 'Red', label = '2007')
    blue_patch = mpatches.Patch(color = 'DarkBlue', label = '2008')
    plt.legend(handles = [red_patch, blue_patch])
    return 'Done'
auto_size(dataFrame, 'YearBuilt', 'SalePrice', 'LotArea')


# marker shape
def auto_shape(df):
    import matplotlib.pyplot as plt
    import matplotlib.lines as mlines
    LotConfig = ['Corner', 'Corner', 'Inside', 'Inside']
    LotShape = ['IR1', 'Reg', 'IR1', 'Reg']
    mrks = ['o', 'o', '+', '+']
    cols = ['Red', 'Blue', 'DarkRed', 'DarkBlue']
    ldg = []
    fig = plt.figure(figsize=(12, 6))
    ax = fig.gca()
    for asp, fl, mk ,cl in zip(LotConfig, LotShape, mrks, cols):
        temp = df.ix[(df['LotConfig'] == asp) & (df['LotShape'] == fl)]
        if temp.shape[0] > 0:
            temp.plot(kind = 'scatter', x = 'YearBuilt', y = 'SalePrice', ax = ax, color = cl, marker = mk, alpha = 0.5)
        ldg.append(mlines.Line2D([], [], color = cl, marker = mk, markersize = 10, label=(asp + ' and ' + fl)))
    plt.legend(handles = ldg)
    return 'Done'
auto_shape(dataFrame)


# split violin plot
sns.set_style('whitegrid')
sns.violinplot(data = dataFrame, x = 'YrSold', y = 'SalePrice', hue = 'Street' , split = True)
# box plots
sns.set_style('whitegrid')
sns.boxplot(data = dataFrame, x = 'YrSold', y = 'SalePrice', hue = 'Street')


# pair wise plot
num_cols = ['YrSold', 'OverallQual', 'OverallCond']
sns.pairplot(dataFrame[num_cols], hue = 'YrSold', palette = 'Set2', diag_kind = 'kde', size = 2)


# facet plot
# g = sns.FacetGrid(dataFrame, col = 'LotShape')
# g.map(sns.distplot, 'SalePrice')
# g = sns.FacetGrid(dataFrame, col = 'LotShape', row = 'Fence')
# g.map(sns.distplot, 'SalePrice')
g = sns.FacetGrid(dataFrame, col = 'LotShape', row = 'Fence', hue = 'Street', palette = 'Set2')
g.map(sns.regplot, 'LotArea', 'SalePrice', fit_reg = False)


