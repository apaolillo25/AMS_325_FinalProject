#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 03:22:27 2021

@author: apaolillo25
"""

#Final Project
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline 
#%conda install statsmodels
import statsmodels.api as sm 
import statsmodels.formula.api as smf
import statsmodels.graphics.api as smg


#Loading Data
df = pd.read_csv('Downloads/cbb.csv')
#Info
df.info
df.dtypes
#Change dtypes
#df['SEED'] = df['SEED'].astype(int)
df['YEAR'] = df['YEAR'].astype(int)
df.dtypes
df['POSTSEASON'].unique()
#Sorting and resave df (inplace = T) #df['YEAR'].sort_values()   

#2019 Data
df_last = df.loc[df.YEAR == 2019]
valid = df.notnull()
df_last = df_last.loc[valid.POSTSEASON]
df_last.sort_values(['WAB'], ascending=False)
#Now df_last has 2019 tournament teams
#correlation between seed and other variables for 2019
df_last.corr()['SEED'].sort_values(ascending=False)
#How we can get correlation of variables; This will be useful with our Postseason rank

#Top Seeds Data
df_one = df.loc[df.SEED == 1]
df_one.SEED.astype(int)
print(df_one.corr()['W'].sort_values(ascending=False))


#Non Game Stats
df[['POSTSEASON','W','BARTHAG','WAB','ADJOE','ADJDE']].groupby('POSTSEASON').mean().sort_values('W', ascending=False)
#Game Stats
df[['POSTSEASON','2P_O','2P_D','3P_O','3P_D','EFG_O',"EFG_D",'TORD','TOR','ADJ_T','FTR','FTRD','ORB','DRB']].groupby('POSTSEASON').mean().sort_values('EFG_O', ascending=False)

#Adding interger for postseason finish
for i, row in df.iterrows():
    if df.at[i, 'POSTSEASON'] == 'R68' or df.at[i, 'POSTSEASON'] == 'R64':
        df.at[i, 'PSZNN'] = 1
    elif df.at[i, 'POSTSEASON'] == 'R32':
        df.at[i, 'PSZNN'] = 2
    elif df.at[i, 'POSTSEASON'] == 'S16':
        df.at[i, 'PSZNN'] = 3
    elif df.at[i, 'POSTSEASON'] == 'E8':
        df.at[i, 'PSZNN'] = 4
    elif df.at[i, 'POSTSEASON'] == 'F4':
        df.at[i, 'PSZNN'] = 5
    elif df.at[i, 'POSTSEASON'] == '2ND':
        df.at[i, 'PSZNN'] = 6
    elif df.at[i, 'POSTSEASON'] == 'Champions':
        df.at[i, 'PSZNN'] = 7
    else:
        df.at[i,'PSZNN'] = 0

df.SEED = df.SEED.fillna(0)

#Exploratory Data Analysis
#For all teams
df.describe()
df['POSTSEASON'].unique()
valid = df.notnull()

#Teams that made the tournament
df_t = df.loc[valid.POSTSEASON]
df_t.describe()

#Pre-Tournament Seeds
df.drop('YEAR',axis=1).groupby('SEED').mean()
#Get a good idea which variables go with what

#Trying to find correlations; Plot some of most important
df.drop(['WAB','G','PSZNN', 'BARTHAG'], axis=1).corr()['W'].sort_values(ascending=False)
#Correlation with Pre-Tournament Seed
df.drop(['PSZNN','G'],axis=1).corr()['SEED'].sort_values(ascending=False)

#First Regression: Wins and game stats
#Use df.drop([v1,v1], axis =1) to include all
#model.fit(df.drop('y', axis=1), df['y'])
#This part is useful for our analysis on what stats correlate with wins

#Stats that directly impact the game
import patsy
#-1 if we want no intercept
model = smf.ols('W~ADJOE+ADJDE+EFG_O+EFG_D+TOR+TORD+ORB+DRB+FTR+FTRD+Q("2P_O")+Q("2P_D")+Q("3P_O")+Q("3P_D")+ADJ_T', df)
result = model.fit()
print(result.summary())
#This is from whole dataset; can tell the story on what stats correlate with more wins
#Look at the r^2, and what values are significant on wins
fig, ax = plt.subplots(figsize=(8, 4))
smg.qqplot(result.resid, ax=ax)

fig.tight_layout()

#IVs that are significant: ADJOe,ADJDE, EFG_O, EDG_D, TOR, TORD, ORB, DRB, FTR, FTRD, ADJ_T

#Teams that appeared in tourney = 1, not in = 0
#Use for logistic regression


df_subset = df.copy()
df_subset.POSTSEASON = df_subset.POSTSEASON.fillna('None')
df_subset['APPEARANCE'] = df_subset.POSTSEASON.map({'2ND':1,'Champions':1,'E8':1,'R32':1,'R64':1,'R68':1,'S16':1, 'None':0})
df_subset.APPEARANCE.sort_values()
#Logistic Regression--> Maybe include wins now?

model = smf.logit('APPEARANCE~W+ADJOE+ADJDE+EFG_O+EFG_D+TOR+TORD+ORB+DRB+FTR+FTRD+Q("2P_O")+Q("2P_D")+Q("3P_O")+Q("3P_D")+ADJ_T', df_subset)
result = model.fit()
print(result.summary())
#IVs that are significant: Wins, ADJOE, ADJDE, DRB,FTR

#Making it past first weekend (S16) vs not- Once you get into tournament

df_first = df_t.copy()
df_first.POSTSEASON = df_first.POSTSEASON.fillna('None')
df_first['APPEARANCE'] = df_first.POSTSEASON.map({'2ND':1,'Champions':1,'E8':1,'R32':0,'R64':0,'R68':0,'S16':1})
#Logistic Regression--> Maybe include wins now?
model = smf.logit('APPEARANCE~W+ADJOE+ADJDE+EFG_O+EFG_D+TOR+TORD+ORB+DRB+FTR+FTRD+Q("2P_O")+Q("2P_D")+Q("3P_O")+Q("3P_D")+ADJ_T', df_first)
result = model.fit()
print(result.summary())

#Sig. Variables Making it past first round
#W,ADJOE,ADJDE,TORD,DRB,(3P_D), (2P_D)
#Making Tournament vs Making it out of the Region (reaching Final 4)

df_ff = df_t.copy()
df_ff.POSTSEASON = df_ff.POSTSEASON.fillna('None')
df_ff['APPEARANCE'] = df_first.POSTSEASON.map({'2ND':1,'Champions':1,'F4':1,'E8':0,'R32':0,'R64':0,'R68':0,'S16':0})

#Logistic Regression--> Maybe include wins now?
model = smf.logit('APPEARANCE~W+ADJOE+ADJDE+EFG_O+EFG_D+TOR+TORD+ORB+DRB++FTR+FTRD+Q("2P_O")+Q("2P_D")+Q("3P_O")+Q("3P_D")+ADJ_T', df_ff)
result = model.fit()
print(result.summary())
#W, ADJDE,EFG_O,TORD,2P_O,3P_O, FTR


#Subset and Plotting
from statsmodels.miscmodels.ordinal_model import OrderedModel
import itertools
from tqdm import tnrange, tqdm_notebook
def fit_logistic_reg(X,Y):
    #Fit linear regression model and return RSS and R squared values
    model_k = OrderedModel(Y, X, distr='logit')

    res_k = model_k.fit(method='bfgs', disp=False)
    R_squared = res_k.prsquared
    AIC = res_k.aic
   
    #RSS = mean_squared_error(Y,model_k.predict(X)) * len(Y)
    #R_squared = model_k.score(X,Y)
    #return RSS, R_squared
    return AIC, R_squared
#Initialization variables
Y = df.PSZNN
X = df.drop(columns = ['PSZNN', 'TEAM', 'CONF', 'POSTSEASON'], axis = 1)
k = 19
remaining_features = list(X.columns.values)
features = []
RSS_list, R_squared_list = [np.inf], [np.inf] #Due to 1 indexing of the loop...
features_list = dict()

for i in tnrange(1,k+1):
    best_RSS = np.inf
    
    for combo in itertools.combinations(remaining_features,1):

            RSS = fit_logistic_reg(X[list(combo) + features],Y)   #Store temp result 

            if RSS[0] < best_RSS:
                best_RSS = RSS[0]
                best_R_squared = RSS[1] 
                best_feature = combo[0]

    #Updating variables for next loop
    features.append(best_feature)
    remaining_features.remove(best_feature)
    
    #Saving values for plotting
    RSS_list.append(best_RSS)
    R_squared_list.append(best_R_squared)
    features_list[i] = features.copy()

print('Forward stepwise subset selection')
print('Number of features |', 'Features |', 'AIC |', 'R2 |', 'BIC')
display([(i,features_list[i], AIC_list[i], round(R_squared_list[i]), BIC_list[i]) for i in range(1,18)])     
df1 = pd.concat([pd.DataFrame({'features':features_list}),pd.DataFrame({'AIC':AIC_list, 'BIC': BIC_list})], axis=1, join='inner')
df1['numb_features'] = df1.index
variables = ['AIC','BIC']
fig = plt.figure(figsize = (18,6))


for i,v in enumerate(variables):
    ax = fig.add_subplot(1, 4, i+1)
    ax.plot(df1['numb_features'],df1[v], color = 'lightblue')
    ax.scatter(df1['numb_features'],df1[v], color = 'darkblue')
    if v == 'R_squared_adj':
        ax.plot(df1[v].idxmax(),df1[v].max(), marker = 'x', markersize = 20)
    else:
        ax.plot(df1[v].idxmin(),df1[v].min(), marker = 'x', markersize = 20)
    ax.set_xlabel('Number of predictors')
    ax.set_ylabel(v)

fig.suptitle('Subset selection AIC, BIC', fontsize = 16)
plt.show()





#Results of the tournament
df.drop('YEAR',axis=1).groupby('PSZNN').mean().sort_values('PSZNN', ascending=False)
df.drop(['WAB','G','PSZNN', 'BARTHAG'], axis=1).corr()['W'].sort_values(ascending=False)
df.drop(['PSZNN','G'],axis=1).corr()['SEED'].sort_values(ascending=False)
   
#Need a Machine Learning
#Use Logistic because dependent is categorical
#Did not work well with all tournament teams; may do a different one with only two categories
#Lets just use all of the teams that made the tournament

#X and Y     
#Remove NA values
#Since a team that doesnt make the tournament doesnt have a seed, replace Nan with 0
#Using Subset: We now have a few new variables we will use

#'DRB','EFG_D',
y = df.PSZNN.astype(int)
y = df.PSZNN[df.SEED > 0]
X = df.drop(['TORD','3P_O','FTR','TORD','G','PSZNN','TEAM','YEAR','CONF','POSTSEASON','W','EFG_O','TOR','ORB','FTRD','ADJ_T','3P_D','2P_O','2P_D'], axis = 1)
X.dropna(inplace=True)
X = X.loc[df.SEED > 0 ]
y.shape, X.shape


#PSZNN variable is our "level variable"
#!pip install git+https://github.com/statsmodels/statsmodels

from sklearn import datasets
from sklearn import model_selection   
from sklearn import linear_model
from sklearn import metrics
from sklearn import tree
from sklearn import neighbors
from sklearn import svm
from sklearn import ensemble
from sklearn import cluster
import seaborn as sns
import numpy as np
import scipy.stats as stats

from statsmodels.miscmodels.ordinal_model import OrderedModel


### I removed Barthag from the model since they are heavily correlated 
### and the coefficient of it did not make any sense in the summary
X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y,train_size=0.70)

#mod_log = OrderedModel(y_train, X_train , distr='logit')
mod_log = OrderedModel(df['PSZNN'],  df[['WAB', 'SEED', 'ADJOE', 'ADJDE', 'DRB', 'EFG_D']], distr='logit') #'BARTHAG'
# -EFG_D, -TOR, TORD
res_log = mod_log.fit(method='bfgs', disp=False)
res_log.summary()


#Random Forest
np.random.seed(555)  
X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y,train_size=0.70)
classifier = ensemble.RandomForestClassifier()
classifier.fit(X_train, y_train)
y_test_pred = classifier.predict(X_test)
#print(metrics.classification_report(y_test, y_test_pred))
metrics.accuracy_score(y_test, y_test_pred)
metrics.plot_confusion_matrix(classifier, X_test, y_test)

#Testing our Model with 2021 Data
df_new = pd.read_csv('cbb21.csv')
#Adding interger for postseason finish
df_new.head()
df_new.SEED = df_new.SEED.fillna(0)
df_new['YEAR'] = 2021
X_new = df_new.drop(['TORD','3P_O','FTR','TORD','G','TEAM','CONF','W','EFG_O','TOR','ORB','FTRD','ADJ_T','3P_D','2P_O','2P_D'], axis = 1)
X_new = X_new[X_new.SEED > 0]
y_test_new = classifier.predict(X_new)


X_new['TEAM'] = df_new['TEAM']
X_new['PRED'] = y_test_new
X_new[X_new.PRED == 1]
result_n = np.array([4,7,2,6,1,1,4,3,2,1,2,4,3,1,1,2,3,1,3,2,
                     2,1,4,1,2,1,1,3,2,1,2,3,1,1,2,1,2,1,2,1,
                     1,1,3,5,1,1,1,4,1,1,2,2,1,1,1,1,2,1,1,3,
                     1,1,1,1,1,1,1,1])
X_new['POST'] = result_n


#Start of True Seed
X_new['RANK'] = X_new.PRED.ran
#WAB was most sig. variable, so thats why we sorted by it
X_new = X_new.sort_values('WAB', ascending=False)
X_new['TSEED'] = ''
count_row = X_new.shape[0]
true_seed = []
seed = 1
print(count_row)
true_seed.append(1)
for i in range(1, count_row+1):
  if i%4 == 0:
    seed = seed + 1

  if seed == 11 or seed == 16:
    for i in range(1,7):
      true_seed.append(seed)
    seed = seed + 1
  
  if seed == 17:
    break
    
  true_seed.append(seed)


print(true_seed)

X_new.sort_values('RANK',ascending=False, inplace= True)
X_new['TSEED'] = true_seed
X_new[['TEAM','TSEED']].head(8)
X_new

#Is our model over or under predicting
print(len(X_new[X_new.PRED < X_new.POST]))
print(len(X_new[X_new.PRED > X_new.POST])) 
print(len(X_new[X_new.PRED == X_new.POST]))

#Corr. Matrix for Slides
matrix = df.drop(columns = ['PSZNN', 'TEAM', 'CONF', 'POSTSEASON','G', 'YEAR','WAB','SEED'], axis = 1)
matrix.head()
matrix.corr()
matrix.corr().style.background_gradient(cmap='coolwarm')


