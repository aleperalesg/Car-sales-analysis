import numpy as np
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


## get car data csv 
raw_data = pd.read_csv("1.04.+Real-life+example.csv")
print("\n\n\n#############################  raw data #############################")
print(raw_data.describe(include = 'all'))
data = raw_data.drop(['Model'],axis = 1)


## check and remove missing values
print("\n\n\n#############################  missing values #############################")
print(data.isnull().sum())
data_no_mv = data.dropna(axis = 0)
print("\n\n\n#############################  data no missing values #############################")
print(data_no_mv.describe(include = 'all'))


## remove outliers

## price
q = data_no_mv['Price'].quantile(0.99) ## get 0.99 quantile
data_1 = data_no_mv[data_no_mv['Price']<q]
#sns.displot(data_1['Price'])

## mileage
q = data_1['Mileage'].quantile(0.99) ## get 0.99 quantile
data_2 = data_1[data_no_mv['Mileage']<q]
#sns.displot(data_2['Mileage'])

## EngineV
data_3 = data_2[data_2['EngineV']<6.5] 
#sns.displot(data_3['EngineV'])

## Year
q = data_3['Year'].quantile(0.01)
data_4 = data_3[data_3['Year']>q]
#sns.displot(data_4['Year'])

## show data cleaned
data_cleaned = data_4.reset_index(drop = True)
print("\n\n\n#############################  data without outliers #############################")
print(data_cleaned.describe(include= 'all'))
#plt.show()

## see nature of the data
f, (ax1,ax2,ax3) = plt.subplots(1,3, sharey = True, figsize = (15,3))
ax3.scatter(data_cleaned['Mileage'],data_cleaned['Price'])
ax3.set_title("Price and Mileage")
ax2.scatter(data_cleaned['EngineV'],data_cleaned['Price'])
ax2.set_title("Price and EngineV")
ax1.scatter(data_cleaned['Year'],data_cleaned['Price'])
ax1.set_title("Price and Year")

## get log of price to see linearity
log_price = np.log(data_cleaned['Price'])
data_cleaned['Log_price'] = log_price 

f2, (axi1,axi2,axi3) = plt.subplots(1,3, sharey = True, figsize = (15,3))
axi3.scatter(data_cleaned['Mileage'],data_cleaned['Log_price'])
axi3.set_title("Price and Mileage")
axi2.scatter(data_cleaned['EngineV'],data_cleaned['Log_price'])
axi2.set_title("Price and EngineV")
axi1.scatter(data_cleaned['Year'],data_cleaned['Log_price'])
axi1.set_title("Price and Year")
plt.show()

## check for multicollinearity 
variable = data_cleaned[['Mileage','Year','EngineV']]
vif = pd.DataFrame()
vif['VIF'] = [variance_inflation_factor(variable.values, i) for i in range(variable.shape[1])]
vif['Features'] = variable.columns
print("\n\n\n",vif)
data_without_multicollinearity = data_cleaned.drop(['Year'],axis = 1) ## remove Year column
data_without_multicollinearity = data_without_multicollinearity.drop(['Price'],axis = 1)

## crete dummy variables
data_with_dummies = pd.get_dummies(data_without_multicollinearity, drop_first = True)
print(data_with_dummies.head())


## rearange columns 
print(data_with_dummies.columns.values)

cols =  ['Log_price' ,'Mileage', 'EngineV' ,'Brand_BMW', 'Brand_Mercedes-Benz'
 ,'Brand_Mitsubishi' ,'Brand_Renault', 'Brand_Toyota' ,'Brand_Volkswagen',
 'Body_hatch' ,'Body_other' ,'Body_sedan' ,'Body_vagon', 'Body_van',
 'Engine Type_Gas' ,'Engine Type_Other' ,'Engine Type_Petrol',
 'Registration_yes']

data_preprocessed = data_with_dummies[cols]

print(data_preprocessed.head())


## get linear regression

## get data
targets = data_preprocessed['Log_price'] 
inputs = data_preprocessed.drop(['Log_price'],axis = 1)

## scale data
scaler = StandardScaler()
scaler.fit(inputs)
inputs_scaled = scaler.transform(inputs)
x_train, x_test, y_train, y_test = train_test_split(inputs_scaled,targets,test_size = 0.2,random_state = 365)

## set linear regression 
reg = LinearRegression()
reg.fit(x_train,y_train)

## predict values 
y_hat = reg.predict(x_train)
plt.scatter(y_train,y_hat)
plt.xlabel('Targets (y_train)',size = 18)
plt.ylabel('Predictions (y_hat)',size = 18)
plt.xlim(6,13)
plt.ylim(6,13)

## show residual error 
sns.displot(y_train-y_hat)
plt.title('Residual PDF',size = 18)
plt.show()

## get R-squared, weights and bias
R2 = reg.score(x_train,y_train)
bias = reg.intercept_

## show summary of regression
reg_summary = pd.DataFrame(inputs.columns.values,columns = ['Features'])
reg_summary['weights'] = reg.coef_
print('####################### summary of regression ##########################')
print(reg_summary)

## test the regression model
y_hat_test = reg.predict(x_test)
plt.scatter(y_test, y_hat_test, alpha = 0.2)
plt.xlabel('Targets y_test')
plt.ylabel('Predictions y_hat_test')
plt.xlim(6,13)
plt.ylim(6,13)
plt.show()

## dataframe performance
df_pf = pd.DataFrame(np.exp(y_hat_test), columns = ['Predict']) ## predict values
y_test = y_test.reset_index(drop = True) ## target values
df_pf['Targets'] =  np.exp(y_test) ## target values
df_pf['Residual'] = df_pf['Targets'] - df_pf['Predict'] ## Residual
df_pf['Difference%'] = np.absolute(df_pf['Residual']/df_pf['Targets'] *100)
print(df_pf.describe())

## set pandas options for visualization data
pd.options.display.max_rows = 99
pd.set_option('display.float_format', lambda x: '%.2f' % x)
print(df_pf.sort_values(by = ['Difference%']) )





