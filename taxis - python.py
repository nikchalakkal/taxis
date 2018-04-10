# Taxi Data
import pandas as pd
df=pd.read_csv(r'C:\Users\YDUAN35\Documents\Emory MSBA\Class\Machine Learning 2\Project\yellow_tripdata_2016-01.csv')
df.head(5)
df.dtypes

# Convert timestamp column
df['tpep_dropoff_datetime']=pd.to_datetime(df['tpep_dropoff_datetime'], format='%Y-%m-%d %H:%M:%S')
df['tpep_pickup_datetime']=pd.to_datetime(df['tpep_pickup_datetime'], format='%Y-%m-%d %H:%M:%S')
df['duration']=df['tpep_dropoff_datetime']-df['tpep_pickup_datetime']
df['duration']=df['duration'].dt.total_seconds()/60

# Remove Extreme Values
df=df[df['duration']>1]
df.isnull().sum()
df=df[df['duration']<180]
df=df[df['total_amount']>0]
df=df[df['total_amount']<1000]
df=df[df['trip_distance']<500]
df=df[df['fare_amount']>0]

from sklearn.preprocessing import LabelEncoder
# encode categorical to numeric
lb_make = LabelEncoder()
df['store_and_fwd_flag'] = lb_make.fit_transform(df['store_and_fwd_flag'])
df['VendorID']=df['VendorID'].astype('category')
df['RatecodeID']=df['RatecodeID'].astype('category')
df['store_and_fwd_flag']=df['store_and_fwd_flag'].astype('category')
df['payment_type']=df['payment_type'].astype('category')

X=df[['trip_distance','RatecodeID','tolls_amount','mta_tax','payment_type','extra','passenger_count']]
y=df['duration']

# Models On Duration
# Train Test Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0)

# Linear Regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
slr2 = LinearRegression()
slr2.fit(X_train, y_train)
y_train_pred = slr2.predict(X_train)
y_test_pred = slr2.predict(X_test)

print('Slope: %.3f', slr2.coef_)
from sklearn.metrics import mean_squared_error

# Lasso
from sklearn.linear_model import Lasso

lasso = Lasso(alpha=0.1)
# alpha = [0.5, 1.0]
lasso.fit(X_train, y_train)

y_test_pred = lasso.predict(X_test)

print(mean_squared_error(y_test, y_test_pred)**(1/2))

# Ridge
from sklearn.linear_model import Ridge

ridge = Ridge(alpha=0.1)
# alpha = [0.5, 1.0]
ridge.fit(X_train, y_train)

y_test_pred = ridge.predict(X_test)
print(mean_squared_error(y_test, y_test_pred)**(1/2))

# Neural Nets
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scale = scaler.transform(X_train)
X_test_scale = scaler.transform(X_test)

from sklearn.neural_network import MLPRegressor
mlp = MLPRegressor(hidden_layer_sizes=(20), max_iter = 200, activation='logistic', learning_rate='adaptive')
# param_grid = {"max_iter": [100,200],
             # "activation": ['identity', 'logistic'],
             # "learning_rate":['constant', 'invscaling', 'adaptive']}
#mlp = GridSearchCV(mlp, param_grid = param_grid)
mlp = mlp.fit(X_train_scale,y_train)
predictions = mlp.predict(X_test_scale)
print(mean_squared_error(y_test, predictions)**(1/2))

# Regression Tree
from sklearn.tree import DecisionTreeRegressor 
from sklearn.model_selection import cross_val_score

tree = DecisionTreeRegressor(max_depth = 9, min_samples_split = 10, splitter = 'best')
# param_grid = {"max_depth": range(0,30),
             # "min_samples_split": range(10,20),
             # "splitter":['best', 'random']}
# tree = GridSearchCV(tree, param_grid = param_grid)
tree= tree.fit(X_train, y_train)
y_pred= tree.predict(X_test)
print(mean_squared_error(y_test, y_pred)**(1/2))

# Adaboost
from sklearn.ensemble import AdaBoostRegressor
regr_2 = AdaBoostRegressor(DecisionTreeRegressor(max_depth=9),
                           random_state=1)
regr_2= regr_2.fit(X_train, y_train)
y_pred = regr_2.predict(X_test)
print(mean_squared_error(y_test, y_pred)**(1/2))

# Models on total_amount
X=df[['trip_distance','RatecodeID','tolls_amount','mta_tax','payment_type','extra','duration']]
y=df['total_amount']
# Rerun the same code as above
