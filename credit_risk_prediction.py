#Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import pickle

# read dataset
og_dataset = pd.read_csv('data/loan_data_2007_2014.csv')

# copy of the dataset
df = og_dataset.copy()

# Assigning Target Variable
# Bad loans = Charged Off, Default, Late (31-120 days), Does not meet the credit policy. Status:Charged Off
df['bad_loan'] = np.where(df['loan_status'].isin(['Charged Off','Default','Late (31-120 days)','Does not meet the credit policy. Status:Charged Off']), 1, 0)
df[['loan_status','bad_loan']]
target_names = ['good loan', 'bad loan']

# Data Prepocessing
# Drop kolom yang terkait dengan data setelah peminjaman dilakukan
drop_cols = ['issue_d', 'loan_status', 'pymnt_plan', 'out_prncp', 'out_prncp_inv', 'total_pymnt', 'total_pymnt_inv',
                'total_rec_prncp', 'total_rec_int', 'total_rec_late_fee', 'recoveries', 'collection_recovery_fee',
                'last_pymnt_d', 'last_pymnt_amnt', 'next_pymnt_d','last_credit_pull_d','sub_grade','funded_amnt','funded_amnt_inv']
df.drop(columns=drop_cols,inplace=True)

# Drop Unnecessary Columns
df.drop(columns=['Unnamed: 0', 'id', 'member_id', 'addr_state', 'desc', 'earliest_cr_line', 'emp_length', 'emp_title', 'inq_last_6mths', 'mths_since_last_delinq', 'mths_since_last_record', 'title', 'url', 'zip_code'], inplace=True)

# Kolom dengan value kosong lebih dari 50%
null_cols = df.isnull().mean()
null_cols = null_cols[null_cols > 0.5]

# Kolom dengan univariate values
uni_cols = df.nunique()[(df.nunique()<2) | (df.nunique() == len(df))]

# Menghapus kolom yang tidak relevan
del_cols = set(list(uni_cols.index) + list(null_cols.index))
df.drop(columns=del_cols, inplace=True)

# Handling Missing Values
# Handling 'annual_inc' missing values with median
df['annual_inc'] = df['annual_inc'].fillna(df['annual_inc'].median())

# Handling 'delinq_2yrs', 'earliest_cr_line', 'inq_last_6mths', 'open_acc', 'pub_rec', 'total_acc' 'acc_now_delinq' missing values delete rows (data terlalu sedikit)
del_rows = df[df['delinq_2yrs'].isnull()].index
df.drop(del_rows,axis=0,inplace=True)
df.reset_index(drop=True,inplace=True)

# Handling 'revol_util' missing values with 0
df['revol_util'] = df['revol_util'].fillna(0)

# Handling 'collections_12_mths_ex_med' missing values with 0
df['collections_12_mths_ex_med'] = df['collections_12_mths_ex_med'].fillna(0)

# Handling 'tot_coll_amt','total_rev_hi_lim','tot_cur_bal' missing values with 0
df['tot_coll_amt'] = df['tot_coll_amt'].fillna(0)
df['total_rev_hi_lim'] = df['total_rev_hi_lim'].fillna(0)
df['tot_cur_bal'] = df['tot_cur_bal'].fillna(0)

# Features Data Types
# handling term data type
df['term'] = pd.to_numeric(df['term'].str.replace(' months', ''))

# Feature Extraction
numeric_features = list(df.select_dtypes(["float64" , "int64",'int32']).columns)
numeric_features.remove('bad_loan')
categorical_features = list(df.select_dtypes("object").columns)
target = 'bad_loan'


# One Hot Encoding for Categorical Features
enc = OneHotEncoder(handle_unknown='ignore')
cat_df = pd.DataFrame(enc.fit_transform(df[categorical_features]).toarray())
cat_df.columns = enc.get_feature_names_out(categorical_features)


# Standard Scaling for Numeric Features
ss = StandardScaler()
ss.fit(df[numeric_features])
ss_df = pd.DataFrame(ss.transform(df[numeric_features]), columns=numeric_features)

# Merge Final Features
final_df = pd.concat([ss_df, cat_df, df[target]], axis=1)

# Model Training
# Data Splitting
X = final_df.drop(columns=target)
y = final_df[target]
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
X_train.shape, X_test.shape

# Handling Imbalance Dataset
os = RandomOverSampler()
X_train_ros, y_train_ros = os.fit_resample(X_train, y_train)
y_train_series = pd.Series(y_train_ros)

# Model Building
# 1. Logistic Regression
LR= LogisticRegression(max_iter=600)
LR.fit(X_train_ros, y_train_ros)
y_pred_LR = LR.predict(X_test)
LR_acc = accuracy_score(y_test, y_pred_LR)

# 2. Decision Tree
DT = DecisionTreeClassifier()
DT.fit(X_train_ros, y_train_ros)
y_pred_DT = DT.predict(X_test)
DT_acc = accuracy_score(y_test, y_pred_DT)

# 3. Random Forest
RF = RandomForestClassifier()
RF.fit(X_train_ros, y_train_ros)
y_pred_RF = RF.predict(X_test)
RF_acc = accuracy_score(y_test, y_pred_RF)

# 4. KNN
KNN = KNeighborsClassifier()
KNN.fit(X_train_ros, y_train_ros)
y_pred_KNN = KNN.predict(X_test)
KNN_acc = accuracy_score(y_test, y_pred_KNN)

# 5. XGBoost
XGB = XGBClassifier()
XGB.fit(X_train_ros, y_train_ros)
y_pred_XGB = XGB.predict(X_test)
XGB_acc = accuracy_score(y_test, y_pred_XGB)

# plot accuracy score of all models
plt.figure(figsize=(10,5))
plt.bar(['LR','DT','RF','KNN','XGB'],[LR_acc,DT_acc,RF_acc,KNN_acc,XGB_acc])
for i, v in enumerate([LR_acc,DT_acc,RF_acc,KNN_acc,XGB_acc]):
    plt.text(i-.1, v+.01, str(round(v,3)))
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.title('Accuracy of Models')
plt.show()

# choose best model based on accuracy score
models_acc = [LR_acc,DT_acc,RF_acc,KNN_acc,XGB_acc]
models = ['LR','DT','RF','KNN','XGB']
best_model = models[models_acc.index(max(models_acc))]
print('Best Model: ',best_model)

# save model
pickle.dump(best_model, open('model/model.pkl','wb'))

