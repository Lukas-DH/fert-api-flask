import pickle
import pandas as pd
import sklearn

X_train,X_test,y_train,y_test= pd.read_pickle("X_train.pkl"),pd.read_pickle("X_test.pkl"),pd.read_pickle("y_train.pkl"),pd.read_pickle("y_test.pkl")
from sklearn.ensemble import RandomForestClassifier
RFC1=RandomForestClassifier(class_weight='balanced_subsample',max_depth=5,criterion='entropy',n_estimators=100)
RFC1.fit(X_train,y_train)
filename = 'RFC1opt.sav'
pickle.dump(RFC1,open(filename,'wb'))