import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings(action='ignore')

train = pd.read_csv()
test = pd.read_csv()

all = pd.concat((train, test), axis=0)
all = all.drop([], axis=1)

print(all.info())

categorical = [col for col in all.columns if all[col].dtype == 'O']
numerical = [col for col in all.columns if all[col].dtype!= 'O']

from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, LabelEncoder
cols = []
label = LabelEncoder()

for col in cols:
    label.fit(list(all[col].values))
    all[col] = label.transform(list(all[col].values))
    if col == :
        onehot= OneHotEncoder()
        onehot_result = onehot.fit_transform(all[col].values.reshape(-1,1)).toarray()
        onehot_col = ['g'+str(i) for i in range(onehot_result.shape[-1])]
        onehot_df = pd.DataFrame(onehot_result, columns = onehot_col).reset_index(drop=True)
        all = pd.concat((all, onehot_df), axis=1)

print(all.describe)

cols = []
scaler = MinMaxScaler()
all[cols] = scaler.fit_transform(all[cols])

all[col] = np.log(all[col])

from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression

X = all[:len(train)]
y = all[]
test = all[len(train):]

X_train, X_test, y_train, y_test = train_test_split(test_size=0.2, random_state=42)

sample_model = GradientBoostingClassifier()
sample_model.fit(X_train, y_train)
proba = sample_model.predict_proba(X_test)

auc = roc_auc_score(y_test, proba[:, 1])
print(auc)

important = sample_model.feature_importances_
import_df = pd.DataFrame(important.reshape(1, -1), columns = [col for col in X.columns])
print(import_df)

candidate_col = []

pca = PCA(n_components=2)
tmp = all.copy()
pca_result = pca.fit_transform(tmp)
pca_col = ['pca' + str(i) for i in range(pca_result.shape[-1])]
pca_df = pd.DataFrame(pca_result, columns = pca_col)
all = pd.concat((all, pca_df), axis = 1).reset_index(drop=True)

all['mean']  = all[candidate_col].mean(axis=1)
all['min']  = all[candidate_col].min(axis=1)
all['max']  = all[candidate_col].max(axis=1)
all['std']  = all[candidate_col].std(axis=1)

X = all[:len(train)]
y = all[]
test = all[len(train):]

rfc = RandomForestClassifier()
abc = AdaBoostClassifier()
gbc= GradientBoostingClassifier()

rf_pred = np.zeros((len(X), 2))
accs, aucs = [], []
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for i, (train_idx, test_idx) in enumerate(skf.split(X, y)):
    X_train, X_test = X.iloc[train_idx, :], X.iloc[test_idx, :]
    y_train, y_test = y.iloc[test_idx], y.iloc[test_idx]
    
    model = rfc
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)
    
    acc = accuracy_score(y_test, pred)
    auc = roc_auc_score(y_test, proba[:, 1])
    
    accs.append(acc)
    auc.append(auc)
    rf_pred[test_idx] += proba
    
    print(f"{i}th acc score : {acc} | auc score : {auc}")
print(f"Mean auc score: {np.mean(aucs)}")


gb_pred = np.zeros((len(X), 2))
accs, aucs = [], []
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for i, (train_idx, test_idx) in enumerate(skf.split(X, y)):
    X_train, X_test = X.iloc[train_idx, :], X.iloc[test_idx, :]
    y_train, y_test = y.iloc[test_idx], y.iloc[test_idx]
    
    model = gbc
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)
    
    acc = accuracy_score(y_test, pred)
    auc = roc_auc_score(y_test, proba[:, 1])
    
    accs.append(acc)
    auc.append(auc)
    gb_pred[test_idx] += proba
    
    print(f"{i}th acc score : {acc} | auc score : {auc}")
print(f"Mean auc score: {np.mean(aucs)}")


ab_pred = np.zeros((len(X), 2))
accs, aucs = [], []
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for i, (train_idx, test_idx) in enumerate(skf.split(X, y)):
    X_train, X_test = X.iloc[train_idx, :], X.iloc[test_idx, :]
    y_train, y_test = y.iloc[test_idx], y.iloc[test_idx]
    
    model = abc
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)
    
    acc = accuracy_score(y_test, pred)
    auc = roc_auc_score(y_test, proba[:, 1])
    
    accs.append(acc)
    auc.append(auc)
    ab_pred[test_idx] += proba
    
    print(f"{i}th acc score : {acc} | auc score : {auc}")
print(f"Mean auc score: {np.mean(aucs)}")


final_train = np.zeros((len(X), 2))
final_test = np.zeros((5, len(X), 2))
accs, aucs = [], []
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for i, (train_idx, test_idx) in enumerate(skf.split(X, y)):
    X_train, X_test = X.iloc[train_idx, :], X.iloc[test_idx, :]
    y_train, y_test = y.iloc[test_idx], y.iloc[test_idx]
    
    model = VotingClassifier(estimators=[('rfc', rfc), ('gbc', gbc), ('abc', abc)], voting='soft', weights=[1,2,3])
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)
    
    acc = accuracy_score(y_test, pred)
    auc = roc_auc_score(y_test, proba[:, 1])
    
    accs.append(acc)
    auc.append(auc)
    final_train[test_idx] += proba
    final_test[i] += model.predict_proba(test)
    
    print(f"{i}th acc score : {acc} | auc score : {auc}")
print(f"Mean auc score: {np.mean(aucs)}")

final = np.mean(final_test, axis=0)

submission = pd.DataFrame({'col' : final[:, 1]})