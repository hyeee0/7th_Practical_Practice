import pandas as pd
import numpy as np

# 1. 데이터 load

X_train = pd.read_csv('../Data/T2/X_train.csv')
y_train = pd.read_csv('../Data/T2/y_train.csv')
X_test = pd.read_csv('../Data/T2/X_test.csv')

print(X_train.head())

# 2.탐색적 데이터 분석 (EDA)

print(X_train.isnull().sum())
print(y_train.isnull().sum())
print(X_test.isnull().sum())

# 3. 목표 변수에 부적합한 설명 변수 제거

X_train = X_train.drop(['CustomerId', 'Surname'], axis=1)
X_test_id = X_test['CustomerId'].copy()
X_test = X_test.drop(['CustomerId', 'Surname'], axis=1)

# 4. 라벨 인코딩 -> 문자를 숫자로 맵핑

X_train.select_dtypes('object')

label_encoding_features = ['Geography', 'Gender']

from sklearn.preprocessing import LabelEncoder

for feature in label_encoding_features:
    le = LabelEncoder()
    X_train[feature] = le.fit_transform(X_train[feature])
    X_test[feature] = le.fit_transform(X_test[feature])

print(X_train.info())
print(X_train.head())

# 5. 범주형 변수 더미변수로 변경
# -> 더미변수는 회귀분석에서 범주형 변수를 처리하기 위해 필요 --> 회귀 분석모델은 숫자 데이터를 사용
# -> 범주형 변수를 더미 변수로 변환하여 각 범주를 대표하는 이진 변수로 표현
# 카테고리 get_dummies 변경 Geography, Gender, NumOfProducts, HasCrCard, IsActiveMember

categorical_features = ['NumOfProducts', 'HasCrCard', 'IsActiveMember']

for feature in categorical_features:
    X_train[feature] = X_train[feature].astype('category')
    X_test[feature] = X_test[feature].astype('category')

X_train = pd.get_dummies(X_train)
X_test = pd.get_dummies(X_test)

print(X_test.head())

#6. 파생변수 변경(5계층으로 범주화)
# X_train['CreditScore'].value_counts()
# pd.qcut : 같은 갯수로 구간 나누기
# pd.cut : 같은 길이로 구간 나누기

X_train['CreditScore_qcut'] = pd.qcut(X_train['CreditScore'], 5, labels=False)
X_test['CreditScore_qcut'] = pd.qcut(X_test['CreditScore'], 5, labels=False)

# 7. 수치데이터 정규화(Standardization) 스케일링 CreditScore, Balance, EstimatedSalary

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train.iloc[:, 1], random_state=2022,
                                                      stratify=y_train.iloc[:, 1])

# 8. 모델링

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier

model1 = LogisticRegression()
model1.fit(X_train, y_train)
predicted1 = model1.predict_proba(X_valid)

model2 = RandomForestClassifier()
model2.fit(X_train, y_train)
predicted2 = model2.predict_proba(X_valid)

model3 = DecisionTreeClassifier()
model3.fit(X_train, y_train)
predicted3 = model3.predict_proba(X_valid)

model4 = VotingClassifier(estimators= [('logistic', model1), ('random', model2)], voting='soft')
model4.fit(X_train, y_train)
predicted4 = model4.predict_proba(X_valid)

print(y_valid.shape, predicted1.shape, predicted2.shape, predicted3.shape)

# 9. 모델링 평가
from sklearn.metrics import roc_auc_score
print('로지스틱 회귀분석 점수 : {}'.format(roc_auc_score(y_valid, predicted1[:, 1])))
print('랜덤포레스트 회귀분석 점수 : {}'.format(roc_auc_score(y_valid, predicted2[:, 1])))
print('의사결정나무 회귀분석 점수 : {}'.format(roc_auc_score(y_valid, predicted3[:, 1])))
print('앙상블 보팅 회귀분석 점수 : {}'.format(roc_auc_score(y_valid, predicted4[:, 1])))

# 10. GridSearchCV 적용(최상의 모델을 찾은 후 최종 모델을 훈련)
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [5, 7],
}

gs = GridSearchCV(estimator = RandomForestClassifier(), param_grid= param_grid, cv = 10)
gs.fit(X_train, y_train)
print(gs.best_params_)

# 11. 적합한 하이퍼파라미터 설정 후 평가
model5 = RandomForestClassifier(max_depth=7 , n_estimators=150)
model5.fit(X_train, y_train)
predicted5 = model5.predict_proba(X_valid)

print('로지스틱 회귀분석 점수 : {}'.format(roc_auc_score(y_valid, predicted1[:, 1])))
print('랜덤포레스트 회귀분석 점수 : {}'.format(roc_auc_score(y_valid, predicted2[:, 1])))
print('의사결정나무 회귀분석 점수 : {}'.format(roc_auc_score(y_valid, predicted3[:, 1])))
print('앙상블 보팅 회귀분석 점수 : {}'.format(roc_auc_score(y_valid, predicted4[:, 1])))
print('GridSearchCV 적용 랜덤포레스트 점수 : {}'.format(roc_auc_score(y_valid, predicted5[:, 1])))

print(X_test_id.shape)

# 12. 최종 예측 모델 적용 후 평가
model6 = RandomForestClassifier(max_depth=7, n_estimators=150)
model6.fit(X_train, y_train)
predicted6 = model6.predict_proba(X_test)

result = pd.DataFrame({'CustomerID' : X_test_id, 'Exited' : predicted6[:, 1]})
result.to_csv('y_test.csv', index=False)


def result_validate(result):
    y_test = pd.read_csv('../Data/T2/test_label/y_test.csv')
    expected = y_test['Exited']
    predicted = result['Exited']

    print('ROC AUC score : ', roc_auc_score(expected, predicted))


result_validate(result)