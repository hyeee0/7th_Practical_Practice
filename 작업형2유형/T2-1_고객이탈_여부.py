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
