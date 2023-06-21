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
    X_train[feature] = X_train[feature].astype('c')


