"""
이상치를 찾아라
데이터에서 IQR을 활용해 Fare컬럼의 이상치를 찾고, 이상치 데이터의 여성 수를 구하시오

"""
# 라이브러리 및 데이터 불러오기
import pandas as pd

train = pd.read_csv('../Data/Titanic/train.csv')
test = pd.read_csv('../Data/Titanic/test.csv')

print(train)
print(test)

# 간단한 탐색적 데이터 분석(EDA)

print(train.shape, test.shape)
print(train.info())
print(test.info())
print(train.isnull().sum())
print(train.head())

# IQR 구하기 / Fare컬럼의 이상치를 구할거니깐 ['Fare']컬럼을 보자

Q1 = train['Fare'].quantile(.25)
Q3 = train['Fare'].quantile(.75)

IQR = Q3 - Q1
print(Q1 - 1.5*IQR)
print(Q3 + 1.5*IQR)

# 이상치 데이터 구하기
data1 = train[train['Fare'] < (Q1 - 1.5*IQR)]
data2 = train[train['Fare'] > (Q3 + 1.5*IQR)]
print(len(data1), len(data2))

# 이상치 데이터에서 여성 수 구하기, 출력하기 print()

print(sum(data2['Sex'] == 'female'))