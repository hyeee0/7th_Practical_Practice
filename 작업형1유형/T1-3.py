"""
결측치 처리

주어진 데이터에서 결측치가 80%이상 되는 컬럼은(변수는) 삭제하고,
 80% 미만인 결측치가 있는 컬럼은 'city'별 중앙값으로 값을 대체하고 'f1'컬럼의 평균값을 출력하세요!

"""

# 라이브러리 및 데이터 불러오기
import pandas as pd
import numpy as np

df = pd.read_csv('../Data/T1-3/basic1.csv')
print(df.head())

# EDA - 결측값 확인(비율 확인) -> 비율 확인하려면 isnull().sum() / shape[0] 을 해주면 된다.
print(df.isnull().sum())
print(df.shape)

print((df.isnull().sum()) / df.shape[0]) # 결측비율 확인

# 80%이상 결측치 컬럼, 삭제 -> ['f3']컬럼이 95% 결측치니깐 해당 컬럼을 삭제해준다
print('삭제 전 :', df.shape)
df = df.drop(['f3'], axis=1)
print('삭제 후 :', df.shape)

# 80%미만 결측치 컬럼, city별 중앙값으로 대체 -> 80%미만인 컬럼이 ['f1'] / 중앙값이니깐 median
print(df['city'].unique())

s = df[df['city'] == '서울']['f1'].median()
b = df[df['city'] == '부산']['f1'].median()
d = df[df['city'] == '대구']['f1'].median()
g = df[df['city'] == '경기']['f1'].median()
print(s, b, d, g)

df['f1'] = df['f1'].fillna(df['city'].map({'서울':s, '부산':b, '대구':d, '경기':g})) # f1 컬럼의 결측치를 각각의 city 중앙값으로 대체(map을 사용)

# f1 평균값 결과 출력
print(df['f1'].mean())