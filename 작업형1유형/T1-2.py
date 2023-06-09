"""

이상치를 찾아라(소수점 나이)

주어진 데이터에서 이상치(소수점 나이)를 찾고 올림, 내림, 버림(절사)했을때 3가지 모두 이상치 'age' 평균을 구한 다음 모두 더하여 출력하시오

"""

# 라이브러리 및 데이터 불러오기
import pandas as pd
import numpy as np

df = pd.read_csv('../Data/T1-2/basic1.csv')
print(df)

# 소수점 데이터 찾기 (소수점 나이)
df = df[df['age'] - np.floor(df['age']) != 0] # 원래 값이랑 내림한 값이랑 0이면 그 값으 소수점이 아닌건데 0이 아닌건 그 값이 소수점이라는 뜻
print(df)

# 이상치를 포함한 데이터 올림, 내림, 버림의 평균값

m_ceil = np.ceil(df['age']).mean() # 올림의 평균값
m_floor = np.floor(df['age']).mean() # 내림의 평균값
m_trunc = np.trunc(df['age']).mean() # 버림의 평균값
print(m_ceil, m_floor, m_trunc)

# 평균값 더한 다음 출력
print(m_ceil + m_floor + m_trunc)
