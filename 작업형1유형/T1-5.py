"""

조건에 맞는 데이터 표준편차 구하기

- 주어진 데이터 중 basic1.csv에서 'f4'컬럼 값이 'ENFJ'와 'INFP'인 'f1'의 표준편차 차이를 절대값으로 구하시오
- 데이터셋 : basic1.csv
- 오른쪽 상단 copy&edit 클릭 -> 예상문제 풀이 시작

"""
import pandas as pd
import numpy as np

# 라이브러리 및 데이터 불러오기

df = pd.read_csv('../Data/T1-5/basic1.csv')
print(df.head())

# 조건에 맞는 데이터 (ENFJ, INFP)

df_ENFJ = df[df['f4'] == 'ENFJ']
df_INFP = df[df['f4'] == 'INFP']

print(df_ENFJ.head())

# 조건에 맞는 f1의 표준편차 (ENFJ, INFP) / 표준편차 std()

entj = df_ENFJ['f1'].std()
print(entj) # 17.727097901235837

infp = df_INFP['f1'].std() # 23.586719427112648
print(infp)

# 두 표준편차 차이 절대값 출력
print(np.abs(entj-infp)) # 5.859621525876811