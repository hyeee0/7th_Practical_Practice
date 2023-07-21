'''

결측치 제거 및 그룹 합계에서 조건에 맞는 값 찾아 출력
주어진 데이터 중 basic1.csv에서 'f1'컬럼 결측 데이터를 제거하고, 'city'와 'f2'을 기준으로 묶어 합계를 구하고, 'city가 경기이면서 f2가 0'인 조건에 만족하는 f1 값을 구하시오
데이터셋 : basic1.csv
오른쪽 상단 copy&edit 클릭 -> 예상문제 풀이 시작

'''
import pandas as pd
import numpy as np

df = pd.read_csv('../Data/T1-6.csv')

# f1컬럼 결측치 제거
df = df[~df['f1'].isnull()]

# 그룹 합계 계산
df2 = df.groupby(['city', 'f2']).sum()
df2
print(df2.iloc[0]['f1']) # 경기이면서 0인 f1값




