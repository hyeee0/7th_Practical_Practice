"""
왜도와 첨도 구하기

주어진 데이터 중 train.csv에서 'SalePrice'컬럼의 왜도와 첨도를 구한 값과,
 'SalePrice'컬럼을 스케일링(log1p)로 변환한 이후 왜도와 첨도를 구해 모두 더한 다음 소수점 2째자리까지 출력하시오

"""

# 라이브러리 및 데이터 불러오기
import pandas as pd
import numpy as np

df = pd.read_csv('../Data/T1-4/train.csv')
print(df.head())

# 'SalePrice'컬럼 왜도와 첨도계산
s1 = df['SalePrice'].skew() # 왜도 계상 df[].skew()
k1 = df['SalePrice'].kurt() # 첨도 계상 df[].kurt()
print('왜도 : ', s1)
print('첨도 :', k1)

# 'SalePrice'컬럼 로그변환

# 'SalePrice'컬럼 왜도와 첨도계산

# 모두 더한 다음 출력