## [1] 대응(쌍체)표본 T검정

# 주어진 데이터는 고혈압 환자 치료 전후의 혈압이다.<br>
# 해당 치료가 효과가 있는지 대응(쌍체)표본 t-검정을 진행하시오

# *   귀무가설(H0):  μ >= 0<br>
# *   대립가설(H1):  μ < 0<br>
# *   μ= (치료 후 혈압 - 치료 전 혈압)의 평균<br>
# *   유의수준: 0.05<br>
#
# 1. μ의 표본평균은?(소수 둘째자리까지 반올림)
# 2. 검정통계량 값은?(소수 넷째자리까지 반올림)
# 3. p-값은?(소수 넷째자리까지 반올림)
# 4. 가설검정의 결과는? (유의수준 5%)

import pandas as pd
from scipy import stats #정규

df = pd.read_csv('../Data/archive/high_blood_pressure.csv')
# μ= (치료 후 혈압 - 치료 전 혈압)의 평균

df['diff'] = df['bp_post'] - df['bp_pre']
print(round(df['diff'].mean(),2)) # μ의 표본평균은?(소수 둘째자리까지 반올림)

st, pv = stats.ttest_rel(df['bp_post'], df['bp_pre'], alternative='less') # 대응 표본 t검정 -> ttest_rel()
#alternative='less' -> 가설에서 작은지가 궁금하니깐
print(round(st, 4)) # 검정통계량 값은?(소수 넷째자리까지 반올림)
print(round(pv, 4)) # p-값은?(소수 넷째자리까지 반올림)

# p_value가 0.0016 < 0.05 이므로 귀무가설 기각, 대립가설 채택