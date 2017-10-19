import pandas as pd
import numpy as np
# Given
Coupon = 0.3168
Freq = 1
CDR = 0.03833
RFR = 0.0293 #risk free rate
# Assumption
SEV = 0.75
AssetFunding = 0.05
AssetSev = 0.85
#Need to compute
SMM = 1 - np.power(1-CDR,1/Freq)
AssetPrice = 0
Price = 0
CumDefault = 0
CumDefault = 0
CumLoss = 0
CheckCumSev = 0
columns = ['Period','Balance','Interest','Principal',
'Default','Loss','CashFlow','Discount','AssetCashFlow',
'Asset','Discounting']


PeriodMax = 19

df = pd.DataFrame(np.zeros((PeriodMax,11)),columns=columns)
df['Period'] = range(0,PeriodMax)


df.loc[0,'Balance'] = 1000000
df.loc[0,'Discount'] = 1/np.power(1+RFR/Freq,df.loc[0,'Period'])

for index in df.index[1:]:
    df.loc[index,'Interest'] = df.loc[index-1,'Balance'] * Coupon / Freq
    df.loc[index,'Default'] = df.loc[index-1,'Balance'] * SMM
    df.loc[index,'Loss'] = df.loc[index,'Default'] * SEV
    df.loc[index,'Principal'] = df.loc[index,'Default'] - df.loc[index,'Loss']
    df.loc[index,'CashFlow'] = df.loc[index,'Interest'] + df.loc[index,'Principal']
    df.loc[index,'Discount'] = 1/np.power(1+RFR/Freq,df.loc[index,'Period'])
    df.loc[index,'Asset'] = 1/np.power(1+AssetFunding/Freq,df.loc[index,'Period'])
    