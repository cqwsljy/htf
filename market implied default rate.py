import pandas as pd
import numpy as np

def getCashflow(upb,rate,cdr,sev):
    smm = 1 - np.power(1-cdr,1/Freq)
    inte = upb * rate/Freq
    defa = upb * smm
    loss = defa * sev
    prin = defa - loss
    bal = upb - prin - defa
    cashflow = inte + prin
    return  [bal,inte,defa,prin,cashflow,loss]

def getDiscount(timeV,rfr,freq):
    return 1/np.power(1+rfr/freq,timeV)

# Given
PeriodMax = 19
PeriodStop = 5
Coupon = 0.316754
Freq = 1
CDR = 0.383306890946272
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
'AssetDiscounting']

# initial
df = pd.DataFrame(np.zeros((PeriodMax,len(columns))),columns=columns)
df['Period'] = range(0,PeriodMax)
df.loc[0,'Balance'] = 1000000
df['Discount'] = getDiscount(range(0,PeriodMax),RFR,Freq)
df.loc[0,'AssetDiscounting'] = 1 
'''
for index in df.index[1:]:
    df.loc[index,'Interest'] = df.loc[index-1,'Balance'] * Coupon / Freq
    df.loc[index,'Default'] = df.loc[index-1,'Balance'] * SMM
    df.loc[index,'Loss'] = df.loc[index,'Default'] * SEV
    df.loc[index,'Principal'] = df.loc[index,'Default'] - df.loc[index,'Loss']
    df.loc[index,'CashFlow'] = df.loc[index,'Interest'] + df.loc[index,'Principal']
    df.loc[index,'Discount'] = 1/np.power(1+RFR/Freq,df.loc[index,'Period'])
    df.loc[index,'AssetDiscounting'] = 1/np.power(1+AssetFunding,index)
    df.loc[index,'Balance'] = df.loc[index-1,'Balance'] - df.loc[index,'Principal'] - df.loc[index,'Loss']
    if index == PeriodStop:
        df.loc[index,'Principal'] = df.loc[index-1,'Balance'] - df.loc[index,'Loss']
        df.loc[index,'CashFlow'] = df.loc[index,'Interest'] + df.loc[index,'Principal']
        df.loc[index,'Discount'] = 1/np.power(1+RFR/Freq,df.loc[index,'Period'])
        df.loc[index,'AssetDiscounting'] = 1/np.power(1+AssetFunding,index)
        df.loc[index,'Balance'] = df.loc[index-1,'Balance'] - df.loc[index,'Principal'] - df.loc[index,'Loss']
        break
'''
for index in df.index[1:]:
    upb = df.loc[index-1,'Balance']
    [bal,inte,defa,prin,cashflow,loss] = getCashflow(upb,Coupon,CDR,SEV) #getCashflow(upb,rate,cdr,sev)
    df.loc[index,'Interest'] = inte
    df.loc[index,'Default'] = defa
    df.loc[index,'Loss'] = loss
    df.loc[index,'Principal'] = prin
    df.loc[index,'CashFlow'] = cashflow
    df.loc[index,'Balance'] = bal
    df.loc[index,'AssetDiscounting'] = 1/np.power(1+AssetFunding,index)
    if index == PeriodStop:
        df.loc[index,'Principal'] = df.loc[index-1,'Balance'] - df.loc[index,'Loss']
        df.loc[index,'CashFlow'] = df.loc[index,'Interest'] + df.loc[index,'Principal']
        df.loc[index,'Discount'] = 1/np.power(1+RFR/Freq,df.loc[index,'Period'])
        df.loc[index,'AssetDiscounting'] = 1/np.power(1+AssetFunding,index)
        df.loc[index,'Balance'] = df.loc[index-1,'Balance'] - df.loc[index,'Principal'] - df.loc[index,'Loss']
        break
Price = sum(df.loc[:,'CashFlow']*df.loc[:,'Discount'])/df.loc[0,'Balance']*100
CumDefault = sum(df.loc[:,'Default'])/df.loc[0,'Balance']
df.loc[index,"AssetCashFlow"] = df.loc[0,'Balance']*((1 - CumDefault) +AssetSev*(1-AssetSev)*CumDefault)
CumLoss = sum(df.loc[:,'Loss'])/df.loc[0,'Balance']
CheckCumSev = CumLoss/CumDefault
AssetPrice = sum(df.loc[:,'AssetCashFlow']*df.loc[:,'AssetDiscounting'])/df.loc[0,'Balance']*100