import pandas as pd
import numpy as np
from scipy.optimize import fsolve
import warnings
warnings.filterwarnings("ignore")

def cashDvideUpb(cdr,*args):
    '''
    args must be tuple (upb,rate,sev,rfr,freq,PeriodStop)
    '''
    upb,rate,sev,rfr,freq,PeriodStop = args
    Interest = [0]
    Balance = [upb]
    Default = [0]
    Loss = [0]
    Principal = [0]
    CashFlow = [0]
    Discount = 1/np.power(1+rfr/freq,range(0,PeriodStop+1))
    #print(Discount)
    SMM = 1 - np.power(1-cdr,1/freq)
    for i in range(1,PeriodStop+1):
        Interest.append(Balance[i-1]*rate/freq)
        Default.append(Balance[i-1] * SMM)
        Loss.append(Default[i] * sev)
        Principal.append(Default[i] - Loss[i])
        CashFlow.append(Interest[i] + Principal[i]) 
        Balance.append(Balance[i-1] - Principal[i] - Loss[i])
        if i == PeriodStop:
            Principal[i] = Balance[i-1] - Loss[i]
            CashFlow[i] = Interest[i] + Principal[i]
            Balance[i] = Balance[i-1] - Principal[i] - Loss[i] 
    CashFlow = np.array(CashFlow)
    Discount = np.array(Discount)
    return sum(CashFlow * Discount)/upb*100 - 100

def getCashflow(args):
    '''
    upb : initial balance
    rate : coupon
    cdr : constant d
    sev :
    rfr :
    freq :
    PeriodMax : 
    '''
    upb,rate,cdr,sev,rfr,freq,PeriodStop,PeriodMax = args
    SMM = 1 - np.power(1-cdr,1/freq)
    # initial
    df = pd.DataFrame(np.zeros((PeriodMax,len(columns))),columns=columns)
    df['Period'] = range(0,PeriodMax)
    df.loc[0,'Balance'] = upb
    df['Discount'] = getDiscount(range(0,PeriodMax),rfr,freq)
    df.loc[0,'AssetDiscounting'] = 1 

    for index in df.index[1:]:
        df.loc[index,'Interest'] = df.loc[index-1,'Balance'] * rate / freq
        df.loc[index,'Default'] = df.loc[index-1,'Balance'] * SMM
        df.loc[index,'Loss'] = df.loc[index,'Default'] * sev
        df.loc[index,'Principal'] = df.loc[index,'Default'] - df.loc[index,'Loss']
        df.loc[index,'CashFlow'] = df.loc[index,'Interest'] + df.loc[index,'Principal']
        df.loc[index,'AssetDiscounting'] = 1/np.power(1+AssetFunding,index)
        df.loc[index,'Balance'] = df.loc[index-1,'Balance'] - df.loc[index,'Principal'] - df.loc[index,'Loss']
        
        if index == PeriodStop:
            df.loc[index,'Principal'] = df.loc[index-1,'Balance'] - df.loc[index,'Loss']
            df.loc[index,'CashFlow'] = df.loc[index,'Interest'] + df.loc[index,'Principal']
            df.loc[index,'AssetDiscounting'] = 1/np.power(1+AssetFunding,index)
            df.loc[index,'Balance'] = df.loc[index-1,'Balance'] - df.loc[index,'Principal'] - df.loc[index,'Loss']
            break
    Price = sum(df.loc[:,'CashFlow']*df.loc[:,'Discount'])/df.loc[0,'Balance']*100
    CumDefault = sum(df.loc[:,'Default'])/df.loc[0,'Balance']
    df.loc[index,"AssetCashFlow"] = df.loc[0,'Balance']*((1 - CumDefault) +AssetSev*(1-AssetSev)*CumDefault)
    CumLoss = sum(df.loc[:,'Loss'])/df.loc[0,'Balance']
    CheckCumSev = CumLoss/CumDefault
    AssetPrice = sum(df.loc[:,'AssetCashFlow']*df.loc[:,'AssetDiscounting'])/df.loc[0,'Balance']*100
    return [df,Price,CumDefault,CheckCumSev,AssetPrice]

def getDiscount(timeV,rfr,freq):
    return 1/np.power(1+rfr/freq,timeV) #timeV could be a vector or a scalar


if __name__ == "__main__":
    # Given
    upb = 1000000
    PeriodMax = 19
    PeriodStop = 5
    Coupon = 0.316754
    Freq = 1
    #CDR = 0.383306890946272
    RFR = 0.0293 #risk free rate
    # Assumption
    SEV = 0.75
    AssetFunding = 0.05
    AssetSev = 0.85
    #Need to compute
    AssetPrice = 0
    Price = 0
    CumDefault = 0
    CumDefault = 0
    CumLoss = 0
    CheckCumSev = 0
    columns = ['Period','Balance','Interest','Principal',
    'Default','Loss','CashFlow','Discount','AssetCashFlow',
    'AssetDiscounting']
    ##calculate CDR
    args = (upb,Coupon,SEV,RFR,Freq,PeriodStop)
    CDR = fsolve(cashDvideUpb,[0.3],args=args)[0]
    
    #getCashflow(upb,rate,cdr,sev,rfr,freq,PeriodStop,PeriodMax=19)
    args = (upb,Coupon,CDR,SEV,RFR,Freq,PeriodStop,PeriodMax)
    [df,Price,CumDefault,CheckCumSev,AssetPrice] = getCashflow(args)