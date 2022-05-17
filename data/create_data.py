import pandas as pd
import numpy as np
import os

def __createSushiX__():
    
    data_customer = pd.read_csv('data/sushi/sushi3.udata', delimiter='\t',
                               header = None, usecols=range(0,10))
    data_customer.columns = ['ID','Gender','Age group','Time for survey',
                             'Prefecture young','Region young','East/west young',
                             'Prefecture now','Region now','East/west now']
    
    age = pd.get_dummies(data_customer['Age group'])
    regionYoung = pd.get_dummies(data_customer['East/west young'])
    regionNow = pd.get_dummies(data_customer['East/west now'])
    data_customer = data_customer.drop(['ID','Age group', 'Prefecture young',
                                        'Region young', 'East/west young', 
                                        'Prefecture now', 'Region now',
                                        'East/west now'], axis=1)
    
    custumer_info = pd.concat([data_customer,age,regionYoung,regionNow], axis=1)
    return(custumer_info)

def createSushiComplete():
    
    raw_sushi = pd.read_csv('data/sushi/sushi3a.5000.10.order', delimiter=' ',
                              header = None, skiprows=1, usecols=range(2,12))
    raw_sushi = raw_sushi+1
    
    sushi_complete = pd.concat([__createSushiX__(),raw_sushi], axis=1)
    os.remove('data/sushiComplete.txt')
    sushi_complete.to_csv('data/sushiComplete.txt', header=None, index=None, sep=',', mode='a')
    
def createSushiPartial():
    
    raw_sushi = pd.read_csv('data/sushi/sushi3b.5000.10.order', delimiter=' ',
                              header = None, skiprows=1, usecols=range(2,12))
    raw_sushi = raw_sushi+1
    
    sushi_partial = pd.concat([__createSushiX__(),raw_sushi], axis=1)
    os.remove('data/sushiPartial.txt')
    sushi_partial.to_csv('data/sushiPartial.txt', header=None, index=None, sep=',', mode='a')
    
def createNascar():
    
    raw_nascar = pd.read_csv('data/nascar/nascar2002.txt', delimiter=' ').astype(int)
    nbRankedMax = raw_nascar['Place'].max()
    nbRaces = raw_nascar['Race'].max()
    
    data_nascar =  np.zeros((nbRaces, nbRankedMax))
    
    for i in range(0, raw_nascar.shape[0]):
        data_nascar[raw_nascar.iloc[i]['Race']-1, raw_nascar.iloc[i]['Place']-1] = int(raw_nascar.iloc[i]['DriverID'])
    
    data_nascar = data_nascar.astype(int)
    os.remove('data/nascar.txt')
    np.savetxt('data/nascar.txt', data_nascar, fmt='%d', delimiter = ',')