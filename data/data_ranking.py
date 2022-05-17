import numpy as np
import inspect
import sys
import sklearn.preprocessing

def __getDataframe__(file, nbElementsX):
    
    data = np.genfromtxt(file, delimiter=',')
    nbElements = data.shape[1]
    data_x = sklearn.preprocessing.scale(data[:,0:nbElementsX])
    data_y = data[:,nbElementsX:nbElements].astype(int)
    data_name = file.split('/', 1)[1]
    data_name = data_name.split('.', 1)[0]
    data_name = data_name.title()
    return(data_x, data_y, data_name)
    
def getAuthorship():
    return(__getDataframe__('data/authorship.txt', 70))
    
def getBodyfat():
    return(__getDataframe__('data/bodyfat.txt', 7))

def getCallhousing():
    return(__getDataframe__('data/callhousing.txt', 4))

def getCpu():
    return(__getDataframe__('data/cpu.txt', 6))

def getElevators():
    return(__getDataframe__('data/elevators.txt', 9))

def getFried():
    return(__getDataframe__('data/fried.txt', 9))

def getGlass():
    return(__getDataframe__('data/glass.txt', 9))

def getHousing():
    return(__getDataframe__('data/housing.txt', 6))

def getIris():
    return(__getDataframe__('data/iris.txt', 4))
    
def getNascar():
    return(np.genfromtxt('data/nascar.txt', delimiter=',').astype(int))

def getPendigits():
    return(__getDataframe__('data/pendigits.txt', 16))

def getSegment():
    return(__getDataframe__('data/segment.txt', 18))
    
def getSushiComplete():
    return(__getDataframe__('data/sushiComplete.txt', 12))   
    
def getSushiPartial():
    return(__getDataframe__('data/sushiPartial.txt', 12))   

def getStock():
    return(__getDataframe__('data/stock.txt', 5))

def getVehicle():
    return(__getDataframe__('data/vehicle.txt', 18))

def getVowel():
    return(__getDataframe__('data/vowel.txt', 10))

def getWine():
    return(__getDataframe__('data/wine.txt', 13))

def getWisconsin():
    return(__getDataframe__('data/wisconsin.txt', 16))    
    
def getAll():
    """
    Get all datasets exccept nascar.
    """
    
    all_functions = inspect.getmembers(sys.modules[__name__], inspect.isfunction)
    all_data = list()
    
    functionsToExclude = ['__getDataframe__','getNascar','getAll',
                          'getAllIB','getAllGLM']
    
    for function in all_functions:
        
        #All the datasets.
        if function[0] in functionsToExclude:
            continue
        
        else:
            data = function[1]()
            all_data.append(data)
        
    return all_data

def getAllIB():
    
    """
    Get all datasets used to evaluate IB algorithm.
    """
    
    all_functions = inspect.getmembers(sys.modules[__name__], inspect.isfunction)
    all_data = list()
    
    functionsToExclude = ['__getDataframe__','getNascar','getAll',
                          'getAllIB','getAllGLM','getCallhousing','getCpu',
                          'getElevators','getFried', 'getPendigits','getSegment',
                          'getSushiComplete','getSushiPartial']
    
    for function in all_functions:
        
        #All the datasets.
        if function[0] in functionsToExclude:
            continue
        
        else:
            data = function[1]()
            all_data.append(data)
        
    return all_data

def getAllGLM():
    
    """
    Get all datasets used to evaluate GLM algorithm (regression).
    """
    
    all_functions = inspect.getmembers(sys.modules[__name__], inspect.isfunction)
    all_data = list()
    
    functionsToExclude = ['__getDataframe__','getNascar','getAll',
                          'getAllIB','getAllGLM','getSushiComplete','getSushiPartial',
                          'getAuthorship', 'getElevators', 'getFried', 'getPendigits',
                          'getSegment', 'getVehicle', 'getVowel', 'getWisconsin']
    
    for function in all_functions:
        
        #All the datasets.
        if function[0] in functionsToExclude:
            continue
        
        else:
            data = function[1]()
            all_data.append(data)
        
    return all_data