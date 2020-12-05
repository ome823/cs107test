import numpy as np
from AutoDiff import AutoDiff as ad

def sin(AD_obj):
    '''
    >>> print(sin(ad.AutoDiff(np.pi)))
    0.0, -1.0
    
    '''
    try:
        return ad.AutoDiff(np.sin(AD_obj.val),AD_obj.der*np.cos(AD_obj.val))
    except AttributeError:
        return np.sin(AD_obj)

def cos(AD_obj):
    '''
    >>> print(cos(ad.AutoDiff(np.pi)))
    -1.0, 0.0
    
    '''
    try:
        return ad.AutoDiff(np.cos(AD_obj.val),-1*AD_obj.der*np.sin(AD_obj.val))
    except AttributeError:
        return np.cos(AD_obj)

def tan(AD_obj):
    '''
    >>> print(tan(AutoDiff(np.pi)))
    -0.0, 1.0
    
    '''
    try:
        return ad.AutoDiff(np.tan(AD_obj.val),AD_obj.der*(1/np.cos(AD_obj.val)**2))
    except AttributeError:
        return np.tan(AD_obj)


def sqrt(AD_obj):
    '''
    >>> print(sqrt(AutoDiff(100)))
    10.0, 0.05
    
    '''
    try:
        return ad.AutoDiff(np.sqrt(AD_obj.val), .5 * (1/AD_obj.val**.5) * AD_obj.der)
    except AttributeError:
        return np.sqrt(AD_obj)
    
    
    
def log(AD_obj):
    '''
    >>> print(log(AutoDiff(np.e)))
    1.0, 0.3679
    
    '''
    try:
        return ad.AutoDiff(np.log(AD_obj.val),(1/AD_obj.val) * AD_obj.der)
    except AttributeError:
        return np.log(AD_obj)
    

def exp(AD_obj):
    '''
    >>> print(exp(AutoDiff(0)))
    1.0, 1.0
    
    '''
    try:
        return ad.AutoDiff(np.exp(AD_obj.val),np.exp(AD_obj.val) * AD_obj.der)
    except AttributeError:
        return np.exp(AD_obj)
    
