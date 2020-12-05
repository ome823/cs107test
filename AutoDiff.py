import numpy as np

class AutoDiff():
    '''Class for generating objects to perform autodifferentiation and basic operations
    
    Arguments:
    val -- scalar (for now) representing value of object
    der -- scalar (for now) representing derivative of object
    
    Methods:
    Overloads addition, multiplication, subtraction, division, power operations
    
    >>> a = 2.0
    >>> x = AutoDiff(a)
    >>> alpha = 2.0
    >>> beta = 3.0
    >>> print(alpha * x + beta)
    7.0, 2.0
    >>> print(x * alpha + beta)
    7.0, 2.0
    >>> print(beta + alpha * x)
    7.0, 2.0
    >>> print(beta + x * alpha)
    7.0, 2.0
    >>> print(2*x)
    4.0, 2.0
    >>> print(x + x)
    4.0, 2.0
    >>> print(x*x)
    4.0, 4.0
    >>> print(x**2)
    4.0, 4.0
    >>> print(2**x)
    4.0, 2.7726
    >>> print(x/4)
    0.5, 0.25
    >>> print(4/x)
    2.0, -1.0
    >>> print(2 - x)
    0.0, -1.0
    >>> print(x - 2)
    0.0, 1.0
    >>> print(3*x**5 +2*x**2-2*x**7/x**6)
    100.0, 246.0
    '''
    
    def __init__(self, val, der = 1.0):
        self.val = val
        self.der = der
        if hasattr(der, "__len__"):
            if not isinstance(der,np.ndarray):
                raise ValueError('If derivative is not a scalar it must be a Numpy array')
    
    def __add__(self, other):
        try:
            if _check_conformable_der(self.der, other.der):
                return AutoDiff(val = self.val + other.val, der = self.der + other.der)
            else:
                raise ValueError('Derivatives are not conformable')           
        except AttributeError:
            ad_other = AutoDiff(other, der = self._make_scalar_der(self.der))
            return self + ad_other
    
    def __radd__(self,other):
        return self.__add__(other)
    
    def __neg__(self):
        return AutoDiff(val = self.val * -1, der = self.der * -1)

    def __mul__(self, other):
        try:
            if _check_conformable_der(self.der, other.der):
                return AutoDiff(val = self.val * other.val, der = self.der * other.val + self.val * other.der)
            else:
                raise ValueError('Derivatives are not conformable')           
        except AttributeError:
            ad_other = AutoDiff(other, der = self._make_scalar_der(self.der))
            return self * ad_other
        
    def __rmul__(self,other):
        return self.__mul__(other)        
    
    def __sub__(self, other):
        try:
            if _check_conformable_der(self.der, other.der):
                return AutoDiff(val = self.val - other.val, der = self.der - other.der)
            else:
                raise ValueError('Derivatives are not conformable')         
        except AttributeError:
            ad_other = AutoDiff(other, der = self._make_scalar_der(self.der))
            return self - ad_other
    
    def __rsub__(self, other):
        try:
            if _check_conformable_der(self.der, other.der):
                return AutoDiff(val = other.val - self.val, der = other.der - self.der)
            else:
                raise ValueError('Derivatives are not conformable')           
        except AttributeError:
            ad_other = AutoDiff(other, der = self._make_scalar_der(self.der))
            return ad_other - self
    
    def __truediv__(self, other):
        try:
            if _check_conformable_der(self.der, other.der):
                return AutoDiff(val = self.val / other.val, der = self.der/other.val - self.val*other.der/other.val**2)
            else:
                raise ValueError('Derivatives are not conformable')           
        except AttributeError:
            ad_other = AutoDiff(other, der = self._make_scalar_der(self.der))
            return self / ad_other
    
    def __rtruediv__(self, other):
        try:
            if _check_conformable_der(self.der, other.der):
                return AutoDiff(val = other.val / self.val, der = other.der/self.val - other.val*self.der/self.val**2)
            else:
                raise ValueError('Derivatives are not conformable')           
        except AttributeError:
            ad_other = AutoDiff(other, der = self._make_scalar_der(self.der))
            return ad_other / self 

    def __pow__(self, other):
        try:
            if _check_conformable_der(self.der, other.der):
                if self.val == 0:
                    return AutoDiff(0, der = self.der*other.val*(self.val**(other.val-1)))
                else:
                    return AutoDiff(self.val**other.val, der = self.der*other.val*(self.val**(other.val-1))+other.der*np.log(np.abs(self.val))*self.val**other.val)
            else:
                raise ValueError('Derivatives are not conformable')           
        except AttributeError:
            ad_other = AutoDiff(other, der = self._make_scalar_der(self.der))
            return self ** ad_other

    
    def __rpow__(self, other):
        try:
            if _check_conformable_der(self.der, other.der):
                return AutoDiff(val = other.val**self.val, der = other.der*self.val*(other.val**(self.val-1))+other.val**self.val*np.log(other.val)*self.self)
            else:
                raise ValueError('Derivatives are not conformable')           
        except AttributeError:
            ad_other = AutoDiff(other, der = self._make_scalar_der(self.der))
            return ad_other**self
            
        
    def __eq__(self, other):
        try:
            if hasattr(self.der, "__len__") and hasattr(other.der, "__len__"):
                if len(self.der) == len(other.der) and len(self.der) == sum([1 for i, j in zip(self.der, other.der) if i == j]):
                    return (self.val == other.val) 
            elif hasattr(self.der, "__len__") or hasattr(other.der, "__len__"):
                return False
            else:
                return (self.val == other.val) and (self.der == other.der)
        except AttributeError:
            return False
    
    def __str__(self):
        if isinstance(self.der,np.ndarray):
            return (f'{round(self.val,4)}, {np.round(self.der,4)}')
        else:
            return (f"{round(self.val,4)}, {round(self.der,4)}")

    def _make_scalar_der(self, copy_der):
        if hasattr(copy_der, "__len__"):
            return np.zeros(len(copy_der))
        else:
            return 0

def makeVars(vals, seed = None):
    if hasattr(vals, "__len__") and not seed:
        seed = np.ones(len(vals)) #seed is all ones
    elif hasattr(vals, "__len__") and hasattr(seed, "__len__"):
        if not len(vals) == len(seed):
            raise ValueError('Values must be the same length as seeds')
    else:
        raise ValueError('makeVars requires at least one array')
    ad_vars = []
    for ind, (v, d) in enumerate(zip(vals, seed)):
        der_seed = np.zeros(len(seed))
        der_seed[ind] = d
        ad_vars.append(AutoDiff(v, der_seed))
    return ad_vars

class multivar():
    def __init__(self, func_list):
        self.func_list = func_list
    def makeJacobian(self):
        jacobian = []
        for func in self.func_list:
            jacobian.append(func.der)#stick the derivatives together
        return np.stack(jacobian)

def _check_conformable_der(der1,der2):
    if hasattr(der1, "__len__") and hasattr(der2, "__len__"):
        return len(der1) == len(der2)
    elif hasattr(der1, "__len__") or hasattr(der2, "__len__"):
        return False
    else:
        if isinstance(der1, (np.number,int, float)) and isinstance(der2, (np.number,int, float)):
            return True
        else:
            return False