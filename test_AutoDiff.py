import AutoDiff as ad
import AutoDiffMath as adm
import numpy as np


def test_init_error():
    assert ad.AutoDiff(2,der = [1,2,3])

def test_ad_addition_error():
    assert ad.AutoDiff(2.0,3.0)+ad.AutoDiff(-1.5,complex(4.0))



def test_multiplication_result():
    assert ad.AutoDiff(2.0)*ad.AutoDiff(2.0)==ad.AutoDiff(2.0)**2

def test_multivariate_multiplication_result():
    x,y = ad.makeVars([2,3])
    assert x*y==ad.AutoDiff(6.0, der = np.array([3.0,2.0])) #df/dx = y, df/dy = x

def test_multivariate_scalar_multiplication_result():
    x = ad.AutoDiff(2.0, der = np.array([1.0,0.0]))
    y = 3.0
    assert x*y==ad.AutoDiff(6.0, der = np.array([3.0,0.0])) #df/dx = y, df/dy = 0

def test_ad_addition():
    assert ad.AutoDiff(2.0,3.0)+ad.AutoDiff(-1.5,4.0)==ad.AutoDiff(0.5,7.0)

def test_scalar_addition():
    assert ad.AutoDiff(2.0,3.0)+5.0 == 5.0 + ad.AutoDiff(2.0,3.0)

def test_scalar_subtraction():
    assert ad.AutoDiff(2.0,3.0) - 5.0 == -1*(5.0 - ad.AutoDiff(2.0,3.0))

def test_division():
    assert ad.AutoDiff(2.0,4.0)/ad.AutoDiff(2.0,1.0)== ad.AutoDiff(1.0,1.5)

def test_sin_cos_1dim():
    a = ad.AutoDiff(np.pi)
    b = ad.AutoDiff(np.pi/2)
    difference = adm.sin(a)-adm.cos(b)
    val_difference = np.linalg.norm(difference.val)
    der_difference = np.linalg.norm(difference.der)
    assert (val_difference <= 1e-10) and (der_difference <= 1e-10)

def test_sin_cos_2dim():
    a,b = ad.makeVars([np.pi,np.pi/2])
    difference = adm.sin(a)-adm.cos(b)
    val_difference = np.linalg.norm(difference.val)
    der_difference = np.linalg.norm(difference.der)
    assert (val_difference <= 1e-10) and (der_difference-np.sqrt(2) <= 1e-10)

def test_sin_cos_scalar():
    a = adm.sin(ad.AutoDiff(np.pi)) + adm.sin(np.pi) - adm.cos(np.pi) + adm.cos(ad.AutoDiff(np.pi/2.0))
    b = ad.AutoDiff(1.0,-2.0)
    difference = a-b
    assert np.abs(difference.val) <= 1e-10 and np.linalg.norm(difference.der) <= 1e-10

def test_div_scalar():
    assert ad.AutoDiff(2.0, 2.0)/3.0 == ad.AutoDiff(2.0, 2.0)/ad.AutoDiff(3.0, 0.0)

def test_rdiv_scalar():
    assert 4.0/ad.AutoDiff(3.0, 2.0) == ad.AutoDiff(4.0, 0.0)/ad.AutoDiff(3.0, 2.0)

def test_pow():
    assert ad.AutoDiff(0.0, 2.0)**ad.AutoDiff(5.0, 2.0) == ad.AutoDiff(0.0, 0.0)

def test_pow_scalar():
    assert 6**ad.AutoDiff(3.0, 2.0) == ad.AutoDiff(6.0, 0.0)**ad.AutoDiff(3.0, 2.0)

def test_exp():
    a = adm.exp(ad.AutoDiff(1))*adm.exp(ad.AutoDiff(1))
    b = adm.exp(ad.AutoDiff(1)+ad.AutoDiff(1))
    difference = a-b
    assert np.abs(difference.val) <= 1e-10 and np.abs(difference.der) <= 1e-10

def test_exp_scalar():
    assert adm.exp(1) == np.exp(1)

def test_log():
    assert adm.log(ad.AutoDiff(1))-adm.log(ad.AutoDiff(2)) == adm.log(ad.AutoDiff(1)/ad.AutoDiff(2))

def test_log_scalar():
    assert adm.log(1)==np.log(1)

def test_inequality():
    a = ad.AutoDiff(3.0, 3.0)
    b = 7
    assert a != b

def test_negation():
    a = ad.AutoDiff(2.0, 3.0)
    assert -a == ad.AutoDiff(-2.0, -3.0)


def test_output():
    x = ad.AutoDiff(3.0)
    print(x)

def test_tan_sin_cos():
    a = adm.tan(ad.AutoDiff(np.pi))
    b = adm.cos(ad.AutoDiff(np.pi)) * adm.sin(ad.AutoDiff(np.pi))
    difference = a-b
    assert (difference.val <= 1e-15 and difference.der <= 1e-15)

def test_square_root():
    a = adm.sqrt(ad.AutoDiff(100))
    b = ad.AutoDiff(10,0.5*0.1)
    difference = a-b
    assert (difference.val <= 1e-15 and np.linalg.norm(difference.der) <= 1e-15)
    
    
def test_nat_log():
    a = adm.log(ad.AutoDiff(np.e))
    b = adm.log(ad.AutoDiff(np.e * 2))
    difference = a* .5 - b
    assert (difference.val <= 1e-15 and np.linalg.norm(difference.der) <= 1e-15)

    
def test_tan_scalar():
    assert adm.tan(0) == np.tan(0)

def test_sqrt_scalar():
    assert adm.sqrt(4) == np.sqrt(4)
