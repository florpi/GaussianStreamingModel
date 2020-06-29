import numpy as np
from scipy import integrate
import scipy.special as spl 

def integrand_v_r(k, r, power):

    return k*power(k)*spl.spherical_jn(1, k*r)

def v_r(r, power, f, bias, k_min=1.e-4, k_max=20.):
    ''' 
    Equation 7 https://arxiv.org/pdf/1105.4165.pdf
    '''
    integral = []
    for r_ in r:
        integral.append(
            integrate.quad(lambda x: integrand_v_r(x, r_, power),
            k_min,
            k_max,
            )[0]
            )

    return -f*bias/np.pi**2 * np.array(integral)

def integrand_psi_r(k, r, power):

    return power(k)*(spl.spherical_jn(0, k*r) - 2.*spl.spherical_jn(1,k*r)/k/r)

def psi_r(r, power,f,k_min=1.e-4, k_max=20.):
    ''' 
    Equation 10 https://arxiv.org/pdf/1105.4165.pdf
    '''
    integral = []
    for r_ in r:
        integral.append(
            integrate.quad(lambda x: integrand_psi_r(x, r_, power),
            k_min,
            k_max,
            )[0]
            )   

    return f**2/2./np.pi**2 * np.array(integral)

def integrand_psi_t(k, r, power):
    return power(k)*spl.spherical_jn(1, k*r)/k/r

def psi_t(r, power, f, k_min=1.e-4, k_max=20.):
    ''' 
    Equation 9 https://arxiv.org/pdf/1105.4165.pdf
    '''
    integral = []
    for r_ in r:
        integral.append(
            integrate.quad(lambda x: integrand_psi_t(x, r_, power),
            k_min,
            k_max,
            )[0]
            )

    return f**2/2./np.pi**2 * np.array(integral)



def sigma_v_sq(power,f, k_min=1.e-5, k_max=10.):

    def integrand_sigma_v(logq):
        q = np.exp(logq)
        return q * power(q)

    sigmasq = integrate.quad(integrand_sigma_v, np.log(k_min), np.log(k_max))[0] / (6 * np.pi ** 2)
    return sigmasq*f**2

def sigma_r_sq(r, power, f):

    return 2.*sigma_v_sq(power,f) - 2*psi_r(r, power, f)

def sigma_t_sq(r, power, f):

    return 2.*sigma_v_sq(power,f) - 2*psi_t(r, power, f)



if __name__ == '__main__':

    print(sigma_v(lambda k: 1/k, 1, 1))

