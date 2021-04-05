import pandas as pd
import numpy as np
import scipy.stats as stats
import math

np.random.seed(6)

school_ages = stats.poisson.rvs(loc = 18,mu = 35 , size = 1500)
classA_ages = stats.poisson.rvs(loc = 18,mu = 30 , size = 60)

classA_ages.mean()

_,p_value = stats.ttest_1samp(a = classA_ages,popmean = school_ages.mean())

print(p_value)

school_ages.mean()

if p_value < 0.05:
    print('reject null hypothesis')
else:
    print('accept null hypothesis')

