
### Questions
* How do we decide which test to use?
* What is the workflow like?
* What about non normal populations?]


## Hypothesis -> Frequentist Approach

## Bayesian Approach

### Objectives
YWBAT
* apply hypothesis testing to groups
* meeting test critera

### Scenarios
* Medical Research
* In sports, does height have an effect on release point from the plate
* Put a price on carbon, does this effect emissions
* Serving a landing page
* Ad campaigns - which ads drive clicks
* Insurance - does this population present more/less of a risk

### Outline


```python
import pandas as pd
import numpy as np

import scipy.stats as scs

import matplotlib.pyplot as plt
```


```python
mu0 = 54.0
```


```python
population1 = np.random.randint(10, 100, 2000)
population2 = np.random.randint(20, 80, 2000)
```


```python
# how can we compare the means of these populations?
population1.mean(), population2.mean()
```




    (54.8145, 48.396)




```python
# sampling distributions
means1 = []
means2 = []

for i in range(30):
    means1.append(np.random.choice(population1, size=50, replace=False).mean())
    means2.append(np.random.choice(population2, size=50, replace=False).mean())
    
    
# based on the clt - the means of the sampling distributions is normally distributed
```


```python
# step 1: pick your test
# step 2: do we meet the criteria of the test?
```


```python
# test for equal variances LEVENE TEST
# h0: var1 = var2
# ha: var2 != var2

scs.levene(means1, means2)

# p = 0.06 -> fail to reject null, variances are equal
```




    LeveneResult(statistic=3.587887243695722, pvalue=0.06319220302449272)




```python
# Which test do we use?
# pick our test: ttest_ind
# what are the assumptions:
# a, b have to be normal
# need to check for equal_variances


# h0: mu1 = mu2
# ha: mu1 != mu2

scs.ttest_ind(means1, means2, equal_var=True)


# pvalue = 0 -> reject the null, so the means are different
```




    Ttest_indResult(statistic=7.51778742021142, pvalue=3.951152287595308e-10)




```python
np.mean(means1), np.mean(means2)
```




    (53.97533333333333, 47.931333333333335)




```python
# Shapiro test
# h0: x is normal
# ha: x is not normal
scs.shapiro(means1), scs.shapiro(means2)

# massive pvalues -> fail to reject null -> normal
```




    ((0.9760878086090088, 0.7147558331489563),
     (0.984048068523407, 0.9198938608169556))




```python
# h0: mu1 = mu2
# ha: mu1 != mu2
scs.ttest_rel(np.random.choice(population1, size=30), np.random.choice(population1, size=30))
```




    Ttest_relResult(statistic=-0.20328351760585, pvalue=0.8403331042360073)




```python
# set up your null/alternative hypothesis
# get normal data through sampling distribution(s)
# pick test to run
# meet assumptions/requirements
# run test
# make conclusion
# dig deeper
```

## ttest_1samp
* **When**
    * See if a population statistics is the same as a statistic (number)
        * comparing an arr to a number
    
* **Assumptions**
    * pop mean
    * normality -> shapiro test

## ttest_ind
* **When**
    * Comparing 2 populations (arrays)
    
* **Assumptions**
    * normality -> shapiro test
    * equal variance -> levene test


# Testing for multiple groups (>2)

### Assessment
