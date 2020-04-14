
### 1. Model 
#### Linear model with random effects:
$y_{it} = \beta x_{it} + u_i + \epsilon_{it}$,
where
- $y_{it}$ univariate, repeated measurements of individual i at time t, $i=1,\cdots,n$, $t = 1,\cdots,T$
- $x_{it}$ univariate, covariate
- $\beta$ univariate coefficient
- $u_i$ random effect, for addressing time dependency, $N(0, \psi_{u})$
- $\epsilon_{it}$ residual, $N(0, \psi_{\epsilon})$

#### Bayesian inference:
##### Prior 
- $\psi_{\epsilon}$: $IG(a_{\epsilon 0}, b_{\epsilon 0})$
- $\beta ~|~ \psi_{\epsilon}$: $N(\beta_0, \sigma_0 \psi_{\epsilon})$
- $\psi_{u}$: $IG(a_{u 0}, b_{u 0})$

where IG(a, b) has pdf $f(x) = \frac{b^a}{\Gamma(a)} x^{-(a+1)} e^{-b/x}$

##### Posterior
1. p($\psi_{\epsilon}$| all others)

$p(\psi_{\epsilon}|\cdot)$

$\propto \prod_{i=1}^n \prod_{t=1}^T p(y_{it}|\beta, u_i, \psi_{\epsilon}) \cdot  p(\psi_{\epsilon})$

$\propto \prod_{i=1}^n \prod_{t=1}^T \frac{1}{\sqrt{2 \pi \psi_{\epsilon}}} \exp \{-\frac{1}{2 \psi_{\epsilon}}(y_{it}-\beta x_{it}-u_i)^2\} \cdot \frac{b_{\epsilon 0}^{a_{\epsilon 0}}}{\Gamma(a_{\epsilon 0})} \psi_{\epsilon}^{-(a_{\epsilon 0}+1)} e^{-\frac{b_{\epsilon 0}}{\psi_{\epsilon}}}$

$\propto \psi_{\epsilon}^{-(\frac{nT}{2}+a_{\epsilon 0}+1)} \exp \{-\frac{1}{\psi_{\epsilon}}(\frac{1}{2} \sum_{i=1}^n \sum_{t=1}^T (y_{it}-\beta x_{it}-u_i)^2 + b_{\epsilon 0})\}$

$\rightarrow IG(a_{\epsilon}^*, b_{\epsilon}^*)$, 

with $a_{\epsilon}^* = a_{\epsilon 0}+\frac{nT}{2}$,  $b_{\epsilon}^* = b_{\epsilon 0}+\frac{1}{2} \sum_{i=1}^n \sum_{t=1}^T (y_{it}-\beta x_{it}-u_i)^2$ 

2. p($\beta$ | $\psi_{\epsilon}$ and all others)

$p(\beta|\cdot)$

$\propto \prod_{i=1}^n \prod_{t=1}^T p(y_{it}|\beta, u_i, \psi_{\epsilon}) \cdot  p(\beta|\psi_{\epsilon})$

$\propto \exp\{-\frac{1}{2 \psi_{\epsilon}}(y_{it}-\beta x_{it}-u_i)^2\} \cdot \exp\{-\frac{1}{2 \sigma_0^2 \psi_{\epsilon}}(\beta-\beta_0)^2\}$

$\propto \exp \{ -\frac{1}{2} \frac{\sigma_0^2 \sum_{i=1}^n \sum_{t=1}^T x_{it}+1}{\psi_{\epsilon} \sigma_0^2} (\beta^2 - 2 \frac{\sigma_0^2\sum_{i=1}^n \sum_{t=1}^T x_{it}(y_{it}-u_i)+\beta_0}{\sigma_0^2 \sum_{i=1}^n \sum_{t=1}^T x_{it}^2+1} \beta) \}$

$\rightarrow N(\beta^*, \sigma^{2*}_{\beta})$

with $\beta^* = \frac{\sigma_0^2\sum_{i=1}^n \sum_{t=1}^T x_{it}(y_{it}-u_i)+\beta_0}{\sigma_0^2 \sum_{i=1}^n \sum_{t=1}^T x_{it}^2+1}$,  $\sigma^{2*}_{\beta} = \frac{\psi_{\epsilon} \sigma_0^2}{\sigma_0^2 \sum_{i=1}^n \sum_{t=1}^T x_{it}+1}$

3. p($u_i$|all others)

$p(u_i|\cdot)$

$\propto \prod_{t=1}^T p(y_{it}|\beta, u_i, \psi_{\epsilon}) \cdot p(u_i|\psi_u)$

$\propto \exp\{-\frac{1}{2 \psi_{\epsilon}}(y_{it}-\beta x_{it}-u_i)^2\} \cdot \exp\{-\frac{u_i^2}{2 \psi_u} \}$

$\propto \exp \{ \frac{1}{2} (\frac{T}{\psi_{\epsilon}}+\frac{1}{\psi_u}) [ u_i^2 + 2 \frac{ \frac{1}{\psi_{\epsilon} }\sum_{t=1}^T (y_{it}-\beta x_{it})}{\frac{T}{\psi_{\epsilon}}+\frac{1}{\psi_u}} u_i] \}$

$\rightarrow N(\mu_i^*, \sigma_u^{2*})$

with $\sigma_u^{2*} = (\frac{T}{\psi_{\epsilon}}+\frac{1}{\psi_u})^{-1}$, $\mu_i^* = \sigma_u^{2*} \cdot \frac{1}{\psi_{\epsilon} }\sum_{t=1}^T (y_{it}-\beta x_{it})$

4. p($\psi_u$|all others)

$p(\psi_u|\cdot)$

$\propto \prod_{i=1}^n p(u_i|\psi_u) \cdot p(\psi_u)$

$\propto \prod_{i=1}^n \frac{1}{\sqrt{2 \pi \psi_u}} \exp\{-\frac{u_i^2}{2 \psi_u} \} \cdot \frac{b_{u0}^{a_{u0}}}{\Gamma(a_{u0})} \psi_u^{-(a_{u0}+1)} e^{-b_{u0}/\psi_u}$

$\propto \psi_u^{-(\frac{n}{2}+a_{u0}+1)} \exp\{ -(b_{u0}+\frac{1}{2}\sum_{i=1}^nu_i^2) /\psi_u \}$

$\rightarrow IG(a_{u0}^*, b_{u0}^*)$

with $a_{u0^*} = a_{u0}+\frac{n}{2}$, $b_{u0}^* = b_{u0}+\frac{1}{2}\sum_{i=1}^nu_i^2$



```R
cat("Linear model with random effects - simulation")
```

    Linear model with random effects - simulation

### 2. True value
Given true values etc.
- $n$ sample size
- $T$ number of time points
- $\beta$, $\psi_u$, $\psi_{\epsilon}$ parameters


```R
cat("Define constants.")
n = 1000
T = 10
beta_true = 0.8
psiU_true = 0.3
psiE_true = 0.3
```

    Define constants.

### 3. Generate data
Generate $x_{it}$, $u_i$, $y_{it}$ in turn.
$x = (x_{1,1}, x_{2,1}, \cdots, x_{n,1}, \cdots, x_{1,T}, x_{2,T}, \cdots, x_{n,T})$ is a vector. $y$ similar.


```R
cat("Generate data.")
x = rnorm(n*T)
u_true = rnorm(n, 0 , sqrt(psiU_true))
y = beta_true * x + rep(u_true, T) + rnorm(n*T, 0, sqrt(psiE_true))
```

    Generate data.

### 4. Prior 
- $\psi_{\epsilon}$: $IG(a_{\epsilon 0}, b_{\epsilon 0})$
- $\beta ~|~ \psi_{\epsilon}$: $N(\beta_0, \sigma_0 \psi_{\epsilon})$
- $\psi_{u}$: $IG(a_{u 0}, b_{u 0})$


```R
cat("Define prior hyperparameters.")
aE0 = 3
bE0 = 5
beta0 = 0
sigma0 = 100
aU0 = 3
bU0 = 5
```

    Define prior hyperparameters.

### 5. Variables for MCMC iterations
nIter - number of MCMC iterations in total
nBurn - number of burn-in interations. 
beta, u, psiE, psiU - current value.  
beta_sample, u_sample, psiE_sample, psiU_sample - store the posterior samples


```R
cat("Variables for MCMC.")
nIter = 500
nBurn = 200
beta = 0
u = rep(0, n)
psiE = 0
psiU = 0
beta_sample = rep(0, nIter-nBurn)
u_sample = array(0, c(n, nIter-nBurn))
psiE_sample = rep(0, nIter-nBurn)
psiU_sample = rep(0, nIter-nBurn)
```

    Variables for MCMC.

### 6. Set initial values


```R
cat("Set initial values.")
beta = 0
u = rnorm(n)
psiE = 1
psiU = 1
```

    Set initial values.

### 7. MCMC iterations
1. update $\psi_{\epsilon}$ and $\beta$
2. update $\psi_u$
2. update $u_i$


```R
cat("MCMC updates.")
for (g in 1:nIter)
{
    # print the loop index 
    if (g%%100 == 0) cat("iteration = ", g, "\n")
        
    # 1. update psiE and beta
    # 1.1 update psiE
    aES = aE0 + n*T/2
    bES = bE0 + 0.5 * sum((y - beta*x - rep(u, T))^2)  
    psiE = 1/rgamma(1, shape = aES, rate = bES) # IG(aES, bES)
    if (g > nBurn)
    {
      psiE_sample[g-nBurn] = psiE      
    }   
    # 1.2 update beta
    betaS = sigma0 * sum(x*(y-rep(u, T))) + beta0
    betaS = betaS/(sigma0*sum(x^2)+1)
    sigBS = psiE*sigma0/(sigma0*sum(x^2) + 1) 
    beta = rnorm(1, betaS, sqrt(sigBS)) # N(betaS, sigBS)
    if (g > nBurn)
    {
      beta_sample[g-nBurn] = beta
    } 
        
    # 2. update u
    sigUS = 1/(T/psiE + 1/psiU) 
    yBX = y - beta*x
    yBX = array(yBX, c(n, T))   
    yBX = rowSums(yBX)
    muS = sigUS/psiE * yBX
    u = rnorm(n, muS, sqrt(sigUS)) # N(muS, sigUS)
    if (g > nBurn)
    {
      u_sample[,g-nBurn] = u
    }
        
    # 3. update psiU  
    aUS = aU0 + n/2
    bUS = bU0 + 0.5 * sum(u^2)    
    psiU = 1/rgamma(1, shape = aUS, rate = bUS) # IG(sUS, bUS)
    if (g > nBurn)
    {
      psiU_sample[g-nBurn] = psiU      
    }
}
```

    MCMC updates.iteration =  100 
    iteration =  200 
    iteration =  300 
    iteration =  400 
    iteration =  500 


### 8. Estimations 


```R
cat("Estimations.")
psiE_est = mean(psiE_sample)
beta_est = mean(beta_sample)
psiU_est = mean(psiU_sample)
cat("psiE_est = ", psiE_est, ", bias = ", psiE_est - psiE_true, "\n")
cat("beta_est = ", beta_est, ", bias = ", beta_est - beta_true, "\n")
cat("psiU_est = ", psiU_est, ", bias = ", psiU_est - psiU_true, "\n")
```

    Estimations.psiE_est =  0.3018733 , bias =  0.001873319 
    beta_est =  0.8009616 , bias =  0.0009615616 
    psiU_est =  0.2975051 , bias =  -0.002494917 



```R

```
