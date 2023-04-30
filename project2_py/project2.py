#
# File: project2.py
#

## top-level submission file

'''
Note: Do not import any other modules here.
        To import from another file xyz.py here, type
        import project2_py.xyz
        However, do not import any modules except numpy in those files.
        It's ok to import modules only in files that are
        not imported here (e.g. for your plotting code).
'''
import numpy as np

params_table = {'simple1': [100, 5], 'simple2': [100, 10], 'simple3': [100, 10], 'secret1': [100, 10], 'secret2': [100, 10]}

def optimize(f, g, c, x0, n, count, prob):
    """
    Args:
        f (function): Function to be optimized
        g (function): Gradient function for `f`
        c (function): Function evaluating constraints
        x0 (np.array): Initial position to start from
        n (int): Number of evaluations allowed. Remember `f` and `c` cost 1 and `g` costs 2
        count (function): takes no arguments are reutrns current count
        prob (str): Name of the problem. So you can use a different strategy 
                 for each problem. `prob` can be `simple1`,`simple2`,`simple3`,
                 `secret1` or `secret2`
    Returns:
        x_best (np.array): best selection of variables found
    """
    x_best = x0
    #if prob == 'simple1':
    x_hist = cross_entropy_method(f, g, c, x0, n, count, prob)
    x_best = x_hist[-1]
    return x_best

def cross_entropy_method(f,g,c, x0, n, count, prob):
    x_hist = [x0]
    m = params_table[prob][0]
    m_elite = params_table[prob][1]
    rng = np.random.default_rng()
    cov = np.eye(len(x0))
    mean = x0
    #print('original: ' + str(x0))
    #print("constraint on original: " + str(c(x0)))

    while count() < n - (m*2):
        samples = rng.multivariate_normal(mean, cov, m)
        
        samples_pruned = []
        for j in samples:
            if in_bounds(j, c):
                samples_pruned.append(j)
                #print(j)

        #print()
        evals = [f(j) for j in samples_pruned]
        elite_samples = [samples_pruned[i] for i in np.argsort(evals)[0:m_elite]]
        #print(elite_samples)
        if(len(elite_samples) < 2):
            cov = cov * 1.5
        else:
            mean = np.mean(elite_samples, axis=0)
            cov = np.cov(elite_samples, rowvar=0)
            x_hist.append(mean)
        #print(mean)
    return x_hist



def gradient_descent(f,g,x0,n,count, prob):
    alpha = alpha_table[prob]
    x_hist = [x0]
    while count() < n:
        x_hist.append(x_hist[-1] - alpha * g(x_hist[-1]))
    return x_hist

def constrained_f_inv(f, c, rho, x):
    return f(x) + -1/(rho * c(x))

def in_bounds(x, c):
    a = c(x)
    #print(str(x) + ": " + str(a))
    return not any(t > 0 for t in a)