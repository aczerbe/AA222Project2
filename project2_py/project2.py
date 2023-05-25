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
import numpy.matlib

params_table = {'simple1': [100, 20, 1], 'simple2': [100, 20, 5], 'simple3': [100, 80, 1], 'secret1': [100, 80, 1], 'secret2': [200, 200, 10]}

CMA_params = {'simple1': 1, 'simple2': 1, 'simple3': 1, 'secret2': 10, 'secret1': 3000, 'ellipse4D': 1} #2000 for secret 1: 780

alpha_table = {'ellipse4D': 2500, 'simple3': .35, 'secret1': 150} # 100 works, score 901
beta_table = {'ellipse4D': .3, 'simple3': .3, 'secret1': .3}


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
    if prob == 'secret1' or prob == 'simple3':
        x_best, x_hist = grad_descend_constraint(f, g, c, x0, n, count, prob)
    else:
        x_best, x_hist = CMA_ES(f, g, c, x0, n, count, prob)
    return x_best

def mom_grad_descend_constraint(f,g, c, x0, n, count, prob, h=0.001):
    alpha = 2000 #alpha_table[prob]
    beta = 0.6 #beta_table[prob]
    x_hist = [x0]
    v = np.zeros_like(x0)
    while count() < n:
        x = x_hist[-1]
        base = c(x)
        if base < 0:
            #print('Pushing to CMA_ES at count: ' + str(count()) + " and location: " + str(x))
            print('Mom method count when done: ' + str(count()))
            #return x, x_hist
            return CMA_ES(f,g,c,x,n,count,prob)
        gradient = np.zeros_like(x)
        for i in range(len(x)):
            point = x 
            point[i] = point[i] + h
            test_val = c(point)
            if test_val < 0:
                return point, x_hist
            gradient[i] = (test_val - base)/h
        #print(gradient)
        v = beta * v - alpha * gradient
        x_hist.append(x_hist[-1] + v)
    return x_hist[-1], x_hist

# def grad_descend_constraint(f,g, c, x0, n, count, prob, h=0.001):
#     alpha = alpha_table[prob]
#     beta = beta_table[prob]
#     x_hist = [x0]
#     v = np.zeros_like(x0)
#     while count() < n:
#         x = x_hist[-1]
#         base = c(x)
#         #print(base)
#         if base < 0:
#             #print('Pushing to CMA_ES at count: ' + str(count()) + " and location: " + str(x))
#             print('Grad method count when done: ' + str(count()))
#             return x, x_hist
#             return CMA_ES(f,g,c,x,n,count,prob)
#         gradient = np.zeros_like(x)
#         for i in range(len(x)):
#             point = x 
#             point[i] = point[i] + h
#             test_val = c(point)
#             if test_val < 0:
#                 return point, x_hist
#             gradient[i] = (test_val - base)/h
#         #print(gradient)
#         #print()
#         v = beta * v - alpha * gradient
#         x_hist.append(x_hist[-1] + v)
#     return x_hist[-1], x_hist


def grad_descend_constraint(f,g, c, x0, n, count, prob, h=0.001):
    alpha = alpha_table[prob]
    #print(h)
    x_hist = [x0]
    while count() < n:
        x = x_hist[-1]
        base = c(x)
        #print(base)
        if base < 0:
            #print('Pushing to CMA_ES at count: ' + str(count()) + " and location: " + str(x))
            #print('Grad count when done: ' + str(count()))
            #return x, x_hist
            return CMA_ES(f,g,c,x,n,count,prob)
        gradient = np.zeros_like(x)
        for i in range(len(x)):
            point = x
            point[i] = point[i] + h
            #print(point)
            test_val = c(point)
            if test_val < 0:
                return point, x_hist
            gradient[i] = (test_val - base)/h
        #print(gradient)
        x_hist.append(x_hist[-1] - alpha * gradient)
    return x_hist[-1], x_hist


def CMA_ES(f,g,c,x0,n,count, prob):
    #print()
    N = len(x0)
    xmean = x0

    x_hist = [x0]

    sigma = CMA_params[prob]
    lambda_ = int(4 + np.floor(3*np.log(N)))
    mu = lambda_/2
    weights = np.log(mu + 1/2) - np.log(range(1, int(mu) + 1))
    mu = int(np.floor(mu))
    weights = weights / np.sum(weights)
    mueff = np.square(np.sum(weights)) / np.sum(np.square(weights))


    cc = (4+mueff/N) / (N+4 + 2*mueff/N)
    cs = (mueff+2) / (N+mueff+5)
    c1 = 2 / (np.square(N+1.3)+mueff)
    cmu = min(1-c1, 2 * (mueff-2+1/mueff) / (np.square(N+2)+mueff))
    damps = 1 + 2*max(0, np.sqrt((mueff-1)/(N+1))-1) + cs

    pc = np.zeros(N)
    ps = np.zeros(N)
    B = np.eye(N,N)
    D = np.ones(N)
    C = B * np.diag(np.square(D)) * B.T
    invsqrtC = B * np.diag(1/D) * B.T
    chiN=np.sqrt(N)*(1-1/(4*N)+1/(21*np.square(N)));
    rng = np.random.default_rng()

    eval_counter = 0

    best_value = 1000000
    x_best = x0


    while count() < n - (lambda_*2):
        samples = rng.multivariate_normal(xmean, sigma * C, lambda_)
        
        samples_pruned = []
        evals = []
        evals_pruned = []
        for x in samples:
            inside, value = constrained_f_infpenalty(f, c, x)
            evals.append(value)
            if inside:
                samples_pruned.append(x)
                evals_pruned.append(value) 
        if len(samples_pruned) >= mu:
            samples = samples_pruned
            evals = evals_pruned
        if(len(samples_pruned) > 0):
            if np.sort(evals_pruned)[0] < best_value:
                x_best = samples_pruned[np.argsort(evals_pruned)[0]]
                best_value = np.sort(evals_pruned)[0]

        eval_counter += lambda_

        samples = np.array(samples)
        sort = np.argsort(evals)
        evals = np.array([evals[j] for j in sort[0:mu]])
        xold = xmean
        xmean = samples[sort[0:mu]].T @ weights

        x_hist.append(samples[sort[0]])

        ps = (1-cs)*ps + np.sqrt(cs*(2-cs)*mueff) * invsqrtC @ (xmean-xold) / sigma
        hsig = np.linalg.norm(ps)/ np.sqrt(1-np.power((1-cs),(2*eval_counter/lambda_))) /chiN < (1.4 + 2/(N+1))
        pc = (1-cc)*pc + hsig *  np.sqrt(cc*(2-cc)*mueff) * (xmean-xold) / sigma;

        artmp = (1/sigma) * (samples[sort[0:mu]] - np.matlib.repmat(xold,mu,1));

        C = (1-c1-cmu) * C + c1 * (np.outer(pc,pc) + (1-hsig) * cc*(2-cc) * C) + cmu * artmp.T @ np.diag(weights) @ artmp
        sigma = sigma * np.exp((cs/damps)*(np.linalg.norm(ps)/chiN - 1))

        C = np.triu(C) + np.triu(C,1).T
        D,B = np.linalg.eig(C)
        D = np.sqrt(D)
        invsqrtC = B @ np.diag(1/D) @ B.T

    #print(x_best)
    return x_best, x_hist


def cross_entropy_method(f,g,c, x0, n, count, prob):
    x_hist = [x0]
    m = params_table[prob][0]
    m_elite = params_table[prob][1]
    inflation = params_table[prob][2]
    rng = np.random.default_rng()
    cov = inflation * np.eye(len(x0))
    mean = x0
    #print('original: ' + str(x0))
    #print("constraint on original: " + str(c(x0)))
    x_best = x0
    best_value = 1000000
    while count() < n - (m*2):
        samples = rng.multivariate_normal(mean, cov, m)
        samples_pruned = []
        evals = []
        evals_full = []
        for x in samples:
            inside, value = constrained_f_infpenalty(f, c, x)
            evals_full.append(value)
            if inside:
                samples_pruned.append(x)
                evals.append(value)
                #print(j)

        #print()
        if(len(samples_pruned) < 2):
            sort = np.argsort(evals_full)[0:m_elite]
            elite_samples = [samples[i] for i in sort]
            mean = np.average(elite_samples, axis=0, weights=calculate_weights([evals_full[i] for i in sort]))
            #print('outside boundary')
        else:
            sort = np.argsort(evals)[0:m_elite]
            elite_samples = [samples_pruned[i] for i in sort]
            mean = np.average(elite_samples, axis=0, weights=calculate_weights([evals[i] for i in sort]))
            #print('inside boundary')
        #print(elite_samples)
        cov = np.cov(elite_samples, rowvar=0)
        x_hist.append(samples[np.argsort(evals_full)[0]])
        if(len(samples_pruned) > 0):
            if np.sort(evals)[0] < best_value:
                x_best = samples_pruned[np.argsort(evals)[0]]
                best_value = np.sort(evals)[0]
    #print(x_best)
    return x_best, x_hist

def cross_entropy_step(f, c, mean, cov, m, m_elite, inflation, rng):
    
    return mean,cov

def calculate_weights(evals):
    minumum = min(evals)
    return [1/(.001 + k - minumum) for k in evals]


def constrained_f_infpenalty(f, c, x):
    k = c(x)
    if not any(t > 0 for t in k):
        return True, f(x)
    else:
        positives = k  * np.array([t > 0 for t in k])
        return 0, 1000 + sum(positives + np.square(positives))
        #return 0, 100*sum(np.array([t > 0 for t in k]))+ sum(positives + np.square(positives))

def in_bounds(x, c):
    a = c(x)
    #print(str(x) + ": " + str(a))
    return not any(t > 0 for t in a)