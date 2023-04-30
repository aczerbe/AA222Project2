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

params_table = {'simple1': [100, 20, 1], 'simple2': [100, 20, 1], 'simple3': [100, 80, 1], 'secret1': [100, 80, 1], 'secret2': [200, 200, 10]}

CMA_params = {'simple1': 1, 'secret2': 10, 'secret1': 100}


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
    if prob == 'secret2' or prob == 'secret1':
        x_best, x_hist = CMA_ES(f, g, c, x0, n, count, prob)
    else:
        x_best, x_hist = cross_entropy_method(f, g, c, x0, n, count, prob)
    return x_best



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
        #print()
        #print(sigma * C)
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

        #samples = [[6.6024, 6.9158], [6.3449, 7.1803], [6.8784, 6.7896], [6.6291, 6.8360], [7.3035, 7.5837], [6.3367, 7.1660]]

        #evals =  [-45.2760,  -45.1735,  -46.3167,  -44.9315,  -55.0030,  -45.0236]

        samples = np.array(samples)
        sort = np.argsort(evals)
        evals = np.array([evals[j] for j in sort[0:mu]])
        xold = xmean
        xmean = samples[sort[0:mu]].T @ weights

        x_hist.append(samples[sort[0]])

        ps = (1-cs)*ps + np.sqrt(cs*(2-cs)*mueff) * invsqrtC @ (xmean-xold) / sigma
        hsig = np.linalg.norm(ps)/ np.sqrt(1-np.power((1-cs),(2*eval_counter/lambda_))) /chiN < (1.4 + 2/(N+1))
        pc = (1-cc)*pc + hsig *  np.sqrt(cc*(2-cc)*mueff) * (xmean-xold) / sigma;

        #print(ps)
        #print(hsig)
        #print(pc)
        #print()

        artmp = (1/sigma) * (samples[sort[0:mu]] - np.matlib.repmat(xold,mu,1));

        #print(artmp)
        #print(artmp @ artmp.T)
        #print(artmp.T @ artmp)
        #print(sort[0:mu])
        #print(samples)
        #print(np.sum([weights[i] * np.outer(((samples[sort[i]] - xold)/sigma), ((samples[sort[i]] - xold)/sigma)) for i in range(mu)],axis=0))
        #print()
        #print(cmu * artmp.T @ np.diag(weights) @ artmp)
        C = (1-c1-cmu) * C + c1 * (np.outer(pc,pc) + (1-hsig) * cc*(2-cc) * C) + cmu * artmp.T @ np.diag(weights) @ artmp
        sigma = sigma * np.exp((cs/damps)*(np.linalg.norm(ps)/chiN - 1))

        #print(sigma)
        #print(C)

        C = np.triu(C) + np.triu(C,1).T
        D,B = np.linalg.eig(C)
        #print(D)
        #B[0][1] = -B[0][1]
        #B[1][1] = -B[1][1]
        #print()
        #print(B)

        #for i in range(len(D)):
        #    if D[i] < 0:
        #        D[i] = -D[i]
        #        B[i] = -B[i]
        #test = 1/D
        D = np.sqrt(D)
        #print(D)
        invsqrtC = B @ np.diag(1/D) @ B.T

        #print(C)
        #print(invsqrtC)



        #print(samples[sort[0]])
        
        #print(samples[sort[0]])

    print(x_best)
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

def gradient_descent(f,g,x0,n,count, prob):
    alpha = alpha_table[prob]
    x_hist = [x0]
    while count() < n:
        x_hist.append(x_hist[-1] - alpha * g(x_hist[-1]))
    return x_hist

def constrained_f_infpenalty(f, c, x):
    k = c(x)
    if not any(t > 0 for t in k):
        return True, f(x)
    else:
        positives = k  * np.array([t > 0 for t in k])
        return 0, 100*sum(np.array([t > 0 for t in k]))+ sum(positives + np.square(positives))

def in_bounds(x, c):
    a = c(x)
    #print(str(x) + ": " + str(a))
    return not any(t > 0 for t in a)