from tqdm import tqdm
import numpy as np
import project2
import helpers
import matplotlib.pyplot as plt

def main():
	p = helpers.Simple1()
	(x_hist_1, f_hist_1) = run_optimizer(project2.cross_entropy_method, p, np.array([2, 2]))
	

def run_optimizer(optimizer, p, x0):
    x_hist = optimizer(p.f, p.g, p.c, x0, p.n, p.count, p.prob)
    f_hist = [p.f(j) for j in x_hist]
    return (x_hist, f_hist)

if __name__ == '__main__':
	main()