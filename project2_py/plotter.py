from tqdm import tqdm
import numpy as np
import project2
import helpers
import matplotlib.pyplot as plt

def main():
	fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

	p = helpers.Simple1()
	x_hist_1a, f_hist_1a, c_hist_1a = run_optimizer(project2.CMA_ES, p, np.array([2, 2]))
	x_hist_1a = np.array(x_hist_1a)

	p = helpers.Simple1()
	x_hist_1b, f_hist_1b, c_hist_1b = run_optimizer(project2.CMA_ES, p, np.array([0, 0]))
	x_hist_1b = np.array(x_hist_1b)
	
	p = helpers.Simple1()
	x_hist_1c, f_hist_1c, c_hist_1c = run_optimizer(project2.CMA_ES, p, np.array([-2, 2]))
	x_hist_1c = np.array(x_hist_1c)
	
	contourSize = [300,300]
	X = np.linspace(-3,3,contourSize[0])
	Y = np.linspace(-3,3,contourSize[1])
	Z = np.zeros(contourSize)
	for i in range(len(X)):
		for j in range(len(Y)):
			Z[j][i] = p.f(np.array([X[i],Y[j]]))
	
	
	ax1.set_title('Simple1, CMA_ES')
	x,y = np.meshgrid(X,Y)
	ax1.imshow( ((x + np.square(y) - 1 <= 0) & (-x -y <= 0)).astype(int) , 
                extent=(x.min(),x.max(),y.min(),y.max()),origin="lower", cmap="Greys", alpha = 0.3);

	ax1.contour(X,Y,Z, 100)

	ax1.plot(x_hist_1a[:,0],x_hist_1a[:,1])
	ax1.plot(x_hist_1b[:,0],x_hist_1b[:,1])
	ax1.plot(x_hist_1c[:,0],x_hist_1c[:,1])



	p = helpers.Simple1()
	x_hist_2a, f_hist_2a, c_hist_2a = run_optimizer(project2.cross_entropy_method, p, np.array([2, 2]))
	x_hist_2a = np.array(x_hist_2a)

	p = helpers.Simple1()
	x_hist_2b, f_hist_2b, c_hist_2b = run_optimizer(project2.cross_entropy_method, p, np.array([0, 0]))
	x_hist_2b = np.array(x_hist_2b)
	
	p = helpers.Simple1()
	x_hist_2c, f_hist_2c, c_hist_2c = run_optimizer(project2.cross_entropy_method, p, np.array([-2, 2]))
	x_hist_2c = np.array(x_hist_2c)
	
	contourSize = [300,300]
	X = np.linspace(-3,3,contourSize[0])
	Y = np.linspace(-3,3,contourSize[1])
	Z = np.zeros(contourSize)
	for i in range(len(X)):
		for j in range(len(Y)):
			Z[j][i] = p.f(np.array([X[i],Y[j]]))
	
	
	ax2.set_title('Simple1, Cross Entropy')
	x,y = np.meshgrid(X,Y)
	ax2.imshow( ((x + np.square(y) - 1 <= 0) & (-x -y <= 0)).astype(int) , 
                extent=(x.min(),x.max(),y.min(),y.max()),origin="lower", cmap="Greys", alpha = 0.3);

	ax2.contour(X,Y,Z, 100)

	ax2.plot(x_hist_2a[:,0],x_hist_2a[:,1])
	ax2.plot(x_hist_2b[:,0],x_hist_2b[:,1])
	ax2.plot(x_hist_2c[:,0],x_hist_2c[:,1])



	p = helpers.Simple2()
	x_hist_3a, f_hist_3a, c_hist_3a = run_optimizer(project2.CMA_ES, p, np.array([2, 2]))
	x_hist_3a = np.array(x_hist_3a)

	p = helpers.Simple2()
	x_hist_3b, f_hist_3b, c_hist_3b = run_optimizer(project2.CMA_ES, p, np.array([0, 0]))
	x_hist_3b = np.array(x_hist_3b)
	
	p = helpers.Simple2()
	x_hist_3c, f_hist_3c, c_hist_3c = run_optimizer(project2.CMA_ES, p, np.array([-1, 0]))
	x_hist_3c = np.array(x_hist_3c)
	
	contourSize = [300,300]
	X = np.linspace(-3,3,contourSize[0])
	Y = np.linspace(-3,3,contourSize[1])
	Z = np.zeros(contourSize)
	for i in range(len(X)):
		for j in range(len(Y)):
			Z[j][i] = p.f(np.array([X[i],Y[j]]))
	
	
	ax3.set_title('Simple2, CMA_ES')
	x,y = np.meshgrid(X,Y)
	ax3.imshow( (((x-1)**3 -y + 1<= 0) & (x + y -2<= 0)).astype(int) , 
                extent=(x.min(),x.max(),y.min(),y.max()),origin="lower", cmap="Greys", alpha = 0.3);

	ax3.contour(X,Y,Z, 500)

	ax3.plot(x_hist_3a[:,0],x_hist_3a[:,1])
	ax3.plot(x_hist_3b[:,0],x_hist_3b[:,1])
	ax3.plot(x_hist_3c[:,0],x_hist_3c[:,1])



	p = helpers.Simple2()
	x_hist_4a, f_hist_4a, c_hist_4a = run_optimizer(project2.cross_entropy_method, p, np.array([2, 2]))
	print(p.f(np.array([0,1])))
	x_hist_4a = np.array(x_hist_4a)

	p = helpers.Simple2()
	x_hist_4b, f_hist_4b, c_hist_4b = run_optimizer(project2.cross_entropy_method, p, np.array([0, 0]))
	x_hist_4b = np.array(x_hist_4b)
	
	p = helpers.Simple2()
	x_hist_4c, f_hist_4c, c_hist_4c = run_optimizer(project2.cross_entropy_method, p, np.array([-1, 0]))
	x_hist_4c = np.array(x_hist_4c)
	
	contourSize = [300,300]
	X = np.linspace(-3,3,contourSize[0])
	Y = np.linspace(-3,3,contourSize[1])
	Z = np.zeros(contourSize)
	for i in range(len(X)):
		for j in range(len(Y)):
			Z[j][i] = p.f(np.array([X[i],Y[j]]))
	
	
	ax4.set_title('Simple2, Cross Entropy')
	x,y = np.meshgrid(X,Y)
	ax4.imshow( (((x-1)**3 -y + 1<= 0) & (x + y -2<= 0)).astype(int) , 
                extent=(x.min(),x.max(),y.min(),y.max()),origin="lower", cmap="Greys", alpha = 0.3);

	ax4.contour(X,Y,Z, 500)

	ax4.plot(x_hist_4a[:,0],x_hist_4a[:,1])
	ax4.plot(x_hist_4b[:,0],x_hist_4b[:,1])
	ax4.plot(x_hist_4c[:,0],x_hist_4c[:,1])
	plt.tight_layout()


	fig2, ((ax12, ax22), (ax32, ax42)) = plt.subplots(2, 2)

	plt.tight_layout()
	
	ax12.plot(f_hist_3a)
	ax12.plot(f_hist_3b)
	ax12.plot(f_hist_3c)
	ax12.set_title('CMA_ES convergence')

	ax32.plot(f_hist_4a)
	ax32.plot(f_hist_4b)
	ax32.plot(f_hist_4c)
	ax32.set_title('Cross-entropy convergence')

	ax22.plot(c_hist_3a)
	ax22.plot(c_hist_3b)
	ax22.plot(c_hist_3c)
	ax22.set_title('CMA_ES constraints')

	ax42.plot(c_hist_4a)
	ax42.plot(c_hist_4b)
	ax42.plot(c_hist_4c)
	ax42.set_title('Cross-entropy constraints')
	
	
	plt.show()


def run_optimizer(optimizer, p, x0):
    x_best, x_hist = optimizer(p.f, p.g, p.c, x0, p.n, p.count, p.prob)
    
    f_hist = [p.f(j) for j in x_hist]
    c_hist = [max(p.c(j)) for j in x_hist]
    print(str(x_best) + ": " + str(p.f(x_best)))
    return x_hist, f_hist, c_hist

if __name__ == '__main__':
	main()