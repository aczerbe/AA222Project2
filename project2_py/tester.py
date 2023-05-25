from tqdm import tqdm
import numpy as np
import project2
import helpers
import matplotlib.pyplot as plt

def main():
	fig, (ax1,ax2) = plt.subplots(1, 2)

	p = helpers.Ellipse4D()
	x_hist_1b, f_hist_1b, c_hist_1b = run_optimizer(project2.grad_descend_constraint, p, np.array([500, 500,500,500]))
	x_hist_1b = np.array(x_hist_1b)

	p = helpers.Ellipse4D()
	x_hist_1a, f_hist_1a, c_hist_1a = run_optimizer(project2.mom_grad_descend_constraint, p, np.array([500, 500,500,500]))
	x_hist_1a = np.array(x_hist_1a)

	

	
	#contourSize = [300,300]
	#X = np.linspace(-3,3,contourSize[0])
	#Y = np.linspace(-3,3,contourSize[1])
	#Z = np.zeros(contourSize)
	# for i in range(len(X)):
	# 	for j in range(len(Y)):
	# 		Z[j][i] = p.f(np.array([X[i],Y[j]]))
	
	#x,y = np.meshgrid(X,Y)
	#ax1.imshow( ((x + np.square(y) - 1 <= 0) & (-x -y <= 0)).astype(int) , 
                #extent=(x.min(),x.max(),y.min(),y.max()),origin="lower", cmap="Greys", alpha = 0.3);

	#ax1.contour(X,Y,Z, 100)
	#plt.xlim(-3,50)
	#ax1.set_xlim(-50,50)
	#ax1.set_ylim(-50,50)
	ax1.plot(x_hist_1a[:,0],x_hist_1a[:,1])
	ax1.plot(x_hist_1b[:,0],x_hist_1b[:,1])


	contourSize = [300,300]
	X = np.linspace(-500,500,contourSize[0])
	Y = np.linspace(-500,500,contourSize[1])
	Z = np.zeros(contourSize)
	for i in range(len(X)):
		for j in range(len(Y)):
			Z[j][i] = p.f(np.array([X[i],Y[j]]))
	
	x,y = np.meshgrid(X,Y)
	ax1.imshow( (p.c(np.array([x,y,0,0]))[0] < 0).astype(int) , 
                extent=(x.min(),x.max(),y.min(),y.max()),origin="lower", cmap="Greys", alpha = 0.3);

	ax2.plot(c_hist_1a, label='momentum')
	#print(c_hist_1a)
	ax2.plot(c_hist_1b, label='no mom')
	#print(c_hist_1b)
	ax2.legend()

	plt.show()


def run_optimizer(optimizer, p, x0):
    x_best, x_hist = optimizer(p.f, p.g, p.c, x0, p.n, p.count, p.prob)
    
    f_hist = [p.f(j) for j in x_hist]
    c_hist = [max(p.c(j)) for j in x_hist]
    print(str(x_best) + ": " + str(p.f(x_best)))
    return x_hist, f_hist, c_hist

if __name__ == '__main__':
	main()