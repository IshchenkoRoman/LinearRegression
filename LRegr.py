import numpy as np
import pandas as pd
import matplotlib as mt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class LRegr:
	
	def __init__(self, path="None"):
		self.path = path
		if (path != None):
			#self.wd = pd.read_csv(path, index_col='Index')
			self.wd = np.loadtxt(path, delimiter=',')
			self.l = len(self.wd[:,0])
		else:
			print("Invalid data!")

	def plotGraphic(self):
		plt.figure(1)
		plt.scatter(self.wd[:,0], self.wd[:,1], s=30, c='r', marker='x', linewidths=1)
		plt.xlim(4, 24)
		plt.xlabel('Population')
		plt.ylabel('Profit')
		plt.show()

	def plotHist(self):
		self.wd.plot(y='Height', kind='hist', color='red', title='Height (inch.) distibution')
		plt.show()

	def funcMinQuadErr(self, thetta_0=0.0, thetta_1=0.0):
		list_ = [((y - (thetta_0 + thetta_1 * x)) ** 2) for (x, y) in zip(self.wd['Weight'], self.wd['Height'])]
		err = sum(list_) / (2 * self.l)
		return(err)

	def measurement_error(self, w0=0.0, w1=0.0):
		summa = 0.0
		for _, row in self.wd.iterrows():
			summa+=(row['Height'] - (w0 + w1 * row['Weight'])) ** 2
		return summa

	def costFunct(self, thetta_0=0.0, thetta_1=0.0):
		list_ = [((y - (thetta_0 + thetta_1 * x)) ** 2) for (x, y) in zip(self.wd[:,0], self.wd[:,1])]
		err = sum(list_) / (2 * self.l)
		return(err)

	def plotLinearDescent(self, thetta_0=0.3, thetta_1=-0.3,alpha=0.001,epochs=1500):
		
		th, J_cost = self.gradDescent(thetta_0, thetta_1,alpha,epochs)
		xx = np.arange(5,23)
		yy = th[0] + th[1] * xx

		print(th)
		plt.scatter(self.wd[:,0], self.wd[:,1], s=30, c='r', marker='x',linewidths=1)
		plt.plot(xx, yy, label='Linear regression (Gradient descent)')

		plt.xlim(4,24)
		plt.xlabel('Population of City in 10,000s')
		plt.ylabel('Profit in $10,000s')
		plt.legend(loc=4)

		plt.show()

	def computeQuadErr0(self, thetta_0, thetta_1):
		list_ = [(2 * (thetta_0 + thetta_1 * x - y)) for (x, y) in zip(self.wd[:,0], self.wd[:,1])]
		err = np.sum(list_) / self.l
		return(err)

	def computeQuadErr1(self, thetta_0, thetta_1):
		list_ = [(2 * x * (thetta_0 + thetta_1 * x - y)) for (x, y) in zip(self.wd[:,0], self.wd[:,1])]
		err = np.sum(list_) / self.l
		return(err)

	def gradDescent(self, thetta_0=0.0, thetta_1=0.0,alpha=0.01,epochs=1500):

		J_hist = np.zeros(epochs)
		t_0 = thetta_0
		t_1 = thetta_1

		for i in np.arange(0, epochs):

			t_01 = t_0 - (alpha * self.computeQuadErr0(t_0, t_1))
			t_11 = t_1 - (alpha * self.computeQuadErr1(t_0, t_1))

			t_0 = t_01
			t_1 = t_11
			J_hist[i] = self.costFunct(t_0, t_1)

		return ([t_0, t_1], J_hist)

	def gradStep(self, b_curr, m_curr, alpha):

		b_gr = 0
		m_gr = 0

		for i in range(0, self.l):
			b_gr += - (2 / self.l) * (self.wd[:,1][i] - ((m_curr * self.wd[:,0][i]) + b_curr))
			m_gr += - (2 / self.l) * self.wd[:,0][i] * (self.wd[:,1][i] - ((m_curr * self.wd[:,0][i]) + b_curr))

		t_0 = b_curr - (alpha * b_gr)
		t_1 = m_curr - (alpha * m_gr)
		return (t_0, t_1)

	def gradDescent(self, thetta_0=0.0, thetta_1=0.0,alpha=0.01,epochs=1500):

		t_0 = thetta_0
		t_1 = thetta_1
		J_hist = np.zeros(epochs)
		
		for i in range(0, epochs):
			t_0, t_1 = self.gradStep(t_0, t_1, alpha)
			J_hist[i] = self.costFunct(t_0, t_1)

		return ([t_0, t_1], J_hist)

	def plotJ_hist(self, thetta_0=.3, thetta_1=-.3):
		
		th, J_cost = self.gradDescent(thetta_0, thetta_1)
		print(th)
		plt.plot(J_cost)
		plt.ylabel('Cost J')
		plt.xlabel('Iterations')
		plt.show()

	def y1(self, x):
		return(60 + 0.05 * x)

	def y2(self, x):
		return(50 + 0.16 * x)

	def plotCor(self):
		x = np.linspace(60, 170)
		self.wd.plot(x='Weight', y='Height', kind='scatter', title='Correlation between weight and height')
		plt.plot(x, self.y1(x), color='green')
		plt.plot(x, self.y2(x), color='red')
		plt.legend( ('line (60, 0.05)', 'line (50, 0.16)') )
		plt.show()

	def plot3DError(self):

		fig = plt.figure()
		ax = fig.gca(projection='3d')

		thetta_0 = np.arange(-5, 5, 0.25)
		thetta_1 = np.arange(-5, 5, 0.25)
		
		thetta_0, thetta_1 = np.meshgrid(thetta_0, thetta_1)
		err = self.funcMinQuadErr(thetta_0, thetta_1)
		surf = ax.plot_surface(thetta_0, thetta_1, err)
		ax.set_xlabel('thetta0')
		ax.set_ylabel('thetta1')
		ax.set_zlabel('Error')
		plt.show()

def main():
	st = "/home/rishchen/Source/ML/CourseTF/Tensorflow-Bootcamp-master/SourceCode/data.txt"
	LR = LRegr(st)
	#print(LR.wd[:,1])
	#LR.plotJ_hist()
	LR.plotLinearDescent()
	LR.plotGraphic()
	print(LR.costFunct())
	print(LR.gradDescent())


if __name__ == '__main__':
	main()