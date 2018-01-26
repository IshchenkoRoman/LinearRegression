# #* ************************************************************************** *#
# #*                                                                            *#
# #*                                                        :::      ::::::::   *#
# #*   tfLRegr.c                                          :+:      :+:    :+:   *#
# #*                                                    +:+ +:+         +:+     *#
# #*   By: rishchen <rishchen@student.42.fr>          +#+  +:+       +#+        *#
# #*                                                +#+#+#+#+#+   +#+           *#
# #*   Created: 2018/01/23 16:27:53 by rishchen          #+#    #+#             *#
# #*   Updated: 2018/01/23 16:28:16 by rishchen         ###   ########.fr       *#
# #*                                                                            *#
# #* ************************************************************************** *#

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from tensorflow.python import debug as tf_debug

from sklearn.linear_model import LinearRegression
from sklearn import linear_model

class		tfLRegr():

	def __init__(self, path=None, W=.3, b=-.3, alpha=0.01):

			# y = W*x + b
			self._wd = np.loadtxt(path, delimiter=',')
			self._l = len(self._wd[:,0])
			self.W = tf.Variable([W], dtype=tf.float32)
			self.b = tf.Variable([b], dtype=tf.float32)

			self.x = tf.placeholder(tf.float32)
			self.y = tf.placeholder(tf.float32)

			self.linear_model = self.W * self.x + self.b
			self.loss = tf.reduce_mean(tf.square(self.linear_model - self.y))

			self.optimizer = tf.train.GradientDescentOptimizer(alpha)
			self.train = self.optimizer.minimize(self.loss)

			self.init = None
			self.sess = None


	def tfLinearRegression(self, epochs=1000):

		#train loop
		x_train = self._wd[:,0]
		y_train = self._wd[:,1]

		# x_train = tf.convert_to_tensor(self._wd[:,0])
		# y_train = tf.convert_to_tensor(self._wd[:,1])

		self.init = tf.global_variables_initializer()
		self.sess = tf.Session()
		self.sess.run(self.init)

		for i in range(epochs):
			self.sess.run(self.train, {self.x: x_train, self.y: y_train})

		curr_W, curr_b, curr_loss = self.sess.run([self.W, self.b, self.loss], {self.x: x_train, self.y: y_train})
		print("W: {0} b: {1}, loss: {2}".format(curr_W, curr_b, curr_loss))


	def	sklearnLinearRegression(self):

		len_2_3 = int((len(self._wd[:,0]) * 2) / 3)

		max_x = self._wd[:,0].max()
		min_x = self._wd[:,0].min()

		regr = linear_model.LinearRegression()
		train_data_X = self._wd[:,0][0:len_2_3].reshape(-1, 1)
		train_data_Y = self._wd[:,1][0:len_2_3]
		regr.fit(train_data_X, train_data_Y)
		m = regr.coef_[0]
		b = regr.intercept_
		print(' y = {0} * x + {1}'.format(m, b))

		plt.scatter(self._wd[:,0], self._wd[:,1], color='blue')  # you can use test_data_X and test_data_Y instead.
		plt.plot([min_x, max_x], [b, m*max_x + b], 'r')
		plt.title('Fitted linear regression', fontsize=16)
		plt.xlabel('x', fontsize=13)
		plt.ylabel('y', fontsize=13)
		plt.show()

def main():

	#Path to data
	path = "/home/rishchen/Source/ML/CourseTF/Tensorflow-Bootcamp-master/SourceCode/data.txt"
	LR = tfLRegr(path)
	LR.sklearnLinearRegression()
	LR.tfLinearRegression()



if __name__ == '__main__':
	main()