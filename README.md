# Linear Regression

## Liner Regression with one variable
In this part of this exercise, you will implement linear regression with one
variable to predict profits for a food truck. Suppose you are the CEO of a
restaurant franchise and are considering different cities for opening a new
outlet. The chain already has trucks in various cities and you have data for
profits and populations from the cities.

You would like to use this data to help you select which city to expand
to next.
The file ex1data1.txt contains the dataset for our linear regression problem. The first column is the population of a city and the second column is
the profit of a food truck in that city. A negative value for profit indicates a
loss

## Plotting the Data

Before starting on any task, it is often useful to understand the data by
visualizing it. For this dataset, you can use a scatter plot to visualize the
data, since it has only two properties to plot (profit and population). (Many
other problems that you will encounter in real life are multi-dimensional and
can’t be plotted on a 2-d plot.)

![image](https://user-images.githubusercontent.com/30857998/36558395-e69d2f34-1802-11e8-86c9-fc895121e351.png)

## Gradient Descent

In this part, you will fit the linear regression parameters θ to our dataset
using gradient descent.
The objective of linear regression is to minimize the cost function
![image](https://user-images.githubusercontent.com/30857998/36558551-43148bfe-1803-11e8-955d-14f4eac5fd51.png)

Recall that the parameters of your model are the θj values. These are
the values you will adjust to minimize cost J(θ). One way to do this is to
use the batch gradient descent algorithm. In batch gradient descent, each
iteration performs the update

![image](https://user-images.githubusercontent.com/30857998/36558620-81c1f49a-1803-11e8-9887-4b74376643ff.png)

## Gradient descent
The loop structure has been written for you, and you only need to supply
the updates to θ within each iteration.
As you program, make sure you understand what you are trying to optimize and what is being updated. Keep in mind that the cost J(θ) is parameterized by the vector θ, not X and y. That is, we minimize the value of J(θ)
by changing the values of the vector θ, not by changing X or y. Refer to the
equations in this handout and to the video lectures if you are uncertain.
A good way to verify that gradient descent is working correctly is to look
at the value of J(θ) and check that it is decreasing with each step. The
starter code for gradientDescent.m calls computeCost on every iteration
and prints the cost. Assuming you have implemented gradient descent and
computeCost correctly, your value of J(θ) should never increase, and should
converge to a steady value by the end of the algorithm.
After you are finished, ex1.m will use your final parameters to plot the
linear fit. The result should look something like Figure 2:
Your final values for θ will also be used to make predictions on profits in
areas of 35,000 and 70,000 people. Note the way that the following lines in
ex1.m uses matrix multiplication, rather than explicit summation or looping, to calculate the predictions.

![image](https://user-images.githubusercontent.com/30857998/36558879-64c425e2-1804-11e8-959d-f4b8ba8a5835.png)
