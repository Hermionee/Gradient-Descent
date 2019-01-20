# ML-HW2
This project compares seven gradient descent algorithms for logistic regression for binary classification, in order to compare the convergency rate and performances.
We implemented those algorithms and ran them on iris dataset and breast cancer dataset, using cross entropy and squared loss functions. Firstly, we worked on iris dataset.
The result is as follows:
![alt text](https://github.com/Hermionee/Gradient-Descent/blob/master/figures/iris_mse.jpg)
![alt text](https://github.com/Hermionee/Gradient-Descent/blob/master/figures/iris_ce.png)

Each algorithm starts with initial weight vector(zero). From the above figures, we can see that SGD and GD works pretty good and converge fast. While the adaptive algorithms converge pretty slowly and unstable because now the dataset is pretty small and only have four features. In this simple case, SGD and GD outperforms adaptive methods(Adagrad, RMSprop and Adadelta). It is worth notice that Adam doesn’t seem to decrease both the two loss functions. It’s because the decreasing rate of the gradient of both the two loss functions are so fast that the momentum term outweighs gradient term in updating. So Adam seems to converge early and doesn’t arrive at minimum. Next, we used breast dataset to test the techniques when feature number becomes 9. The results are as follows:
![alt text](https://github.com/Hermionee/Gradient-Descent/blob/master/figures/breast_ce.png)
![alt text](https://github.com/Hermionee/Gradient-Descent/blob/master/figures/breast_mse.png)


From the results above, we see that except the adaptive methods, the other techniques work pretty fast and stable. This means that, still, the small feature space prevents the adaptive methods from performing themselves. To zoom in the first 50 iterations, we get the following result.

