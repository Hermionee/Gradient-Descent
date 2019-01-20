# ML-HW2
This project compares seven gradient descent algorithms for logistic regression for binary classification, in order to compare the convergency rate and performances.
We implemented those algorithms and ran them on iris dataset and breast cancer dataset, using cross entropy and squared loss functions. Firstly, we worked on iris dataset.
The result is as follows:

<a href="url"><img src="https://github.com/Hermionee/Gradient-Descent/blob/master/figures/iris_mse.jpg" align="center" height="400" width="600" ></a>

<a href="url"><img src="https://github.com/Hermionee/Gradient-Descent/blob/master/figures/iris_ce.png" align="center" height="400" width="600" ></a>

Each algorithm starts with initial weight vector(zero). From the above figures, we can see that SGD and GD works pretty good and converge fast. While the adaptive algorithms converge pretty slowly and unstable because now the dataset is pretty small and only have four features. In this simple case, SGD and GD outperforms adaptive methods(Adagrad, RMSprop and Adadelta). It is worth notice that Adam doesn’t seem to decrease both the two loss functions. It’s because the decreasing rate of the gradient of both the two loss functions are so fast that the momentum term outweighs gradient term in updating. So Adam seems to converge early and doesn’t arrive at minimum. Next, we used breast dataset to test the techniques when feature number becomes 9. The results are as follows:

<a href="url"><img src="https://github.com/Hermionee/Gradient-Descent/blob/master/figures/breast_ce.png" align="center" height="400" width="600" ></a>

<a href="url"><img src="https://github.com/Hermionee/Gradient-Descent/blob/master/figures/breast_mse.png" align="center" height="400" width="600" ></a>

From the results above, we see that except the adaptive methods, the other techniques work pretty fast and stable. This means that, still, the small feature space prevents the adaptive methods from performing themselves. To zoom in the first 50 iterations, we get the following result.

<a href="url"><img src="https://github.com/Hermionee/Gradient-Descent/blob/master/figures/breast_50_ce.png" align="center" height="400" width="600" ></a>

<a href="url"><img src="https://github.com/Hermionee/Gradient-Descent/blob/master/figures/breast_50_mse.png" align="center" height="400" width="600" ></a>

From the first 50 iterations, we can see that Adagrad, RMSprop and Adadelta still works slowly while others converge really fast. And SGDM and Adam outperms GD and SGD because the gradient now is in normal range (unlike in the previous dataset). And because both loss functions are convex, no techniques get stuck in the local minima.
