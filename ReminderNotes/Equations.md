# Mean Squared Error (MSE)

The Mean Squared Error (MSE) is a commonly used metric to evaluate the performance of regression models. It measures the average squared difference between the predicted and actual values.

- \( n \) is the number of data points (observations).
- \( y_i \) is the actual value for the \( i \)-th data point.
- \( \hat{y}_i \) is the predicted value for the \( i \)-th data point.
- \( (y_i - \hat{y}_i)^2 \) represents the squared difference between the actual and predicted values.

$$
\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
$$

# Cost Function \( J(\theta) \)

The cost function, denoted as \( J(\theta) \), is used to evaluate the performance of a linear regression model. It represents the average of the squared differences between the predicted values and the actual values, scaled by a factor of \( \frac{1}{2m} \).

- \( m \) is the number of data points (observations).
- \( y_i \) is the actual value for the \( i \)-th data point.
- \( \hat{y}_i \) is the predicted value for the \( i \)-th data point.
- \( h_\theta(x_i) \) is the hypothesis (predicted value) for the \( i \)-th data point.

$$
J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} (h_\theta(x_i) - y_i)^2
$$

# Hypothesis Function \( h_\theta(x) \)

The hypothesis function \( h_\theta(x) \) is used to predict the output \( \hat{y} \) given an input \( x \). In the context of simple linear regression, this function is a linear combination of the input feature(s) and the model parameters \( \theta_0 \) (intercept) and \( \theta_1 \) (slope).

- \( \theta_0 \) is the intercept of the linear model.
- \( \theta_1 \) is the slope of the linear model.
- \( x_i \) is the input feature for the \( i \)-th data point.

$$
h_\theta(x) = \theta_0 + \theta_1 x
$$

# Gradient Descent Algorithm

Gradient Descent is an optimization algorithm used to minimize the cost function \( J(\theta) \) by iteratively updating the model parameters \( \theta_0 \) and \( \theta_1 \) in the direction of the steepest descent (negative gradient).

- \( \alpha \) is the learning rate.
- \( \frac{\partial J(\theta)}{\partial \theta_j} \) represents the partial derivative of the cost function with respect to the parameter \( \theta_j \).

**Parameter Update Rule:**

$$
\theta_j := \theta_j - \alpha \frac{\partial J(\theta)}{\partial \theta_j}
$$

# Normal Equation

The Normal Equation provides a closed-form solution to find the optimal parameters \( \theta \) without requiring an iterative process like Gradient Descent.

- \( X \) is the matrix of input features.
- \( y \) is the vector of output values.
- \( \theta \) is the vector of model parameters.
- \( X^T \) is the transpose of the matrix \( X \).

$$
\theta = (X^T X)^{-1} X^T y
$$

# R-squared (Coefficient of Determination)

R-squared is a statistical measure that represents the proportion of the variance for the dependent variable that's explained by the independent variables in the regression model.

- \( SS_{res} \) is the sum of squares of residuals.
- \( SS_{tot} \) is the total sum of squares.

$$
R^2 = 1 - \frac{SS_{res}}{SS_{tot}}
$$

**Where:**
- \( SS_{res} = \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 \) is the sum of squared residuals.
- \( SS_{tot} = \sum_{i=1}^{n} (y_i - \bar{y})^2 \) is the total sum of squares, where \( \bar{y} \) is the mean of the actual values \( y_i \).
