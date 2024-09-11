# Linear Regression and Gradient Descent

## Linear Regression Model

- **Model:**  
  \( f(w, b) = wx + b \)  
  This is the hypothesis function for linear regression, where:
  - \( w \) is the weight (slope of the line),
  - \( b \) is the bias (intercept of the line),
  - \( x \) is the input feature.

- **Cost Function:**  
  \( J(w, b) = \frac{1}{2m} \sum (f(w, b) - y)^2 \)  
  This function measures the average squared difference between the predicted values (\( f(w, b) \)) and the actual values (\( y \)). The goal is to minimize this cost function during training.

## Gradient Descent Algorithm

- **Update Rule for \( w \):**  
  \( w = w - \frac{\alpha}{m} \sum (f(w, b) - y) x \)  
  This is the rule used to update the weight parameter \( w \). Here:
  - \( \alpha \) is the learning rate, which controls the step size of each update,
  - \( m \) is the number of training examples,
  - The term \( \sum (f(w, b) - y) x \) represents the gradient of the cost function with respect to \( w \).

- **Update Rule for \( b \):**  
  \( b = b - \frac{\alpha}{m} \sum (f(w, b) - y) \)  
  This is the rule used to update the bias parameter \( b \). Similar to the weight update, this rule adjusts \( b \) based on the gradient of the cost function with respect to \( b \).

## Multiple Linear Regression Model

- **Model:**  
  \( f(w, b) = w_1 x_1 + w_2 x_2 + \ldots + w_n x_n + b \)  
  This is an extension of the linear regression model to multiple features. Each feature \( x_i \) has a corresponding weight \( w_i \), and \( b \) is the bias term.

- **Cost Function:**  
  \( J(w, b) = \frac{1}{2m} \sum (f(w, b) - y)^2 \)  
  The cost function remains the same as in simple linear regression, but it now considers multiple features.

## Derivatives for Gradient Descent

- **Derivative with respect to \( w \):**  
  \( \frac{1}{m} \sum (f(w, b) - y) x \)  
  This derivative measures how the cost function changes with respect to changes in the weight parameter \( w \). It is used to compute the gradient and update \( w \) accordingly.

- **Derivative with respect to \( b \):**  
  \( \frac{1}{m} \sum (f(w, b) - y) \)  
  This derivative measures how the cost function changes with respect to changes in the bias term \( b \). It is used to compute the gradient and update \( b \) accordingly.

These equations and their explanations form the foundation for training linear regression models using gradient descent. The goal is to find the optimal parameters \( w \) and \( b \) that minimize the cost function and provide the best fit to the training data.
