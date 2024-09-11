# Linear Regression and Gradient Descent

## Linear Regression Model

- **Model:**  
  \( f(w, b) = wx + b \)

- **Cost Function:**  
  \( J(w, b) = \frac{1}{2m} \sum (f(w, b) - y)^2 \)

## Gradient Descent Algorithm

- **Update Rule for \( w \):**  
  \( w = w - \frac{\alpha}{m} \sum (f(w, b) - y) x \)

- **Update Rule for \( b \):**  
  \( b = b - \frac{\alpha}{m} \sum (f(w, b) - y) \)

## Multiple Linear Regression Model

- **Model:**  
  \( f(w, b) = w_1 x_1 + w_2 x_2 + \ldots + w_n x_n + b \)

- **Cost Function:**  
  \( J(w, b) = \frac{1}{2m} \sum (f(w, b) - y)^2 \)

## Derivatives for Gradient Descent

- **Derivative with respect to \( w \):**  
  \( \frac{1}{m} \sum (f(w, b) - y) x \)

- **Derivative with respect to \( b \):**  
  \( \frac{1}{m} \sum (f(w, b) - y) \)
