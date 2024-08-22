# Machine Learning Architectures Cheatsheet

## 1. **Linear Models**

### **1.1. Linear Regression**
- **Purpose**: Predict continuous outcomes.
- **Key Components**:
  - **Model**: \( y = \beta_0 + \beta_1 x + \epsilon \)
  - **Loss Function**: Mean Squared Error (MSE)
  - **Optimization**: Gradient Descent

### **1.2. Logistic Regression**
- **Purpose**: Classification (binary or multi-class).
- **Key Components**:
  - **Model**: \( p = \frac{1}{1 + e^{-(\beta_0 + \beta_1 x)}} \)
  - **Loss Function**: Binary Cross-Entropy (for binary classification)
  - **Optimization**: Gradient Descent

## 2. **Decision Trees**

### **2.1. Decision Tree Classifier**
- **Purpose**: Classification tasks.
- **Key Components**:
  - **Model**: Tree structure with nodes (decision points) and leaves (outcomes).
  - **Splitting Criteria**: Gini Impurity, Entropy
  - **Pruning**: Reduces overfitting

### **2.2. Decision Tree Regressor**
- **Purpose**: Regression tasks.
- **Key Components**:
  - **Model**: Tree structure similar to classification, but with continuous outputs.
  - **Splitting Criteria**: Mean Squared Error (MSE)

## 3. **Ensemble Methods**

### **3.1. Random Forest**
- **Purpose**: Classification and regression.
- **Key Components**:
  - **Model**: Collection of decision trees (forest).
  - **Aggregation**: Majority voting (classification) or averaging (regression).

### **3.2. Gradient Boosting Machines (GBMs)**
- **Purpose**: Classification and regression.
- **Key Components**:
  - **Model**: Sequentially trained decision trees (boosting).
  - **Loss Function**: Customizable per task (e.g., MSE, Log Loss).

## 4. **Support Vector Machines (SVM)**

### **4.1. SVM for Classification**
- **Purpose**: Classification tasks.
- **Key Components**:
  - **Model**: Finds the optimal hyperplane that maximizes the margin between classes.
  - **Kernel Functions**: Linear, Polynomial, RBF (Radial Basis Function)

### **4.2. SVM for Regression (SVR)**
- **Purpose**: Regression tasks.
- **Key Components**:
  - **Model**: Finds the optimal hyperplane that fits within a margin of tolerance.

## 5. **Neural Networks**

### **5.1. Feedforward Neural Networks (FNN)**
- **Purpose**: Classification and regression.
- **Key Components**:
  - **Layers**: Input, Hidden, Output
  - **Activation Functions**: ReLU, Sigmoid, Tanh
  - **Loss Function**: Cross-Entropy, MSE

### **5.2. Convolutional Neural Networks (CNN)**
- **Purpose**: Image and spatial data processing.
- **Key Components**:
  - **Layers**: Convolutional layers, Pooling layers, Fully connected layers
  - **Activation Functions**: ReLU
  - **Pooling**: Max Pooling, Average Pooling

### **5.3. Recurrent Neural Networks (RNN)**
- **Purpose**: Sequential data and time series.
- **Key Components**:
  - **Layers**: Recurrent layers, such as LSTM (Long Short-Term Memory) or GRU (Gated Recurrent Unit)
  - **Activation Functions**: Sigmoid, Tanh

### **5.4. Transformers**
- **Purpose**: Sequence-to-sequence tasks, NLP.
- **Key Components**:
  - **Layers**: Self-Attention, Feedforward Neural Networks
  - **Key Components**: Encoder-Decoder architecture
  - **Attention Mechanism**: Multi-Head Attention

## 6. **Generative Models**

### **6.1. Generative Adversarial Networks (GANs)**
- **Purpose**: Generate new data samples.
- **Key Components**:
  - **Models**: Generator (creates samples), Discriminator (evaluates samples)
  - **Loss Function**: Adversarial loss

### **6.2. Variational Autoencoders (VAEs)**
- **Purpose**: Generate new samples and learn latent representations.
- **Key Components**:
  - **Models**: Encoder (maps to latent space), Decoder (reconstructs data from latent space)
  - **Loss Function**: Reconstruction loss + Kullback-Leibler divergence

### **6.3. Normalizing Flows**
- **Purpose**: Model complex distributions with invertible transformations.
- **Key Components**:
  - **Models**: Series of invertible transformations (e.g., RealNVP, Glow)
  - **Loss Function**: Likelihood of data under the model

### **6.4. Diffusion Models**
- **Purpose**: Generate high-quality samples by simulating diffusion processes.
- **Key Components**:
  - **Models**: Forward and reverse diffusion processes
  - **Loss Function**: Denoising score matching

## 7. **Reinforcement Learning**

### **7.1. Q-Learning**
- **Purpose**: Learn action-value functions in environments.
- **Key Components**:
  - **Model**: Q-table or Q-network
  - **Algorithm**: Update Q-values using Bellman Equation

### **7.2. Policy Gradient Methods**
- **Purpose**: Directly optimize policy functions.
- **Key Components**:
  - **Models**: Policy network
  - **Algorithm**: Optimize policy using gradient ascent

## 8. **Unsupervised Learning**

### **8.1. K-Means Clustering**
- **Purpose**: Group data into clusters.
- **Key Components**:
  - **Algorithm**: Iterative refinement of cluster centroids
  - **Distance Metric**: Euclidean distance

### **8.2. Principal Component Analysis (PCA)**
- **Purpose**: Dimensionality reduction.
- **Key Components**:
  - **Algorithm**: Projects data onto principal components
  - **Objective**: Maximize variance explained

---

Feel free to copy and use this updated cheatsheet for a quick reference to various machine learning architectures, including the additional models. If you have any more questions or need further details on any topic, just let me know!
