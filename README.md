## About  
**EwigesKind** is a growing suite of **machine learning (ML) and deep reinforcement learning (DRL) models** for geochemical classification and analysis. The name, meaning **“Eternal Child”** in German, reflects an insatiable curiosity for the Earth—one that never ceases to question, explore, and uncover the unknown.  

### EwigesKind X1: First Generation  
EwigesKind X1 is the first model in this series, where **X** comes from **["XGBoost"](https://xgboost.readthedocs.io/en/latest/)** in **[Julia](https://julialang.org/downloads/)**. It focuses on classifying **igneous rocks** using **limited rare earth element (REE) data**, trained on **19,000+ samples** from a global geochemical database.  

### Models in Development  
- **EwigesKind X1o** – The TerraneChron Method  
- **EwigesKind X2** – PCA-filtered classification  
- **EwigesKind L2** – LightGBM with PCA filtering  
- **EwigesKind R-series** – Fine-tuned **LLMs** (e.g., DeepSeek) for multi-model benchmarking  

### Future Plans  
- **Data and training visualization**  
- **Possible GUI integration** *(based on demand)*  

# Math, if you are interested

XGBoost (Extreme Gradient Boosting) is a gradient-boosted tree-based machine learning algorithm with the following core mathematical principles:

---

## 1. Objective Function
The objective function consists of **loss function** and **regularization term**:
$$
\text{Obj}(\theta) = \sum_{i=1}^n L(y_i, \hat{y}_i) + \sum_{k=1}^K \Omega(f_k)
$$

- **Loss function**: $L(y_i, \hat{y}_i)$ measures the difference between predicted value $\hat{y}_i$ and true value $y_i$  
  Common loss functions: Mean Squared Error (MSE), Cross-Entropy, etc.  
- **Regularization term**: $\Omega(f_k)$ controls model complexity to prevent overfitting  
  $$ 
  \Omega(f_k) = \gamma T + \frac{1}{2}\lambda \sum_{j=1}^T w_j^2 
  $$  
  Where $T$ is the number of leaf nodes, $w_j$ are leaf weights, $\gamma$ and $\lambda$ are hyperparameters

---

## 2. Additive Model
XGBoost builds models by sequentially adding weak learners (decision trees):
$$
\hat{y}_i^{(t)} = \hat{y}_i^{(t-1)} + f_t(x_i)
$$
- $\hat{y}_i^{(t)}$: Predicted value after $t$ iterations  
- $f_t(x_i)$: New decision tree added in the $t$-th iteration

---

## 3. Taylor Expansion Approximation
Second-order Taylor approximation of the objective function at the $t$-th iteration:
$$
\text{Obj}^{(t)} \approx \sum_{i=1}^n \left[ L(y_i, \hat{y}_i^{(t-1)}) + g_i f_t(x_i) + \frac{1}{2} h_i f_t^2(x_i) \right] + \Omega(f_t)
$$
- $g_i = \partial_{\hat{y}^{(t-1)}} L(y_i, \hat{y}^{(t-1)})$: First-order gradient of the loss function  
- $h_i = \partial_{\hat{y}^{(t-1)}}^2 L(y_i, \hat{y}^{(t-1)})$: Second-order gradient of the loss function

---

## 4. Optimal Leaf Weights
For each leaf node $j$, the optimal weight $w_j^*$ is:
$$
w_j^* = -\frac{\sum_{i \in I_j} g_i}{\sum_{i \in I_j} h_i + \lambda}
$$
- $I_j$: Sample set belonging to leaf node $j$

---

## 5. Node Splitting Criterion
Maximize the **Gain** to select optimal splits:
$$
\text{Gain} = \frac{1}{2} \left[ \frac{(\sum_{i \in I_L} g_i)^2}{\sum_{i \in I_L} h_i + \lambda} + \frac{(\sum_{i \in I_R} g_i)^2}{\sum_{i \in I_R} h_i + \lambda} - \frac{(\sum_{i \in I} g_i)^2}{\sum_{i \in I} h_i + \lambda} \right] - \gamma
$$
- $I_L/I_R$: Sample sets in left/right child nodes after splitting  
- $\gamma$: Hyperparameter for split cost

---

## 6. Algorithm Workflow
1. Initialize model: $\hat{y}_i^{(0)} = \text{base\_model}$  
2. For each iteration $t=1,2,...,T$:  
   - Compute $g_i$ and $h_i$  
   - Generate new decision tree $f_t$ to minimize objective function  
   - Update model: $\hat{y}_i^{(t)} = \hat{y}_i^{(t-1)} + \eta f_t(x_i)$  
     ($\eta$ is learning rate)
