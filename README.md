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



# XGBoost math, if you are interested

XGBoost（Extreme Gradient Boosting）是一种基于梯度提升树的机器学习算法，其核心数学原理如下：

---

## 1. 目标函数
XGBoost 的目标函数由**损失函数**和**正则化项**组成：
$$
\text{Obj}(\theta) = \sum_{i=1}^n L(y_i, \hat{y}_i) + \sum_{k=1}^K \Omega(f_k)
$$

- **损失函数**：$L(y_i, \hat{y}_i)$ 衡量预测值 $\hat{y}_i$ 与真实值 $y_i$ 的差异  
  常用损失函数：均方误差（MSE）、交叉熵（Cross-Entropy）等  
- **正则化项**：$\Omega(f_k)$ 控制模型复杂度，防止过拟合  
  $$ 
  \Omega(f_k) = \gamma T + \frac{1}{2}\lambda \sum_{j=1}^T w_j^2 
  $$  
  其中 $T$ 为叶子节点数，$w_j$ 为叶子权重，$\gamma$ 和 $\lambda$ 为超参数

---

## 2. 加法模型
XGBoost 通过逐步添加弱学习器（决策树）构建模型：
$$
\hat{y}_i^{(t)} = \hat{y}_i^{(t-1)} + f_t(x_i)
$$
- $\hat{y}_i^{(t)}$：第 $t$ 轮迭代后的预测值  
- $f_t(x_i)$：第 $t$ 轮新增的决策树

---

## 3. 泰勒展开近似
目标函数在第 $t$ 轮的泰勒二阶展开近似：
$$
\text{Obj}^{(t)} \approx \sum_{i=1}^n \left[ L(y_i, \hat{y}_i^{(t-1)}) + g_i f_t(x_i) + \frac{1}{2} h_i f_t^2(x_i) \right] + \Omega(f_t)
$$
- $g_i = \partial_{\hat{y}^{(t-1)}} L(y_i, \hat{y}^{(t-1)})$：损失函数的一阶导数  
- $h_i = \partial_{\hat{y}^{(t-1)}}^2 L(y_i, \hat{y}^{(t-1)})$：损失函数的二阶导数

---

## 4. 最优叶子权重
对于每个叶子节点 $j$，最优权重 $w_j^*$ 为：
$$
w_j^* = -\frac{\sum_{i \in I_j} g_i}{\sum_{i \in I_j} h_i + \lambda}
$$
- $I_j$：属于叶子节点 $j$ 的样本集合

---

## 5. 节点分裂准则
分裂节点时通过最大化**增益（Gain）**选择最优分割：
$$
\text{Gain} = \frac{1}{2} \left[ \frac{(\sum_{i \in I_L} g_i)^2}{\sum_{i \in I_L} h_i + \lambda} + \frac{(\sum_{i \in I_R} g_i)^2}{\sum_{i \in I_R} h_i + \lambda} - \frac{(\sum_{i \in I} g_i)^2}{\sum_{i \in I} h_i + \lambda} \right] - \gamma
$$
- $I_L/I_R$：分裂后左右子节点的样本集合  
- $\gamma$：分裂代价的超参数

---

## 6. 算法流程
1. 初始化模型：$\hat{y}_i^{(0)} = \text{base\_model}$  
2. 对每轮迭代 $t=1,2,...,T$：  
   - 计算 $g_i$ 和 $h_i$  
   - 生成新决策树 $f_t$ 以最小化目标函数  
   - 更新模型：$\hat{y}_i^{(t)} = \hat{y}_i^{(t-1)} + \eta f_t(x_i)$  
     （$\eta$ 为学习率）
