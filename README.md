## About  
**EwigesKind** is a growing suite of **machine learning (ML) and deep reinforcement learning (DRL) models** for geochemical classification and analysis. The name, meaning **“Eternal Child”** in German, reflects an insatiable curiosity for the Earth—one that never ceases to question, explore, and uncover the unknown.  

### EwigesKind X1: First Generation  
EwigesKind X1 is the first model in this series, where **X** comes from **["XGBoost"](https://xgboost.readthedocs.io/en/latest/)** in Julia. It focuses on classifying **igneous rocks** using **limited rare earth element (REE) data**, trained on **19,000+ samples** from a global geochemical database.  

### Models in Development  
- **EwigesKind X1o** – The TerraneChron Method  
- **EwigesKind X2** – PCA-filtered classification  
- **EwigesKind L2** – LightGBM with PCA filtering  
- **EwigesKind R-series** – Fine-tuned **LLMs** (e.g., DeepSeek) for multi-model benchmarking (compare transformer-based models with traditional decision-tree models on geochemical datasets)

### Future Plans  
- Data and training visualization
- Possible GUI integration *(based on demand)*


---


# How to Use

### Prerequisites
Ensure you have a **[Julia](https://julialang.org/downloads/)** environment set up.

### Installation
Run [`Pkg_install.jl`](https://github.com/Yoli-Wu/EwigesKind-X1/blob/main/Pkg_install.jl) to install dependencies.

### Training the Model
To train the model on your own computer:
1. Download the following files:
   - [`earthchem_download_32495.csv`](https://github.com/Yoli-Wu/EwigesKind-X1/blob/main/Data%20and%20Reference/earthchem_download_32495.csv)
   - [`Train.jl`](https://github.com/Yoli-Wu/EwigesKind-X1/blob/main/src/Train.jl)
   - [`Predict.jl`](https://github.com/Yoli-Wu/EwigesKind-X1/blob/main/src/Predict.jl)

2. Modify **line 26** in `Train.jl` to reflect the absolute path of the downloaded CSV file:

   ```julia
   file_path = "C:\your_own_path\earthchem_download_32495.csv"
   ```

4. Run `Train.jl`. Upon successful execution, you should see:
   
    ![Training Output](https://github.com/Yoli-Wu/EwigesKind-X1/blob/main/img/training.png)

### Making Predictions
1. Insert **REE data** between **line 51 to line 54** in `Predict.jl`.
2. Run `Predict.jl` to generate predictions and explore the model's output.

---

## Demos
### Demo 1
- Example result:

  ![Demo 1](https://github.com/Yoli-Wu/EwigesKind-X1/blob/main/img/demo_1.png)

  ![Result Demo](https://github.com/Yoli-Wu/EwigesKind-X1/blob/main/img/result_demo.png)

### Demo 2
The model struggles with prediction when too much data is missing but can still predict the first two categories (**material** and **type**).
- Example result:

  ![Demo 2](https://github.com/Yoli-Wu/EwigesKind-X1/blob/main/img/demo_2.png)

  ![If Missing](https://github.com/Yoli-Wu/EwigesKind-X1/blob/main/img/if_missing.png)
