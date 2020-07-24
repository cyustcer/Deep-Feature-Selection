# Source Codes
This folder contains the necessary files for the implement of our methods (`.py`) and comparison methods (`.r`). Please see the following for the details.

## Source Files

The source code files are organized in two perspectives. 
* `0i_xxx.xx`, the files start with number either `01` or `02` and followed by methods name is the implementation files of each methods.
   - `01` indicates linear regression examples
   - `02` indicates nonlinear classification examples
* `utils.xx`, `dfs.py` and `models.py`
   - `utils.xx` contains
        - dataset generators (linear_generator, nonlinear_generator), for simulation studies;
        - data loading function (data_load_l, data_load_n), for data read-in and pre-process;
        - metrics calculation functions (measure, accuracy, mse), for the calculation of (fsr, nsr), accuracy, mse;
        - model needed elements (WeightNorm, DotProduct), for the weight normalization and selection layer in PyTorch Neural network definition.
   - models.py contains Neural network models used for simulation studies
   - dfs.py contains function for one DFS algorithm iteration, and wrapped up functions for training simulation studies with fixed $s$.

## Packages
Here we list the packages we used for our methods and comparison methods

### Python packages for DFS

* [___Pytorch___](https://pytorch.org/)
* [___numpy___](https://numpy.org/)
* [___pandas___](https://pandas.pydata.org/)

### R packages

* [___glmnet___](https://cran.r-project.org/web/packages/glmnet/index.html) (For LASSO and Elastic Net)
* [___ncvreg___](https://cran.r-project.org/web/packages/ncvreg/index.html) (For SCAD)
* [___bartMachine___](https://cran.r-project.org/web/packages/bartMachine/index.html) (For Bayesian Additive Regression Tree)
* [___gamsel___](https://cran.r-project.org/web/packages/gamsel/index.html) (For Generalized Additive Models)
* [___BNN___](https://cran.r-project.org/web/packages/BNN/index.html) (For Bayesian Neural Networks)
* [___h2o___](https://cran.r-project.org/web/packages/h2o/index.html) (For Random Forest)

## Computing Environment
We have provided the guidance of implement our methods in `../docs/notebooks/`. If you are interested in replicate our results, please make sure you are in the same environment as ours.

### Python environment

Current code works under both Python 2 and Python 3.
To replicate results in paper, please create the environment following these steps:

```bash
conda env create -f dfs.yml
conda install pytorch==0.3.0.post4 cpuonly -c pytorch
```

It should create the same environment as ours, and make sure the following package version is the same.
Package Versions:

* Python:
    - numpy: 1.14.3
    - pandas: 0.21.0
    - torch: 0.3.0.post4
    


## Computing Resource

* CPU:
    - Intel(R) Xeon(R) CPU E5-2660 v3 @ 2.60GHz (1 thread per core, single thread used for the experiment)


### Guidance of code using:

After you have the required python and R packages installed, you can follow the steps to get a text report of multiple datasets.

For DFS method, each training codes consists of 4 major parts:

1. Data Generation (only need once for simulation each example, seed is set in generator functions)
2. Data Loading (see `../docs/notebooks/`)
3. Single dataset training and tuning for s (see `../docs/notebooks/`)
4. Multiple datasets training

Simply using command: `python xxx.py &` under `src` directory. Result files and reports will be saved under `outputs/reports` directory.
Here is the example report for the nonlinear classification example with 10 datasets:

```
For 10 datasets:
  False Selection Rate: 0.0
  Negative Selection Rate: 0.0
  Training error: 0.0(0.0)
  Testing error: 0.043000000000000003(0.008621678104251714)
For datasets 0:
    Optimal s: 4
    Training Accuracy: 1.0
    Testing Accuray: 0.9666666666666667
For datasets 1:
    Optimal s: 4
    Training Accuracy: 1.0
    Testing Accuray: 0.94
For datasets 2:
    Optimal s: 4
    Training Accuracy: 1.0
    Testing Accuray: 0.9533333333333334
For datasets 3:
    Optimal s: 4
    Training Accuracy: 1.0
    Testing Accuray: 0.97
For datasets 4:
    Optimal s: 4
    Training Accuracy: 1.0
    Testing Accuray: 0.95
For datasets 5:
    Optimal s: 4
    Training Accuracy: 1.0
    Testing Accuray: 0.96
For datasets 6:
    Optimal s: 4
    Training Accuracy: 1.0
    Testing Accuray: 0.9533333333333334
For datasets 7:
    Optimal s: 4
    Training Accuracy: 1.0
    Testing Accuray: 0.9533333333333334
For datasets 8:
    Optimal s: 4
    Training Accuracy: 1.0
    Testing Accuray: 0.9666666666666667
For datasets 9:
    Optimal s: 4
    Training Accuracy: 1.0
    Testing Accuray: 0.9566666666666667
```

For comparison methods, simply using command: `Rscript xxx.r &` under `src` directory. Result files and reports will be saved under `outputs/reports` directory.


