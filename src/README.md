Please read before using provided codes !!!

The codes includes two major parts:

## Source Files

1. functions files (tools.py, models.py, dfs.py)
    * utils.py contains:
        - dataset generators (linear_generator, nonlinear_generator), for simulation studies;
        - data loading function (data_load_l, data_load_n), for data read-in and pre-process;
        - metrics calculation functions (measure, accuracy, mse), for the calculation of (fsr, nsr), accuracy, mse;
        - model needed elements (WeightNorm, DotProduct), for the weight normalization and selection layer in PyTorch Neural network definition.
    * models.py contains Neural network models used for simulation studies
    * dfs.py contains function for one DFS algorithm iteration, and wrapped up functions for training simulation studies
    
2. Training files (01_linear_DFS.py, 01_linear_Lasso_Elastic.py, 01_linear_SCAD.r, 02_nonlinear.py)
    * 01_linear.py: DFS models training for linear regression example
    * 02_nonlinear.py: DFS models training for nonlinear classification example

## Python environment

Current code works under both Python 2 and Python 3.

To use code in Python 2 & 3, please refer jupyter notebook in `../docs/notebooks/`
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
    - sklearn: 0.19.1
* R:
    - ncvreg: 3.10-0 
    


## Computing Resource

* CPU:
    - Intel(R) Xeon(R) CPU E5-2660 v3 @ 2.60GHz (1 thread per core, single thread used for the experiment)


## Guidance of code using:

Each training codes consists of 4 major parts:

1. Data Generation (only need once for simulation each example, seed is set in generator functions)
2. Data Loading (see `../docs/notebooks/`)
3. Single dataset training and tuning for s (see `../docs/notebooks/`)
4. Multiple datasets training

For DFS method, simply using command: "python xxx.py &" under `src` directory. Result files and reports will be saved under current directory.
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


