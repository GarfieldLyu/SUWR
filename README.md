# SUWR
This repository contains the source code for our paper **Local Feature Selection without Label or Feature Leakage for Interpretable Machine Learning Predictions**, accepted at ICML 2024. 

<img src="./Images/fig_fashion.pdf">

### Experiment 1: Pareto Front Analysis
We first generate a toy dataset with 10 binary features, resulting in 2^10 = 1024 data samples. To generate the dataset, simply call `load_pareto_data` in `utils.py` 

Run the following line to train SUWR on toy dataset (note this is a regression task):

```bash
python run_pareto.py --data_type toy --model_type simple --max_budget 10 --lamda 0.5 --name_your_model my_toy_model
```

### Experiment 2: Synthetic Benchmark
This experiment is conducted on 6 synthetic datasets established in previous work. We took the data generation code from [INVASE](https://github.com/jsyoon0823/INVASE/). To generate the dataset, simply call `load_synthetic_data` in `utils.py`. 

Run the following line to train SUWR on Synthetic Benchmark (classification task):

```bash
python run_syns.py --data_type syn6 --model_type simple --max_budget 6 --lamda 0.01 --name_your_model my_synthetic_model
```

### Experiment 3: MNIST Digits and Fashion
For this experiment, we include both digit-MNIST and fashion-MNIST (classification task). 

Run the following line to train SUWR on image dataset:
```bash
python run_images.py --data_type digit_mnist, --model_type simple, --max_budget 50 --lamda 0.2 --name_your_model my_image_model --select_patch
```




