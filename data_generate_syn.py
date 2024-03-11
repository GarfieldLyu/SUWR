"""Functions for synthetic data generation

Reference: Jinsung Yoon, James Jordon, Mihaela van der Schaar, 
           "IINVASE: Instance-wise Variable Selection using Neural Networks," 
           International Conference on Learning Representations (ICLR), 2019.
Paper Link: https://openreview.net/forum?id=BJg_roAcK7
Contact: jsyoon0823@gmail.com

---------------------------------------------------

Generating 6 synthetic datasets

x ~ N(0,I) where d = 11 or 100

y = 1/(1+logit) where logit for each synthetic dataset is
- syn1: logit = exp(x1 * x2)
- syn2: logit = exp(x3^2 + x4^2 + x5^2 + x6^2 -4)
- syn3: logit = -10 sin(2 * x7) + 2|x8| + x9 + exp(-x10) - 2.4
- syn4: If x11 < 0, follows syn1, else if x11 >= 0, follows syn2
- syn5: If x11 < 0, follows syn1, else if x11 >= 0, follows syn3
- syn6: If x11 < 0, follows syn2, else if x11 >= 0, follows syn3
"""

# Necessary packages
import numpy as np 
import torch
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score

keyfeatures = {
  'syn1': 2,
  'syn2': 4,
  'syn3': 4,
  'syn4': 7,
  'syn5': 7,
  'syn6': 9
}

def generate_x (n, dim):
  """Generate the features (x).
  
  Args:
    - n: the number of samples
    - dim: the number of features (feature dimensions)
    
  Returns:
    - x: (n x dim) data points sample from N(0, 1)
  """
  x = np.random.randn(n, dim)
  return x


def generate_y (x, data_type):
  """Generate corresponding label (y) given feature (x).
  
  Args:
    - x: features
    - data_type: synthetic data type (syn1 to syn6)

  Returns:
    - y: corresponding labels
  """
  # number of samples
  n = x.shape[0]
    
  # Logit computation
  if data_type == 'syn1':
    logit = np.exp(x[:, 0]*x[:, 1])
  elif data_type == 'syn2':       
    logit = np.exp(np.sum(x[:, 2:6]**2, axis = 1) - 4.0) 
  elif data_type == 'syn3':
    logit = np.exp(-10 * np.sin(0.2*x[:, 6]) + abs(x[:, 7]) + \
                   x[:, 8] + np.exp(-x[:, 9])  - 2.4)     
  elif data_type == 'syn4':
    logit1 = np.exp(x[:, 0]*x[:, 1])
    logit2 = np.exp(np.sum(x[:, 2:6]**2, axis = 1) - 4.0) 
  elif data_type == 'syn5':
    logit1 = np.exp(x[:, 0]*x[:, 1])
    logit2 = np.exp(-10 * np.sin(0.2*x[:, 6]) + abs(x[:, 7]) + \
                    x[:, 8] + np.exp(-x[:, 9]) - 2.4) 
  elif data_type == 'syn6':
    logit1 = np.exp(np.sum(x[:,2:6]**2, axis = 1) - 4.0) 
    logit2 = np.exp(-10 * np.sin(0.2*x[:, 6]) + abs(x[:, 7]) + \
                    x[:, 8] + np.exp(-x[:, 9]) - 2.4) 
    
  # For syn4, syn5 and syn6 only
  if data_type in ['syn4', 'syn5', 'syn6']:
    # Based on X[:,10], combine two logits        
    idx1 = (x[:, 10]< 0)*1
    idx2 = (x[:, 10]>=0)*1    
    logit = logit1 * idx1 + logit2 * idx2    
        
  # Compute P(Y=0|X)
  prob_0 = np.reshape((logit / (1+logit)), [n, 1])
    
  # Sampling process
  y = np.zeros([n, 2])
  y[:, 0] = np.reshape(np.random.binomial(1, prob_0), [n,])
  y[:, 1] = 1-y[:, 0]

  return y


def generate_ground_truth(x, data_type):
  """Generate ground truth feature importance corresponding to the data type
     and feature.
  
  Args:
    - x: features
    - data_type: synthetic data type (syn1 to syn6)

  Returns:
    - ground_truth: corresponding ground truth feature importance
  """

  # Number of samples and features
  n, d = x.shape

  # Output initialization
  ground_truth = np.zeros([n, d])
        
  # For each data_type
  if data_type == 'syn1':
    ground_truth[:, :2] = 1
  elif data_type == 'syn2':
    ground_truth[:, 2:6] = 1
  elif data_type == 'syn3':
    ground_truth[:, 6:10] = 1
        
  # Index for syn4, syn5 and syn6
  if data_type in ['syn4', 'syn5', 'syn6']:        
    idx1 = np.where(x[:, 10]<0)[0]
    idx2 = np.where(x[:, 10]>=0)[0]
    ground_truth[:, 10] = 1
        
  if data_type == 'syn4':        
    ground_truth[idx1, :2] = 1
    ground_truth[idx2, 2:6] = 1
  elif data_type == 'syn5':        
    ground_truth[idx1, :2] = 1
    ground_truth[idx2, 6:10] = 1
  elif data_type == 'syn6':        
    ground_truth[idx1, 2:6] = 1
    ground_truth[idx2, 6:10] = 1
        
  return ground_truth

    
def generate_dataset(n = 10000, dim = 11, data_type = 'syn1', seed = 0):
  """Generate dataset (x, y, ground_truth).
  
  Args:
    - n: the number of samples
    - dim: the number of dimensions
    - data_type: synthetic data type (syn1 to syn6)
    - seed: random seed
    
  Returns:
    - x: features
    - y: labels
    - ground_truth: ground truth feature importance
  """

  # Seed
  np.random.seed(seed)

  # x generation
  x = generate_x(n, dim)
  # y generation
  y = generate_y(x, data_type)
  # ground truth generation
  ground_truth = generate_ground_truth(x, data_type)
  
  return x, y, ground_truth


def feature_performance_metric (ground_truth, importance_score):
  """Performance metrics for feature importance (TPR and FDR).
  
  Args:
    - ground_truth: ground truth feature importance
    - importance_score: computed importance scores for each feature
    
  Returns:
    - mean_tpr: mean value of true positive rate
    - std_tpr: standard deviation of true positive rate
    - mean_fdr: mean value of false discovery rate
    - std_fdr: standard deviation of false discovery rate
  """

  n = ground_truth.shape[0]
  #print(importance_score.shape, importance_score)
  if importance_score.shape[0] != n:
    importance_score = importance_score.repeat(n, axis=0)
  
  print(f'check shape: {ground_truth.shape}, {importance_score.shape}.')


  tpr = np.zeros([n, ])
  fdr = np.zeros([n, ])

  # For each sample
  for i in range(n):    
    # tpr   
    tpr_nom = np.sum(importance_score[i, :] * ground_truth[i, :])
    tpr_den = np.sum(ground_truth[i, :])
    tpr[i] = 100 * float(tpr_nom)/float(tpr_den + 1e-8)

    # fdr
    fdr_nom = np.sum(importance_score[i, :] * (1-ground_truth[i, :]))
    fdr_den = np.sum(importance_score[i,:])
    fdr[i] = 100 * float(fdr_nom)/float(fdr_den+1e-8)
    
  mean_tpr = np.mean(tpr)
  std_tpr = np.std(tpr)
  mean_fdr = np.mean(fdr)
  std_fdr = np.std(fdr)  
  
  return mean_tpr, std_tpr, mean_fdr, std_fdr


def prediction_performance_metric(y_test, y_hat):
  """Performance metrics for prediction (AUC, APR, Accuracy).
  
  Args:
    - y_test: testing set labels
    - y_hat: prediction on testing set
    
  Returns:
    - auc: area under roc curve
    - apr: average precision score
    - acc: accuracy
  """
  
  auc = roc_auc_score (y_test[:, 1], y_hat[:, 1])
  apr = average_precision_score (y_test[:, 1], y_hat[:, 1])
  acc = accuracy_score (y_test[:, 1], 1.*(y_hat[:, 1] > 0.5))
  
  return auc, apr, acc


def evaluate_synthetic(Y_test, gt_test, Pred_test, feat_test):
  """ Evaluate feature importance and prediction performance on synthetic data."""

  # copy feature importance metrics from INVASE
  mean_tpr, std_tpr, mean_fdr, std_fdr = feature_performance_metric(gt_test, feat_test)
  auc, apr, acc = prediction_performance_metric(Y_test, Pred_test)

  performance = {'tpr_mean': mean_tpr, 'tpr_std': std_tpr, 'fdr_mean': mean_fdr, 'fdr_std': std_fdr,
                  'auc': auc, 'apr': apr, 'acc': acc}
  return performance

