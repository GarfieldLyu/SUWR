
import tensorflow as tf
from tensorflow.keras.datasets import mnist, fashion_mnist
from sklearn.metrics import roc_auc_score, accuracy_score, average_precision_score
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
from data_generate_syn import generate_dataset


fashion_class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

image_data = ['mnist', 'fashion_mnist']
synthtic_benchmark = ['syn1', 'syn2', 'syn3', 'syn4', 'syn5', 'syn6']


def update_arguments(args: dict):
    """  Update model and training parameters based on user-defined arguments. """
    model_params, train_params = {}, {}
    model_params.update({'data_type': args['data_type'],
                         'model_type': args['model_type'],
                         'add_step': args['add_step'],
                        'hidden_dim': args['hidden_dim'], 
                        'num_layers': args['num_layers'],
                        'valid_metric_func': classification_performance_metric,
                        'exploration_epochs': args['exploration_epochs'],
                        'exploration': 1,
                        'exploration_stop': 1,
                        'max_budget': args['max_budget'],
                        'step_size': args['step_size'],
                        'lamda': args['lamda'],
                        'select_patch': args['select_patch']
                        })

    train_params.update({'name_your_model': args['name_your_model'],
                        'max_epochs': args['max_epochs'],
                        'patience': args['patience'],
                        'mode': 'max',
                        'batch_size': args['batch_size'],
                        'tries': args['tries'],
                        'device': args['device'],
                        'resume': args['resume'],
                        'eval_only': args['eval_only'],
                        'ckpt_eval': args['ckpt_eval'],
                        })
    
    if args['eval_only']:
        train_params.update('tries', 1)   # only evaluate once.
        assert args['ckpt_eval'] is not None, 'Please specify the checkpoint for evaluation.'

    if args['resume']:
        assert args['ckpt_eval'] is not None, 'Please specify the checkpoint for resuming training.'

    if args['data_type'] == 'toy':
        model_params.update({'input_dim': 10})
        model_params.update({'output_dim': 1})
        model_params.update({'loss_func': F.mse_loss})
        model_params.update({'eval_func': F.mse_loss})
        model_params.update({'task': 'regression'})
        train_params.update({'mode': 'min'})

    else:
        model_params.update({'task': 'classification'})
        train_params.update({'mode': 'max'})
        model_params.update({'loss_func': F.cross_entropy})
        model_params.update({'eval_func': F.cross_entropy})

        if args['data_type'] in image_data:
            model_params.update({'input_dim': 784})
            model_params.update({'output_dim': 10})
    
        elif args['data_type'] in synthtic_benchmark:
            model_params.update({'input_dim': 11})
            model_params.update({'output_dim': 2})

    return model_params, train_params


def load_image_data(data_type: str): 
    """
    Prepare image data for training and testing.
    data_type: str, 'mnist' or 'fashion_mnist'
    return: (x_train, y_train, x_test, y_test)
    """
    if data_type == 'mnist':
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
    elif data_type == 'fashion_mnist':
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    else:
        raise ValueError(f'Invalid data type {data_type}')
    x_train = x_train.reshape(60000, 784).astype('float64') / 255
    x_test = x_test.reshape(10000, 784).astype('float64') / 255
    data = (x_train, y_train, x_test, y_test)
    print(f'Loaded {data_type} data.')
    return data

def load_synthetic_data(data_type: str, num_samples: int, num_feat: int, seed: int):
    return generate_dataset(n = num_samples, dim = num_feat, data_type = data_type, seed = seed)

def classification_performance_metric(y_true, y_pred):
    """ Compute auroc, aupr, acc, given prediction and true labels. """
    if len(y_true.shape) < 2:
        y_true = tf.keras.utils.to_categorical(y_true, num_classes=10)
    auroc = roc_auc_score(y_true, y_pred)
    auprc = average_precision_score(y_true, y_pred)
    acc = accuracy_score(y_true.argmax(1), y_pred.argmax(1))
    return auroc, auprc, acc
    

def plot_results(x_test, y_test, pred_test, selection, num_per_class=2, save_path="selection.pdf"):
    if len(y_test.shape) > 1:
        y_test = np.argmax(y_test, axis = 1)
    fig = plt.figure(figsize=(20, 10))
    j = 1
    for num in range(10):
        for i in np.where(y_test==num)[0][:num_per_class]: 
            ax = fig.add_subplot(10, num_per_class, j)
            ax.imshow(selection[i,:].reshape((28,28)), cmap= 'Reds', alpha = 0.8)
            ax.imshow(x_test[i,:].reshape((28,28)), alpha = 0.1, cmap = 'Greys')
            ax.set_xlabel(f'pred: {np.argmax(pred_test[i,:])}')
            #ax.set_axis_off()
            j+=1
    fig.subplots_adjust(wspace=0, hspace=0)
    fig.savefig(save_path, bbox_inches='tight')
    plt.show()


def plot_steps(x_test, y_test, pred_test, selection, num_per_class=2, save_path="steps.pdf"):
    if len(y_test.shape) > 1:
        y_test = np.argmax(y_test, axis = 1)
    steps = selection.shape[1]
    fig = plt.figure(figsize=(20, 10))
    j = 1
    for num in range(10):
        for i in np.where(y_test==num)[0][0]:   # only plot the first sample
            ax = fig.add_subplot(10, steps, j)
            ax.imshow(selection[i,:].reshape((28,28)), cmap= 'Reds', alpha = 0.9)
            ax.imshow(x_test[i,:].reshape((28,28)), alpha = 0.2, cmap = 'Greys')
            ax.set_xlabel(f'step: {j%steps}')
            #ax.set_axis_off()
            j+=1
    fig.subplots_adjust(wspace=0, hspace=0)
    fig.savefig(save_path, bbox_inches='tight')
    plt.show()