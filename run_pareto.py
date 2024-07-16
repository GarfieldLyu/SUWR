import numpy as np
import json
import torch
from torch.utils.data import TensorDataset, DataLoader
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from suwr import NonLeakingSelector
from utils import load_pareto_data, update_arguments
from tqdm import tqdm
import argparse


project_dir = "set your own project directory here"


def build_model(kwargs: dict):
    """
    Init model for training.
    kwargs: dict, hyperparameters for model.
    """
    model = NonLeakingSelector(kwargs).double()
    return model


def main(args: dict):
    """ Regression task with MSE loss."""
    model_params, train_params = update_arguments(args)

    (X, Y) = load_pareto_data(args['num_feat']) 
    train_dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(Y)) 
    valid_dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(Y)) # use the same data for training and validation.
       
    for t in tqdm(range(train_params['tries'])):
        seed_everything(t, workers=True)
        train_dataloader = DataLoader(train_dataset, batch_size=train_params['batch_size'], num_workers=8)
        valid_dataloader = DataLoader(valid_dataset, batch_size=train_params['batch_size'], num_workers=8)
        
        """===================== Train model and observe results on tensorboard ======================"""
        
        save_model_to_dir = f'{project_dir}/experiments/{train_params["name_your_model"]}/{t}'
  
        early_stop_callback = EarlyStopping(monitor="validation", \
                                            min_delta=0.00, \
                                            patience=train_params['patience'], \
                                            verbose=True, \
                                            mode=train_params['mode'])
        
        checkpoint_callback = ModelCheckpoint(save_top_k=1, \
                                              verbose=True, \
                                              dirpath=save_model_to_dir,\
                                              #filename=f"best_model",
                                              monitor='validation', \
                                              mode=train_params['mode'],)

        trainer = Trainer(detect_anomaly= True, \
                          deterministic='warn', \
                          callbacks=[checkpoint_callback, \
                          early_stop_callback], \
                          max_epochs=train_params['max_epochs'], \
                          devices=1, \
                          accelerator=train_params['device'], \
                          default_root_dir=save_model_to_dir)
        
        model = build_model(model_params)

        trainer.fit(model, train_dataloader, valid_dataloader)     # if set ckpt path here, it will load the previous training params.
      
            
def argument_parser():
    parser = argparse.ArgumentParser(description='Run experiments')
    parser.add_argument('--data_type', type=str, default='toy', help='data type')
    parser.add_argument('--num_feat', type=int, default=10, help='number of features')

    parser.add_argument('--hidden_dim', type=int, default=64, help='hidden dimension')
    parser.add_argument('--num_layers', type=int, default=3, help='number of layers')
    
    parser.add_argument('--add_step', type=int, default=0, option=[0, 1], help='add step index as part of input in no_leaking_selector.')
    parser.add_argument('--max_budget', type=int, default=6, help='selection budget, between 1 to 10.')
    parser.add_argument('--step_size', type=int, default=1, help='step size in no_leaking_selector.')
    parser.add_argument('--lamda', type=float, default=0.5, help='sparsity weight')
    
    parser.add_argument('--device', type=str, default='gpu', help='cpu or gpu')
    parser.add_argument('--tries', type=int, default=5, help='number of tries for training the same model.')
    parser.add_argument('--max_epochs', type=int, default=1000, help='max epochs for training.')
    parser.add_argument('--patience', type=int, default=100, help='patience for early stopping.')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = argument_parser()
    args = vars(args)
    main(args)


            
            
    
    
