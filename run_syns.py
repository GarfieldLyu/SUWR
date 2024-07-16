import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from suwr import NonLeakingSelector
from utils import update_arguments, load_synthetic_data, evaluate_synthetic
import pathlib 
import argparse
import json


project_dir = "set your own project directory here"


def build_model(kwargs: dict):
    """
    Init model for training.
    kwargs: dict, hyperparameters for model.
    """
    model = NonLeakingSelector(kwargs).double()
    return model

def main(args: dict):
    """ categorical classification"""

    # update model and training parameters.
    model_params, train_params = update_arguments(args)

    # load synthetic datasets.
    X_train, Y_train, _ = load_synthetic_data(args['data_type'], \
                                              num_samples=args['num_samples'], \
                                              num_feat=args['num_feat'], \
                                              seed=args['train_seed'])
    X_test, Y_test, gt_test = load_synthetic_data(args['data_type'], \
                                                num_samples=args['num_samples'], \
                                                num_feat=args['num_feat'], \
                                                seed=args['test_seed'])
    
    tpr_mean, tpr_std, fdr_mean, fdr_std = [], [], [], []
    auc, apr, acc = [], [], []
    for t in range(1, train_params['tries']):
        seed_everything(t, workers=True)
        
        # split training data into training and validation.
        train_idxes = int(X_train.shape[0] * 0.8)
        train_dataset = TensorDataset(torch.tensor(X_train[: train_idxes]), torch.tensor(Y_train[: train_idxes]))
        valid_dataset = TensorDataset(torch.tensor(X_train[train_idxes: ]), torch.tensor(Y_train[train_idxes: ]))
        
        train_dataloader = DataLoader(train_dataset, batch_size=1000, num_workers=8)
        valid_dataloader = DataLoader(valid_dataset, batch_size=1000, num_workers=8, shuffle=False)

        """===================== evaluate a saved checkpoint only ======================"""
        if train_params['eval_only']:
            ckpt_resume = train_params['ckpt_eval']
            print(f'Evaluate a saved checkpoint {ckpt_resume}\n')
            model = NonLeakingSelector.load_from_checkpoint(ckpt_resume).double()
            model.eval()
            save_model_to_dir = pathlib.Path(ckpt_resume).parent

        else:
            """===================== train and evaluate ======================"""
            save_model_to_dir = f'{project_dir}/experiments/{train_params["name_your_model"]}/{t}'

            early_stop_callback = EarlyStopping(monitor="validation", \
                                                min_delta=0.00, \
                                                patience=train_params["patience"], \
                                                verbose=True, \
                                                mode=train_params["mode"])
        
            checkpoint_callback = ModelCheckpoint(save_top_k=5, \
                                                  verbose=True, \
                                                  dirpath=save_model_to_dir, \
                                                  monitor='validation', \
                                                  mode=train_params["mode"])

            trainer = Trainer(detect_anomaly= True, \
                              deterministic='warn', \
                              callbacks=[checkpoint_callback, \
                              early_stop_callback], \
                              max_epochs=train_params['max_epochs'], \
                              devices=1, \
                              accelerator=train_params['device'],
                              default_root_dir=save_model_to_dir)
        
            if train_params['resume']:
                ckpt_resume = train_params['ckpt_eval']
                print(f'Continue training from {ckpt_resume}\n\n')
                model_params['exploration'] = 0.05
                model_params['exploration_stop'] = 0.1
                model_params['exploration_epochs'] = 1000
                model = build_model(model_params)
                model.load_state_dict(torch.load(ckpt_resume)['state_dict'])              
            else: 
                ckpt_resume = None  # training from scratch.
                model = build_model(model_params)
            
            trainer.fit(model, train_dataloader, valid_dataloader)     # if set ckpt path here, it will load the previous training params.
        
            # ================load the best model and evaluate on test data================.
            print(f"Loading best model from {checkpoint_callback.best_model_path}...\n")
            model = model.load_from_checkpoint(checkpoint_callback.best_model_path).double()
            model.eval()
        
        """================================== evaluate on test data=================================="""
      
        X_test_tensor = torch.from_numpy(X_test).to(model.device)
        print(f'X_test shape: {X_test_tensor.shape}')
        print(f'Y_test shape: {Y_test.shape}')
        Out_test = model.predict(X_test)   # (label, stop, mask) 
        pred_test = torch.softmax(Out_test[0].cpu(), -1).numpy()
        feat_test = Out_test[2].cpu().numpy()
        test_results = evaluate_synthetic(Y_test, gt_test, pred_test, feat_test)

        with open(f'{save_model_to_dir}/test_results.json', 'w') as f:
            json.dump(test_results, f)
        
        print(f'Test results: {test_results}')
        print(f'Saved test results to {save_model_to_dir}.')
        
        tpr_mean.append(test_results['tpr_mean'])   
        tpr_std.append(test_results['tpr_std'])
        fdr_mean.append(test_results['fdr_mean'])
        fdr_std.append(test_results['fdr_std'])
        auc.append(test_results['auc'])
        apr.append(test_results['apr'])
        acc.append(test_results['acc'])
    
    # ================print the average results over multiple tries ================.
    print(f'tpr_mean: {np.round(np.mean(tpr_mean), 4)},  \
            tpr_std: {np.round(np.mean(tpr_std), 4)},    \
            fpr_mean: {np.round(np.mean(fdr_mean), 4)},  \
            fpr_std: {np.round(np.mean(fdr_std), 4)},    \
            auc: {np.round(np.mean(auc), 4)},            \
            apr: {np.round(np.mean(apr), 4)},            \
            acc: {np.round(np.mean(acc), 4)}')           
    

def argument_parser():
    parser = argparse.ArgumentParser(description='Run experiments')
    parser.add_argument('--data_type', type=str, help='syn1, syn2, syn3, syn4, syn5, syn6')
    parser.add_argument('--num_samples', type=int, default=10000, help='number of samples')
    parser.add_argument('--num_feat', type=int, default=11, help='number of features')
    parser.add_argument('--train_seed', type=int, default=0, help='random seed for training data')
    parser.add_argument('--test_seed', type=int, default=100, help='random seed for testing data')

    parser.add_argument('--model_type', type=str, default='simple', help='simple or selector_predictor.')
    parser.add_argument('--hidden_dim', type=int, default=100, help='hidden dimension')
    parser.add_argument('--num_layers', type=int, default=3, help='number of layers')
    
    parser.add_argument('--add_step', type=int, default=0, option=[0, 1], help='add step index as part of input in no_leaking_selector.')
    parser.add_argument('--max_budget', type=int, help='selection budget, between 1 to 10.')
    parser.add_argument('--step_size', type=int, default=1, help='step size in no_leaking_selector.')
    parser.add_argument('--lamda', type=float, help='sparsity weight')
    parser.add_argument('--exploration_epochs', type=int, default=50, help='exploration epochs for no_leaking_simple.')

    parser.add_argument('--name_your_model', type=str,  help='name your model.')
    parser.add_argument('--device', type=str, default='gpu', help='cpu or gpu')
    parser.add_argument('--tries', type=int, default=5, help='number of tries for training the same model.')
    parser.add_argument('--max_epochs', type=int, default=1000, help='max epochs for training.')
    parser.add_argument('--patience', type=int, default=100, help='patience for early stopping.')
    parser.add_argument('--resume', action='store_true', help='resume training or not.')
    parser.add_argument('--eval_only', action='store_true', help='only evaluate the model.')
    parser.add_argument('--ckpt_eval', type=str, default=None, help='checkpoint for evaluation.')
    
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = argument_parser()
    args = vars(args)
    main(args)


            
            
    
    
