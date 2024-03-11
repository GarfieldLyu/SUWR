import json
import torch
from torch.utils.data import TensorDataset, DataLoader
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from suwr import NonLeakingSelector
from utils import *
import pathlib
import argparse
import pickle


#torch.set_float32_matmul_precision('medium' | 'high')

project_dir = '/home/lyu/featureselection'


        

def build_model(kwargs: dict):
    """
    Init model for training.
        kwargs: dict, hyperparameters for model.
    """
    model = NonLeakingSelector(kwargs).double()
    return model


def main_mnist(data_type: str, model_params: dict, train_params: dict):
    """ categorical classification"""

    #load data first
    (X_train, Y_train, X_test, Y_test) = load_image_data(data_type)
    train_idxes = int(X_train.shape[0] * 0.8)            # 80% for training, 20% for validation.
    model_params.update({'input_dim': X_train.shape[1]}) # update input dimension for the model based on the data.
    Auc, Apr, Acc = [], [], []

    for t in range(train_params['tries']):    # train multiple times, simply take 0 - tries-1 as random seeds.
        seed_everything(t, workers=True)
        train_dataset = TensorDataset(torch.tensor(X_train[: train_idxes]), torch.tensor(Y_train[: train_idxes], dtype=torch.long))
        valid_dataset = TensorDataset(torch.tensor(X_train[train_idxes:]), torch.tensor(Y_train[train_idxes:], dtype=torch.long))
        train_dataloader = DataLoader(train_dataset, batch_size=train_params['batch_size'], num_workers=8, shuffle=True)
        valid_dataloader = DataLoader(valid_dataset, batch_size=train_params['batch_size'], num_workers=8)
        
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

            early_stop_callback = EarlyStopping(monitor="validation", min_delta=0.00, 
                                                patience=train_params["patience"], verbose=True, mode=train_params["mode"])
        
            checkpoint_callback = ModelCheckpoint(save_top_k=3, verbose=True, 
                                                  dirpath=save_model_to_dir,
                                                  filename='model-{epoch:02d}-{validation:.2f}',
                                                  monitor='validation', mode=train_params["mode"])

            # Init trainer. deiveces=1, accelerator='gpu', use single gpu'.
            trainer = Trainer(detect_anomaly=True, deterministic='warn',
                              callbacks=[checkpoint_callback, early_stop_callback], 
                              max_epochs=train_params['max_epochs'], devices=1, accelerator=train_params['device'],
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
                ckpt_resume = None
                model = build_model(model_params)

            trainer.fit(model, train_dataloader, valid_dataloader)

            # load the best model and evaluate on test data
            print(f"Loading best model from {checkpoint_callback.best_model_path}...\n")
            model = model.load_from_checkpoint(checkpoint_callback.best_model_path).double()
            model.eval()


        """================================== evaluate on test data=================================="""
      
        X_test_tensor = torch.from_numpy(X_test).to(model.device)
        print(f'X_test shape: {X_test_tensor.shape}')
        print(f'Y_test shape: {Y_test.shape}')
        Out_test = model.predict(X_test_tensor)                    # (predict, stop, mask, step_masks) 
        pred_test = torch.softmax(Out_test[0].cpu(), -1).numpy()   # softmax for classification.
        feat_test = Out_test[2].cpu().numpy()                      # N, feature_dim, a final mask for each sample.
        step_masks = [[step.cpu().numpy() for step in instance ] for instance in Out_test[3]]  # list of mask indices for each step. Keep selection order.
        sparsity = feat_test.sum(-1).mean()                        # Average ratio of selected features.

        # compute auroc, aupr, acc
        (auc, apr, acc) = classification_performance_metric(Y_test, pred_test)
        Auc.append(auc)
        Apr.append(apr)
        Acc.append(acc)
        print(f'Round {t}: auc: {auc}, apr: {apr}, acc: {acc}, sparsity: {sparsity}')

        # save evaluation results.
        with open(f'{save_model_to_dir}/test_results.json', 'w') as f:
            json.dump({'auc': auc, 'apr': apr, 'acc': acc, 'sparsity': sparsity}, f)

        # save test outputs.
        save_results = {'pred': pred_test, 'selection': feat_test, 'step_masks': step_masks}
        with open(f'{save_model_to_dir}/test_outputs.pkl', 'wb') as f:
            pickle.dump(save_results, f)


def main(args: dict):
    """ main function for training and testing."""
    model_params, train_params = update_arguments(args)
    main_mnist(args['data_type'], model_params, train_params)
                    
                    
def argument_parser():
    parser = argparse.ArgumentParser(description='Run experiments')
    parser.add_argument('--data_type', type=str, default='mnist', help='mnist or fashion_mnist.')
    parser.add_argument('--model_type', type=str, default='simple', help='simple or selector_predictor.')
    parser.add_argument('--hidden_dim', type=int, default=200, help='hidden dimension')
    parser.add_argument('--num_layers', type=int, default=8, help='number of layers')
    
    parser.add_argument('--step_size', type=int, default=1, help='step size in no_leaking_selector.')
    parser.add_argument('--lamda', type=float, default=0.5, help='sparsity weight')
    parser.add_argument('--max_budget', type=int, default=100, help='max budget for no_leaking_selector.')
    parser.add_argument('--exploration_epochs', type=int, default=50, help='exploration epochs for no_leaking_selector.')
    parser.add_argument('--select_patch', action='store_true', help='select patch or pixel.')

    parser.add_argument('--name_your_model', type=str,  help='name your model.')
    parser.add_argument('--device', type=str, default='cpu', help='cpu or gpu')
    parser.add_argument('--tries', type=int, default=1, help='number of tries for training the same model.')
    parser.add_argument('--max_epochs', type=int, default=1000, help='max epochs for training.')
    parser.add_argument('--patience', type=int, default=100, help='patience for early stopping.')
    parser.add_argument('--batch_size', type=int, default=1000, help='batch size.')
    parser.add_argument('--resume', action='store_true', help='resume training or not.')
    parser.add_argument('--eval_only', action='store_true', help='only evaluate the model.')
    parser.add_argument('--ckpt_eval', type=str, default=None, help='checkpoint for evaluation.')
    
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = argument_parser()
    args = vars(args)
    main(args)


            
            
    
    
