import torch.nn as nn


class SimpleNet(nn.module):
    # Trainable linear model, shares representation for label prediction, stop signal, and selection logits.
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers:int):
        super(SimpleNet, self).__init__()
        """
        input_dim: original input dimension + 1, where the additional dimension is for step t.
        hidden_dim: hidden dimension.
        output_dim: output label dimension. 1 for MSE.
        num_layers: number of hidden layers.
        """
        self.input_dim, self.hidden_dim, self.output_dim, self.num_layers = input_dim, hidden_dim, output_dim, num_layers
        layers = []
        layers.append(nn.Linear(self.input_dim, self.hidden_dim))
        layers.append(nn.LayerNorm(self.hidden_dim))
        layers.append(nn.Tanh())  # add relu, following invase.
         
        for l in range(self.num_layers-2):
            layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            layers.append(nn.Tanh())
            layers.append(nn.Dropout(0.2))
        self.layers = nn.Sequential(*layers)

        """ Init layers for prediction, stop signal, and selection logits."""
        
        self.layer_stop = nn.Linear(self.hidden_dim, 1)  # one layer for stop logits.
        #self.layer_out = nn.Linear(self.hidden_dim, self.output_dim)   # one layer for label prediction.
        self.layer_out = nn.Sequential(*[nn.Linear(self.hidden_dim, self.hidden_dim), \
                                        nn.Tanh(), \
                                        nn.Linear(self.hidden_dim, self.output_dim)])   # complex layers for label prediction, for hard task.
        self.layer_select = nn.Linear(self.hidden_dim, self.input_dim - 1)  # input_dim - 1 because one dimension is for step t. 
        
    def forward(self, Input):   
        hidden = self.layers(Input)            # hidden representation.
        selection = self.layer_select(hidden)  # selection logits.
        pred = self.layer_out(hidden)          # prediction logits.
        stop_signal = self.layer_stop(hidden)  # stop logits.
       
        return selection, pred, stop_signal
    

class SelectorPredictor(nn.module):
    # Trainable linear models, individual representation for label prediction, stop signal, and selection logits.
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers:int):  
        super(SelectorPredictor, self).__init__()
        """================Init selector================="""
        selector_layers = []
        selector_layers.append(nn.BatchNorm1d(input_dim, momentum=0.01)) # add a batch normalization layer.
        selector_layers.append(nn.Linear(input_dim, hidden_dim))
        selector_layers.append(nn.Tanh())  # add activation. 
        for l in range(num_layers-2):
            selector_layers.append(nn.Linear(hidden_dim, hidden_dim))
            selector_layers.append(nn.Tanh())
            #selector_layers.append(nn.Dropout(0.2))
        selector_layers.append(nn.Linear(hidden_dim, input_dim - 1))  # input_dim  - 1 because one dimension is for step t.
        self.selector = nn.Sequential(*selector_layers)

        """================Init predictor================="""
        predictor_layers = []
        predictor_layers.append(nn.BatchNorm1d(input_dim, momentum=0.01)) # add a batch normalization layer.
        predictor_layers.append(nn.Linear(input_dim, hidden_dim))
        predictor_layers.append(nn.Tanh())  # add activation.

        for l in range(num_layers-2):
            predictor_layers.append(nn.Linear(hidden_dim, hidden_dim))
            predictor_layers.append(nn.Tanh())
            #predictor_layers.append(nn.Dropout(0.2))
        predictor_layers.append(nn.Linear(hidden_dim, output_dim))  # matrix for output
        self.predictor = nn.Sequential(*predictor_layers)

        """================Init stop signal================="""
        stop_layers = []
        stop_layers.append(nn.BatchNorm1d(input_dim, momentum=0.01)) # add a batch normalization layer.
        stop_layers.append(nn.Linear(input_dim, hidden_dim))
        stop_layers.append(nn.Tanh())  # add activation.
        stop_layers.append(nn.Linear(hidden_dim, hidden_dim))  # matrix for stop signal.
        stop_layers.append(nn.Tanh())  # add activation.
        stop_layers.append(nn.Linear(hidden_dim, 1))  # matrix for stop signal.
        self.stop_model = nn.Sequential(*stop_layers)

    def forward(self, Input):
        selection = self.selector(Input)
        pred = self.predictor(Input)
        stop_signal = self.stop_model(Input)
        return selection, pred, stop_signal
