class MLP(nn.Module):
  def __init__(self, d_input, d_layers, d_output, activation=nn.Tanh,
               use_dropout=False, dropout_prob=0.5, norm=None, use_cuda=False):
    '''
    :param d_input: int
      dimensionality of input
    :param d_layers: list of ints
      dimensionality of each layer
    :param d_output: int
      dimensionality of output
    :param activation:
      activation function for network
    :param use_dropout: bool
      use dropout during training
    :param dropout_prob: float or list of floats
      probability of dropout (per layer if list)
    :param norm:
      normalization function (e.g. batchnorm) to apply to each layer
    '''
    super(MLP, self).__init__()
    self.use_cuda = torch.cuda.is_available() if use_cuda else False
    self.device = torch.device('cuda') if self.use_cuda else torch.device('cpu') 
    self.d_input = d_input
    self.n_layers = len(d_layers)
    self.d_output = d_output
    self.activation = activation
    self.use_dropout = use_dropout
    self.norm = norm

    # create layers
    layers = []
    d_in = d_input
    d_out = d_layers[0]
    if self.n_layers > 0:
      for i in range(self.n_layers):
        layers.append(nn.Linear(d_in, d_out))
        if self.norm is not None:
          layers.append(self.norm)
        if self.use_dropout:
          layers.append(nn.Dropout(self.dropout_prob))
        layers.append(self.activation())
        if i < self.n_layers-1:
          d_in = d_layers[i]
          d_out = d_layers[i+1]
      layers.append(nn.Linear(d_out, d_output))
    else:
      layers.append(nn.Linear(d_in, d_out))
    self.net = nn.Sequential(*layers)

  # @property
  # def learn_param(self):
  #   return next(self.parameters()).requires_grad

  # def set_learn_param(self, learn):
  #   for p in self.parameters():
  #     p.requires_grad_(learn)

  def set_parameters(self, params):
    idx = 0
    for layer in self._modules:
      self._modules[layer].weight.data = params[idx]
      self._modules[layer].bias.data = params[idx+1]
      idx += 2

  def forward(self, x):
    # print x.shape
    return self.net(x.view(x.shape[0], -1))

  def print_parameters(self):
    for name, param in self.named_parameters():
      if param.requires_grad:
        print name, param.data  