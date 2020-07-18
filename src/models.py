import torch
import torch.nn.functional as F

from tools import WeightNorm, DotProduct

### Linear regression model
class Net_linear(torch.nn.Module):
    def __init__(self, n_feature, n_hidden1, n_output):
        super(Net_linear, self).__init__()
        self.hidden0 = DotProduct(n_feature) # Selection Layer
        self.hidden1 = torch.nn.Linear(n_feature, n_hidden1) # Approximation Layer
        self.out = torch.nn.Linear(n_hidden1, n_output)
    
    def forward(self, x):
        x = self.hidden0(x)
        x = F.relu(self.hidden1(x))
        x = self.out(x)
        return x


### Nonlinear classification model
class Net_nonlinear(torch.nn.Module):
	def __init__(self, n_feature, n_hidden1, n_hidden2, n_output):
		super(Net_nonlinear, self).__init__()
		self.hidden0 = DotProduct(n_feature) # Selection Layer
		self.hidden1 = torch.nn.Linear(n_feature, n_hidden1) # Hidden layer 1
		self.hidden2 = torch.nn.Linear(n_hidden1, n_hidden2) # Hidden layer 2
		self.out = torch.nn.Linear(n_hidden2, n_output)
	
	def forward(self, x):
		x = self.hidden0(x)
		x = F.relu(self.hidden1(x))
		x = F.relu(self.hidden2(x))
		x = self.out(x)
		return x
