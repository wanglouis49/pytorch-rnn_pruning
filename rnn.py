import torch
import torch.nn as nn
from torch.autograd import Variable


def to_var(x):
	if torch.cuda.is_available():
		x = x.cuda()
	return Variable(x)
	

def compute_accuracy(rnn, sequence_length, input_size, data_loader):
	# Test the Model
	correct = 0; total = 0
	for images, labels in data_loader:
		images = to_var(images.view(-1, sequence_length, input_size))
		outputs = rnn(images)
		_, predicted = torch.max(outputs.data, 1)
		total += labels.size(0)
		correct += (predicted.cpu() == labels).sum()
	accuracy = 100. * float(correct) / total
	return accuracy


class RNN(nn.Module):
	def __init__(self, input_size, hidden_size, num_layers, num_classes):
		super(RNN, self).__init__()
		self.hidden_size = hidden_size
		self.num_layers = num_layers
		self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
		self.fc = nn.Linear(hidden_size, num_classes)

	def forward(self, x):
		# Set inital states (num_layers, batch, hidden_size)
		h0 = to_var(torch.zeros(self.num_layers, x.size(0), self.hidden_size))

		# Forward propagate RNN (input, h_0 -> output, h_n)
		out, _ = self.rnn(x, h0)

		# Decode hidden state of last time step
		out = self.fc(out[:, -1, :])
		return out


class LSTM(nn.Module):
	def __init__(self, input_size, hidden_size, num_layers, num_classes):
		super(LSTM, self).__init__()
		self.hidden_size = hidden_size
		self.num_layers = num_layers
		self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
		self.fc = nn.Linear(hidden_size, num_classes)

	def forward(self, x):
		# Set initial states (num_layers, batch, hidden_size)
		h0 = to_var(torch.zeros(self.num_layers, x.size(0), self.hidden_size))
		c0 = to_var(torch.zeros(self.num_layers, x.size(0), self.hidden_size))

		# Forward propagate RNN (input, (h_0, c_0) -> output, (h_n, c_n))
		out, _ = self.lstm(x, (h0, c0))

		# Decode hidden state of last time step
		out = self.fc(out[:, -1, :])
		return out


class GRU(nn.Module):
	def __init__(self, input_size, hidden_size, num_layers, num_classes):
		super(GRU, self).__init__()
		self.hidden_size = hidden_size
		self.num_layers = num_layers
		self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
		self.fc = nn.Linear(hidden_size, num_classes)

	def forward(self, x):
		# Set inital states (num_layers, batch, hidden_size)
		h0 = to_var(torch.zeros(self.num_layers, x.size(0), self.hidden_size))

		# Forward propagate RNN (input, h_0 -> output, h_n)
		out, _ = self.gru(x, h0)

		# Decode hidden state of last time step
		out = self.fc(out[:, -1, :])
		return out



