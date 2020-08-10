import torch.nn as nn
import torch
import torch.nn.functional as F

class DecoderLSTM(nn.Module):
    def __init__(self, hidden_size, output_size, dropout, num_layers=1, bidirectional=False):
        super(DecoderLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.embedding = nn.Embedding(output_size, self.hidden_size)
        self.num_layers = num_layers
        self.lstm = nn.LSTM(self.hidden_size, self.hidden_size, num_layers=num_layers, bidirectional=self.bidirectional)
        self.out = nn.Linear(self.hidden_size * (1 + self.bidirectional), output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden, cell):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, (hidden, cell) = self.lstm(output, (hidden, cell))
        output = self.softmax(self.out(output[0]))
        return output, hidden, cell

    def initHidden(self, batch=1):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.zeros(self.num_layers * (1 + self.bidirectional), batch, self.hidden_size, device=device)