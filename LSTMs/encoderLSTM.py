import torch.nn as nn
import torch

class EncoderLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, dropout, num_layers=1, bidirectional=False):
        super(EncoderLSTM, self).__init__()
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, bidirectional=bidirectional)

    def forward(self, input, hidden, cell):
        output, (hidden, cell) = self.lstm(input, (hidden, cell))
        return output, hidden, cell

    def initHidden(self, batch=1, num_layers=1):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.zeros(self.num_layers * (1 + self.bidirectional), batch, self.hidden_size, device=device), torch.zeros(self.num_layers * (1 + self.bidirectional), batch, self.hidden_size, device=device) 