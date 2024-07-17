import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, batch_first=False, dropout=0.0):
        super(CustomGRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.dropout_prob = dropout

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer_input_size = input_size if i == 0 else hidden_size
            self.layers.append(GRULayer(layer_input_size, hidden_size, bias))

        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def forward(self, x, h_0=None):
        if self.batch_first:
            x = x.transpose(0, 1)  # Convert batch_first to seq_first

        seq_len, batch_size, _ = x.size()
        if h_0 is None:
            h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device)

        output_seq = []
        h_n = []

        for layer in range(self.num_layers):
            h_t = h_0[layer]
            layer_output = []

            for t in range(seq_len):
                h_t = self.layers[layer](x[t], h_t)
                layer_output.append(h_t.unsqueeze(0))

                if self.dropout and layer < self.num_layers - 1:
                    h_t = self.dropout(h_t)

            x = torch.cat(layer_output, 0)
            output_seq.append(x)
            h_n.append(h_t.unsqueeze(0))

        output_seq = output_seq[-1]  # Take output from the last layer
        h_n = torch.cat(h_n, 0)

        if self.batch_first:
            output_seq = output_seq.transpose(0, 1)  # Convert back to batch_first

        return output_seq, h_n


class GRULayer(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super(GRULayer, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Reset gate parameters
        self.W_ir = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.W_hr = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_ir = nn.Parameter(torch.Tensor(hidden_size)) if bias else None
        self.b_hr = nn.Parameter(torch.Tensor(hidden_size)) if bias else None

        # Update gate parameters
        self.W_iz = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.W_hz = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_iz = nn.Parameter(torch.Tensor(hidden_size)) if bias else None
        self.b_hz = nn.Parameter(torch.Tensor(hidden_size)) if bias else None

        # New memory gate parameters
        self.W_in = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.W_hn = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_in = nn.Parameter(torch.Tensor(hidden_size)) if bias else None
        self.b_hn = nn.Parameter(torch.Tensor(hidden_size)) if bias else None

        self.init_weights()

    def init_weights(self):
        for param in self.parameters():
            if param.dim() > 1:
                nn.init.kaiming_uniform_(param)
            else:
                nn.init.zeros_(param)

    def forward(self, x, h_prev):
        # Reset gate
        r_t = torch.sigmoid(
            F.linear(x, self.W_ir, self.b_ir) +
            F.linear(h_prev, self.W_hr, self.b_hr)
        )

        # Update gate
        z_t = torch.sigmoid(
            F.linear(x, self.W_iz, self.b_iz) +
            F.linear(h_prev, self.W_hz, self.b_hz)
        )

        # New memory content
        n_t = torch.tanh(
            F.linear(x, self.W_in, self.b_in) +
            r_t * F.linear(h_prev, self.W_hn, self.b_hn)
        )

        # Final hidden state
        h_t = (1 - z_t) * n_t + z_t * h_prev

        return h_t