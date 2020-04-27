import torch
import torch.nn as nn
import torch.nn.functional as F

# See https://papers.nips.cc/paper/5955-convolutional-lstm-network-a-machine-learning-approach-for-precipitation-nowcasting.pdf
class ConvLSTMCell(nn.Module):

  def __init__(self, in_chans, hidden_chans, k, bias):
    super(ConvLSTMCell, self).__init__()
    pad = k // 2
    def in_conv(): return nn.Conv2d(in_chans, hidden_chans, k, padding=pad, bias=bias)
    def hidden_conv(): return nn.Conv2d(hidden_chans, hidden_chans, k, padding=pad, bias=False)

    self.w_xi = in_conv()
    self.w_hi = hidden_conv()
    self.w_ci = hidden_conv()

    self.w_xf = in_conv()
    self.w_hf = hidden_conv()
    self.w_cf = hidden_conv()

    self.w_xc = in_conv()
    self.w_hc = hidden_conv()

    self.w_xo = in_conv()
    self.w_ho = hidden_conv()
    self.w_co = hidden_conv()

  def forward(self, x, h, c):
    i = (self.w_xi(x) + self.w_hi(h) + self.w_ci(c)).sigmoid()
    f = (self.w_xf(x) + self.w_hf(h) + self.w_cf(c)).sigmoid()
    ct = f*c + i*(self.w_xc(x) + self.w_hc(h)).tanh()
    o = (self.w_xo(x) + self.w_ho(h) + self.w_co(ct)).sigmoid()
    ht = o*ct.tanh()
    return ht, ct

class ConvLSTM(nn.Module):

  def __init__(self, in_chans, hidden_chans, kernel_size, layers=1, bias=True, dropout=None):
    super(ConvLSTM, self).__init__()
    self.layers = layers
    self.in_chans = in_chans
    self.hidden_chans = hidden_chans
    self.cells = []
    for i in range(layers):
      if i == 0:
        cell = ConvLSTMCell(in_chans, hidden_chans, kernel_size, bias)
      else:
        cell = ConvLSTMCell(hidden_chans, hidden_chans, kernel_size, bias)
      self.cells.append(cell)
      self.add_module(f'lstmcell{i}', cell)

  def forward(self, x, h0c0 = None):
    # verify shape of input
    seq_len, bs, chans, height, width = tuple(x.size())
    assert(chans == self.in_chans)

    # initialize h, c if necessary
    if h0c0:
      h0, c0 = h0c0
    else:
      h0 = torch.zeros(self.layers, bs, self.hidden_chans, height, width, dtype=x.dtype, device=x.device)
      c0 = torch.zeros(self.layers, bs, self.hidden_chans, height, width, dtype=x.dtype, device=x.device)

    # iterate through layers and input sequence
    newh, newc = [], []
    for l in range(self.layers):
      cell = self.cells[l]
      h, c = h0[l], c0[l]
      outs = []
      for i in range(seq_len):
        h, c = cell(x[i], h, c)
        outs.append(h)

      newh.append(h)
      newc.append(c)
      x = torch.stack(outs)

      # apply dropout if configured, but not to last layer
      if self.dropout and l < self.layers - 1:
        x = F.dropout2d(x, self.dropout, training=self.training)

    return x, (torch.stack(newh), torch.stack(newc))
