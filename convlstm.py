import torch
import torch.nn as nn

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

class ConvLSTMUnit(nn.Module):
  def __init__(self, in_chans, hidden_chans, k, layers, bias, dropout, reverse=False):
    super(ConvLSTMUnit, self).__init__()
    self.reverse = reverse
    self.layers = layers
    self.dropout = None if dropout == None else nn.Dropout2d(dropout, inplace=True)
    self.in_chans = in_chans
    self.hidden_chans = hidden_chans
    self.cells = []
    for i in layers: # changed from range(layers)
      if i == 0:
        cell = ConvLSTMCell(in_chans, hidden_chans, k, bias)
      else:
        cell = ConvLSTMCell(hidden_chans, hidden_chans, k, bias)
      self.cells.append(cell)
      self.add_module(f'cell_{i}', cell)

  def order(self, seq_len):
    if self.reverse:
      return reversed(range(seq_len))
    else:
      return range(seq_len)

  def forward(self, x, out, h0, c0):
    for layer in range(self.layers):
      cell = self.cells[layer]
      h, c = h0, c0
      for i in self.order(len(x)):
        h, c = cell(x[i], h, c)
        out[i] = h
      if self.dropout and layer < (self.layers - 1):
        self.dropout(out)
      x = out
    return out

class ConvLSTM(nn.Module):

  def __init__(self, in_chans, hidden_chans, kernel_size, layers=1, bidirectional=False, bias=True, dropout=None):
    super(ConvLSTM, self).__init__()
    self.bidirectional = bidirectional
    self.left_to_right = ConvLSTMUnit(in_chans, hidden_chans, kernel_size, layers, bias, dropout)
    self.in_chans = in_chans
    self.hidden_chans = hidden_chans
    self.out_chans = hidden_chans
    if bidirectional:
      self.out_chans *= 2
      self.right_to_left = ConvLSTMUnit(in_chans, hidden_chans, kernel_size, layers, bias, dropout, reverse=True)

  def forward(self, x):
    seq_len, bs, chans, height, width = tuple(x.size())
    assert(chans == self.in_chans)
    out = torch.empty(seq_len, bs, self.out_chans, height, width, dtype=x.dtype, device=x.device)
    h0 = torch.zeros(bs, self.hidden_chans, height, width, dtype=x.dtype, device=x.device)
    c0 = torch.zeros(bs, self.hidden_chans, height, width, dtype=x.dtype, device=x.device)

    self.left_to_right(x, out[:,:,:self.hidden_chans,:,:], h0, c0)
    if self.bidirectional:
      self.right_to_left(x, out[:,:,self.hidden_chans:,:,:], h0, c0)
    return out
