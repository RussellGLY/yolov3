import torch
import convlstm

torch.nn.LSTM()

data = torch.arange(16*128*256*32*32).reshape(16, 128, 256, 32, 32).float().cuda()
lstm = convlstm.ConvLSTM(256, 150, 5, 3, bidirectional=True, dropout=0.5).cuda()
out = lstm(data)
assert(tuple(out.size()) == (16, 128, 512, 32, 32))
print(out.mean())
