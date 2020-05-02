import torch
import models

source_config = 'custom/yolov3-spp-custom.cfg'
target_config = 'custom/yolov3-spp-lstm3.cfg'
source_weights = 'weights/pretrained-converted-28.pt'
target_weights = 'weights/pretrained-lstm3.pt'

source_model = models.Darknet(source_config)
target_model = models.Darknet(target_config)
chkpt = torch.load(source_weights)
source_model.load_state_dict(chkpt['model'])

# use LSTM layer from target, everything else from source
source_model.module_list[88] = target_model.module_list[88]
source_model.module_list[101] = target_model.module_list[101]
source_model.module_list[114] = target_model.module_list[114]

# save new weights
target_model.module_list = source_model.module_list
chkpt['model'] = target_model.state_dict()
torch.save(chkpt, target_weights)

# test
model = models.Darknet(target_config)
chkpt = torch.load(target_weights)
model.load_state_dict(chkpt['model'])
