import numpy as np
import os
import random
from pathlib import Path

TRAIN_SPLIT = 0.8
TASKS_DIR = '../tasks'
OUT_DIR = 'custom'

BUFFER = 10
CONTEXT = 30

clips = []
for task in os.listdir(TASKS_DIR):
  paths = Path(f'{TASKS_DIR}/{task}').rglob('*.jpg')
  paths = list(sorted(paths, key=lambda path: int(path.name.split('.jpg')[0])))

  in_clip = False
  clip_start = None
  buffer = None
  for i in range(len(paths)):
    imgfile = paths[i]
    lblfile = str(imgfile).replace('.jpg', '.txt')
    if os.path.getsize(lblfile) > 0:
      if not in_clip:
        in_clip = True
        clip_start = i
      buffer = 0
    else:
      if in_clip:
        if buffer > BUFFER:
          ctx_start = max(0, clip_start - CONTEXT)
          ctx_end = min(len(paths), i + CONTEXT)
          clips.append(list(map(str, paths[ctx_start:ctx_end])))
          in_clip = False
          clip_start = None
          buffer = None
        else:
          buffer += 1
  if in_clip:
    ctx_start = max(0, clip_start - CONTEXT)
    clips.append(list(map(str(paths[ctx_start:]))))

total_clips = len(clips)
num_train_clips = int(TRAIN_SPLIT * total_clips)
num_valid_clips = total_clips - num_train_clips

train_clip_idxs = random.sample(list(range(len(clips))), num_train_clips)
valid_clip_idxs = []
for i in range(len(clips)):
  if i not in train_clip_idxs:
    valid_clip_idxs.append(i)

data = np.array(list(map(np.array, clips)))
train_data = data[train_clip_idxs]
valid_data = data[valid_clip_idxs]

np.save(f'{OUT_DIR}/train.npy', train_data, allow_pickle=True)
np.save(f'{OUT_DIR}/valid.npy', valid_data, allow_pickle=True)

print(list(map(len, clips)))

