import os
import random
from pathlib import Path

TRAIN_SPLIT = 0.8
BATCH_SIZE = 128
TASKS_DIR = '../../tasks'
OUT_DIR = '.'

train_paths = []
valid_paths = []

for task in os.listdir(TASKS_DIR):
  paths = Path(f'{TASKS_DIR}/{task}').rglob('*.jpg')
  paths = list(sorted(paths, key=lambda path: int(path.name.split('.jpg')[0])))
  num_batches = (len(paths) + BATCH_SIZE - 1) // BATCH_SIZE
  num_train_batches = int(TRAIN_SPLIT * num_batches)
  train_batch_idxs = set(random.sample(range(num_batches), num_train_batches))
  for b in range(num_batches):
    batch = paths[b*BATCH_SIZE:(b+1)*BATCH_SIZE]
    if b in train_batch_idxs:
      train_paths += batch
    else:
      valid_paths += batch

with open(f'{OUT_DIR}/train.txt', 'w') as f:
  for path in train_paths:
    f.write(f'{path}\n')
with open(f'{OUT_DIR}/valid.txt', 'w') as f:
  for path in valid_paths:
    f.write(f'{path}\n')
