import os
import random
from pathlib import Path

VALID_SPLIT = 0.3
TASKS_DIR = '../../tasks'
OUT_DIR = '.'

train_paths = []
valid_paths = []

for task in os.listdir(TASKS_DIR):
  paths = Path(f'{TASKS_DIR}/{task}').rglob('*.jpg')
  paths = list(sorted(paths, key=lambda path: int(path.name.split('.jpg')[0])))
  val = int(VALID_SPLIT * len(paths))
  valid_paths += paths[:val]
  train_paths += paths[val:]

with open(f'{OUT_DIR}/train.txt', 'w') as f:
  for path in train_paths:
    f.write(f'{path}\n')
with open(f'{OUT_DIR}/valid.txt', 'w') as f:
  for path in valid_paths:
    f.write(f'{path}\n')
