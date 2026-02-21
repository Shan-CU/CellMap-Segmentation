import socket
import sys
sys.path.insert(0, '.')

hostname = socket.gethostname().lower()
print(f'Hostname: {hostname}')
print(f'Contains rocinante: {"rocinante" in hostname}')

from config_rocinante import N_GPUS, NUM_WORKERS, MODEL_CONFIGS, DATALOADER_CONFIG
print(f'\nConfig values for Rocinante:')
print(f'  N_GPUS: {N_GPUS}')
print(f'  NUM_WORKERS: {NUM_WORKERS}')
print(f'  Batch size (unet_2d): {MODEL_CONFIGS["unet_2d"]["batch_size"]}')
print(f'  DATALOADER_CONFIG: {DATALOADER_CONFIG}')
