import os

# Set the environment variable to use only GPU 1
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
import GPUtil

torch.cuda.set_device(1)


# Verify the CUDA devices
#print("CUDA_VISIBLE_DEVICES:", os.environ["CUDA_VISIBLE_DEVICES"])
print("Available CUDA devices:", torch.cuda.device_count())
print("Current CUDA device:", torch.cuda.current_device())
print("CUDA device name:", torch.cuda.get_device_name(torch.cuda.current_device()))
print("is avail:", torch.cuda.is_available())
print("memory reserved", torch.cuda.memory_reserved())
print("memory allocated", torch.cuda.memory_allocated())

# Empyy cache
torch.cuda.empty_cache()

GPUtil.showUtilization()

