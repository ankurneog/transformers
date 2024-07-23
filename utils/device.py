"""
Accelerator abstraction 

"""
import torch

class Device:
    def __init__(self,device: str = "cpu"):
        self.device_name=device
        self.create_device_attributes()
        self.handle = getattr(torch,)
    def create_device_attributes(self):
        if self.device is "cuda":
            self.distributed_backend = "nccl"
        elif self.device is "hpu":
            self.distributed_backend = "hccl"
        else:
            self.distributed_backend = "gloo"
            

