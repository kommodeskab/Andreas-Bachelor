from diffusers import DDPMScheduler
import torch

scheduler = DDPMScheduler()
scheduler.set_timesteps(10)
print(scheduler.previous_timestep(999))
print(scheduler.previous_timestep(800))