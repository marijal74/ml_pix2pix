# %%
import torch
import torch.nn as nn
import torch.optim as optim
from src.discriminator import Discriminator
from src.generator import UNet

# %%
loss_comparison = nn.BCEWithLogitsLoss() 
L1_loss = nn.L1Loss()

# %%
def train_discriminator_step(discriminator: Discriminator, generator:UNet, inputs, targets, opt):
    opt.zero_grad()

    # real image loss
    output = discriminator(inputs, targets)
    real_loss = loss_comparison(output, True)

    gen_image = generator(inputs).detach()

    # fake image loss
    fake_output = discriminator(inputs, gen_image)
    
    fake_loss = loss_comparison(fake_output, False)

    total_loss = (real_loss + fake_loss)/2

    total_loss.backward()
    
    opt.step()

    return total_loss

# %%
def generator_training_step(discriminator: Discriminator, generator:UNet, inputs, targets, opt, L1_lambda = 100):
          
    opt.zero_grad()
    
    generated_image = generator(inputs)
    
    disc_output = discriminator(inputs, generated_image)
    
    generator_loss = loss_comparison(disc_output, True) + L1_lambda * L1_loss(generated_image, targets)
    generator_loss.backward()
    opt.step()

    return generator_loss, generated_image

# %%
def get_optimizer(parameters):
    lr=0.0002 
    beta1=0.5
    beta2=0.999
    return optim.Adam(parameters, lr=lr, betas=(beta1, beta2))


