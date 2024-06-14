import torch
import torch.nn as nn

from torch.nn.utils import clip_grad_norm_

import numpy as np
import jax
import jax.numpy as jnp
from flax import linen as jnn
from flax.training import train_state
import optax

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from training.config import *

'''
if no GPU
'''
jax.config.update('jax_platform_name', 'cpu')



'''
JAX training without eikonal constraint
'''



def train_jax(net, dataset, num_epochs, learning_rate, batch_size, loss_threshold=1e-4):
    def loss_fn(params, batch):
        inputs, distances = batch
        pred_distances = net.apply(params, inputs)
        loss = jnp.mean((pred_distances - distances) ** 2)
        return loss
    
    @jax.jit
    def train_step(state, batch):
        loss, grads = jax.value_and_grad(loss_fn)(state.params, batch)
        state = state.apply_gradients(grads=grads)
        return state, loss

    # Optimizer
    tx = optax.adam(learning_rate=learning_rate)



    state = train_state.TrainState.create(
        apply_fn=net.apply, params=net.init(jax.random.PRNGKey(0), jnp.zeros((1, INPUT_SIZE))), tx=tx
    )

    num_batches = len(dataset) // batch_size
    for epoch in range(num_epochs):

        epoch_loss = 0.0
        
        for i in range(num_batches):
            batch = dataset[i * batch_size: (i + 1) * batch_size]
            
            state, loss = train_step(state, batch)
            epoch_loss += loss
        
        epoch_loss /= num_batches

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")
        
        if epoch_loss < loss_threshold:
            print(f"Reached loss threshold of {loss_threshold}. Stopping training.")
            break
    
    return state.params


'''
torch training without eikonal constraint
'''

def train(net, dataloader, val_dataloader, num_epochs, learning_rate, device, loss_threshold=1e-4):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=1e-5)
    

    net.to(device)

    for epoch in range(num_epochs):
        net.train()
        running_loss = 0.0

        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = torch.mean((outputs - targets)**2)
            loss.backward()


            optimizer.step()
            #clip_grad_norm_(net.parameters(), max_norm=1.0)  # Gradient clipping

            running_loss += loss.item()

        epoch_loss = running_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {epoch_loss:.4f}")

        # Evaluate the model on the validation set
        net.eval()
        with torch.no_grad():
            val_loss = 0.0
            for val_inputs, val_targets in val_dataloader:
                val_inputs, val_targets = val_inputs.to(device), val_targets.to(device)
                val_outputs = net(val_inputs)
                val_loss += criterion(val_outputs, val_targets).item()

            val_loss /= len(val_dataloader)
            print(f"Validation Loss: {val_loss:.4f}")

        # Check if both training loss and validation loss are smaller than the threshold
        if epoch_loss < loss_threshold and val_loss < loss_threshold:
            print(f"Training stopped early at epoch {epoch+1} as both losses are below the threshold.")
            break

    return net


'''
training with eikonal constraint
'''

def train_with_eikonal(net, dataloader, val_dataloader, num_epochs, learning_rate, device, loss_threshold=1e-4, lambda_eikonal=0.1):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=1e-5)
    net.to(device)

    for epoch in range(num_epochs):
        net.train()
        running_loss = 0.0
        running_eikonal_loss = 0.0

        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)

            # Distance loss
            distance_loss = criterion(outputs, targets)

            # Eikonal loss (defined for each link)
            eikonal_loss = 0.0


            for i in range(outputs.shape[1]):
                workspace_pt_grad = torch.autograd.grad(outputs[:, i], inputs, grad_outputs=torch.ones_like(outputs[:, i]),
                                                        create_graph=True, allow_unused=True)[0][:, -2:]
                

                eikonal_loss += torch.mean((torch.norm(workspace_pt_grad, dim=1) - 1.0) ** 2)
                #print('eikonal_loss:', eikonal_loss)

            eikonal_loss = eikonal_loss / outputs.shape[1]
            # Total loss
            loss = distance_loss + lambda_eikonal * eikonal_loss

            loss.backward()
            optimizer.step()
            #clip_grad_norm_(net.parameters(), max_norm=1.0)  # Gradient clipping

            running_loss += distance_loss.item()
            running_eikonal_loss += eikonal_loss.item()

        epoch_distance_loss = running_loss / len(dataloader)
        epoch_eikonal_loss = running_eikonal_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Distance Loss: {epoch_distance_loss:.4f}, Eikonal Loss: {epoch_eikonal_loss:.4f}")

        # Evaluate the model on the validation set
        net.eval()
        with torch.no_grad():
            val_loss = 0.0
            for val_inputs, val_targets in val_dataloader:
                val_inputs, val_targets = val_inputs.to(device), val_targets.to(device)
                val_outputs = net(val_inputs)
                val_loss += criterion(val_outputs, val_targets).item()
            val_loss /= len(val_dataloader)
            print(f"Validation Loss: {val_loss:.4f}")

        # Check if both training loss and validation loss are smaller than the threshold
        if epoch_distance_loss < loss_threshold and val_loss < loss_threshold:
            print(f"Training stopped early at epoch {epoch+1} as both losses are below the threshold.")
            break

    return net



'''
train with normal loss
'''

def train_with_normal_loss(net, dataloader, val_dataloader, num_epochs, learning_rate, device, loss_threshold=1e-4, lambda_eikonal=0.1):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=1e-5)
    net.to(device)

    for epoch in range(num_epochs):
        net.train()
        running_loss = 0.0
        running_normal_loss = 0.0
        running_eikonal_loss = 0.0

        for inputs, targets, normals in dataloader:
            inputs, targets, normals = inputs.to(device), targets.to(device), normals.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)

            # Distance loss
            distance_loss = torch.mean(torch.abs(outputs))

            # Normal loss
            gradients = torch.autograd.grad(outputs.sum(), inputs, create_graph=True)[0][:, -2:]
            print('gradients:', gradients)

            print('normals:', normals)

            normal_loss = criterion(gradients, normals)

            eikonal_loss = 0.0

            for i in range(outputs.shape[1]):
                workspace_pt_grad = torch.autograd.grad(outputs[:, i], inputs, grad_outputs=torch.ones_like(outputs[:, i]),
                                                        create_graph=True, allow_unused=True)[0][:, -2:]
                
                
                eikonal_loss += torch.mean((torch.norm(workspace_pt_grad, dim=1) - 1.0) ** 2)
                #print('eikonal_loss:', eikonal_loss)

            eikonal_loss = eikonal_loss / outputs.shape[1]

            # Total loss
            loss = distance_loss + normal_loss + lambda_eikonal * eikonal_loss
            loss.backward()
            optimizer.step()

            running_loss += distance_loss.item()
            running_normal_loss += normal_loss.item()
            running_eikonal_loss += eikonal_loss.item()

        epoch_distance_loss = running_loss / len(dataloader)
        epoch_normal_loss = running_normal_loss / len(dataloader)
        epoch_eikonal_loss = running_eikonal_loss / len(dataloader)
        epoch_loss = epoch_distance_loss + epoch_normal_loss + lambda_eikonal * epoch_eikonal_loss

        print(f"Epoch [{epoch+1}/{num_epochs}], Distance Loss: {epoch_distance_loss:.4f}, Normal Loss: {epoch_normal_loss:.4f}, Eikonal Loss: {epoch_eikonal_loss:.4f}")

        # Evaluate the model on the validation set
        net.eval()
        with torch.no_grad():
            val_loss = 0.0
            for val_inputs, val_targets, val_normals in val_dataloader:
                val_inputs, val_targets = val_inputs.to(device), val_targets.to(device)
                val_outputs = net(val_inputs)
                val_loss += torch.mean(torch.abs(val_outputs)).item()
            val_loss /= len(val_dataloader)
            print(f"Validation Loss: {val_loss:.4f}")

        # Check if both training loss and validation loss are smaller than the threshold
        if epoch_loss < loss_threshold and val_loss < loss_threshold:
            print(f"Training stopped early at epoch {epoch+1} as both losses are below the threshold.")
            break

    return net