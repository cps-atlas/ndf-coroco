import torch
import torch.nn as nn

from torch.nn.utils import clip_grad_norm_
import numpy as np


import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

'''
if no GPU
'''
# jax.config.update('jax_platform_name', 'cpu')


'''
torch training without eikonal constraint
'''

def train_3d(net, dataloader, val_dataloader, num_epochs, learning_rate, device, loss_threshold=1e-4):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=1e-5)
    

    net.to(device)

    for epoch in range(num_epochs):
        net.train()
        running_loss = 0.0

        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = net(inputs).squeeze()

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
                val_outputs = net(val_inputs).squeeze()
                val_loss += criterion(val_outputs, val_targets).item()

            val_loss /= len(val_dataloader)
            print(f"Validation Loss: {val_loss:.4f}")

        # Check if both training loss and validation loss are smaller than the threshold
        if epoch_loss < loss_threshold and val_loss < loss_threshold:
            print(f"Training stopped early at epoch {epoch+1} as both losses are below the threshold.")
            break

    return net, epoch_loss


'''
training with eikonal constraint
'''

def train_with_eikonal_3d(net, dataloader, val_dataloader, num_epochs, learning_rate, device, log_file='training_log.txt', loss_threshold=1e-4, lambda_eikonal=0.02):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=1e-5)
    net.to(device)

    # Open the log file
    with open(log_file, 'w') as f:
        f.write("Starting training...\n")

        for epoch in range(num_epochs):
            net.train()
            running_loss = 0.0
            running_eikonal_loss = 0.0

            for inputs, targets in dataloader:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = net(inputs).squeeze()

                # Distance loss
                distance_loss = criterion(outputs, targets)

                # Eikonal loss
                eikonal_loss = 0.0
                workspace_pt_grad = torch.autograd.grad(outputs, inputs, grad_outputs=torch.ones_like(outputs),
                                                        create_graph=True, allow_unused=True)[0][:, -3:]
                eikonal_loss += torch.mean((torch.norm(workspace_pt_grad, dim=1) - 1.0) ** 2)

                # Total loss
                loss = distance_loss + lambda_eikonal * eikonal_loss

                loss.backward()
                optimizer.step()
                clip_grad_norm_(net.parameters(), max_norm=1.0)  # Gradient clipping

                running_loss += distance_loss.item()
                running_eikonal_loss += eikonal_loss.item()

            epoch_distance_loss = running_loss / len(dataloader)
            epoch_eikonal_loss = running_eikonal_loss / len(dataloader)

            if (epoch + 1) % 20 == 0:
                log_msg = (f"Epoch [{epoch+1}/{num_epochs}], Distance Loss: {epoch_distance_loss:.4f}, "
                           f"Eikonal Loss: {epoch_eikonal_loss:.4f}")
                print(log_msg)
                f.write(log_msg + '\n')
                
                # Evaluate the model on the validation set
                net.eval()
                with torch.no_grad():
                    predicted_distances = []
                    true_distances = []
                    for val_inputs, val_targets in val_dataloader:
                        val_inputs, val_targets = val_inputs.to(device), val_targets.to(device)
                        val_outputs = net(val_inputs).squeeze()
                        
                        # Store the predicted and true distances
                        predicted_distances.extend(val_outputs.cpu().numpy())
                        true_distances.extend(val_targets.cpu().numpy())
                    
                    # Calculate the absolute differences between predicted and true distances
                    differences = np.array(predicted_distances) - np.array(true_distances)
                    
                    # Calculate MAE/MAD and RMSE
                    mae = np.mean(np.abs(differences))
                    rmse = np.sqrt(np.mean(differences**2))
                    
                    # Print MAE/MAD and RMSE
                    val_log_msg = (f"Validation MAE/MAD: {mae:.4f}\n"
                                   f"Validation RMSE: {rmse:.4f}")
                    print(val_log_msg)
                    f.write(val_log_msg + '\n')
                
                # Check if both training loss and validation loss are smaller than the threshold
                if epoch_distance_loss < loss_threshold and mae < loss_threshold:
                    early_stop_msg = (f"Training stopped early at epoch {epoch+1} as both losses are below the threshold.")
                    print(early_stop_msg)
                    f.write(early_stop_msg + '\n')
                    break

    return net, epoch_distance_loss



'''
training with eikonal constraint + encourage no distance under estimation 
'''

def train_eikonal_moe(net, dataloader, val_dataloader, num_epochs, learning_rate, device, log_file='training_log.txt', loss_threshold=1e-4, lambda_eikonal=0.02, lambda_moe=2.0):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=1e-5)
    net.to(device)

    # Open the log file
    with open(log_file, 'w') as f:
        f.write("Starting training...\n")

        for epoch in range(num_epochs):
            net.train()
            running_loss = 0.0
            running_eikonal_loss = 0.0
            running_moe_loss = 0.0

            for inputs, targets in dataloader:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = net(inputs).squeeze()

                # Distance loss
                distance_loss = criterion(outputs, targets)

                # Eikonal loss
                workspace_pt_grad = torch.autograd.grad(outputs, inputs, grad_outputs=torch.ones_like(outputs),
                                                        create_graph=True, allow_unused=True)[0][:, -3:]
                eikonal_loss = torch.mean((torch.norm(workspace_pt_grad, dim=1) - 1.0) ** 2)

                # Mean Overestimation Error (MOE) loss
                overestimation_error = torch.relu(outputs - targets)
                moe_loss = torch.mean(overestimation_error ** 2)

                # Total loss
                loss = distance_loss + lambda_eikonal * eikonal_loss + lambda_moe * moe_loss

                loss.backward()
                optimizer.step()
                clip_grad_norm_(net.parameters(), max_norm=1.0)  # Gradient clipping

                running_loss += distance_loss.item()
                running_eikonal_loss += eikonal_loss.item()
                running_moe_loss += moe_loss.item()

            epoch_distance_loss = running_loss / len(dataloader)
            epoch_eikonal_loss = running_eikonal_loss / len(dataloader)
            epoch_moe_loss = running_moe_loss / len(dataloader)

            if (epoch + 1) % 10 == 0:
                log_msg = (f"Epoch [{epoch+1}/{num_epochs}], Distance Loss: {epoch_distance_loss:.4f}, "
                           f"Eikonal Loss: {epoch_eikonal_loss:.4f}, MOE Loss: {epoch_moe_loss:.4f}")
                print(log_msg)
                f.write(log_msg + '\n')
                
                # Evaluate the model on the validation set
                net.eval()
                with torch.no_grad():
                    predicted_distances = []
                    true_distances = []

                    for val_inputs, val_targets in val_dataloader:
                        val_inputs, val_targets = val_inputs.to(device), val_targets.to(device)
                        val_outputs = net(val_inputs).squeeze()
                        
                        # Store the predicted and true distances
                        predicted_distances.extend(val_outputs.cpu().numpy())
                        true_distances.extend(val_targets.cpu().numpy())

                    # Calculate the differences between predicted and true distances
                    differences = np.array(predicted_distances) - np.array(true_distances)
                    
                    # Calculate MAE/MAD and RMSE
                    mae = np.mean(np.abs(differences))
                    rmse = np.sqrt(np.mean(differences**2))

                    moe = np.mean(np.maximum(0, differences))
                    
                    # Print MAE/MAD, RMSE, and MOE
                    val_log_msg = (f"Validation MAE/MAD: {mae:.4f}\n"
                                f"Validation RMSE: {rmse:.4f}\n"
                                f"Validation MOE: {moe:.4f}")
                    print(val_log_msg)
                    f.write(val_log_msg + '\n')
                
                # Check if both training loss and validation loss are smaller than the threshold
                if epoch_distance_loss < loss_threshold and mae < loss_threshold:
                    early_stop_msg = (f"Training stopped early at epoch {epoch+1} as both losses are below the threshold.")
                    print(early_stop_msg)
                    f.write(early_stop_msg + '\n')
                    break

    return net, epoch_distance_loss