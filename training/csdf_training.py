import torch
import torch.nn as nn


'''
training without eikonal constrainy
'''

def train(net, dataloader, val_dataloader, num_epochs, learning_rate, device, loss_threshold=1e-4, lambda_eikonal=0.1):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    net.to(device)

    for epoch in range(num_epochs):
        net.train()
        running_loss = 0.0

        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

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
