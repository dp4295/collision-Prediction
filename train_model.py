from Data_Loaders import Data_Loaders
from Networks import Action_Conditioned_FF
from torch.optim.lr_scheduler import ReduceLROnPlateau

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F


class CustomLoss(nn.Module):
    def __init__(self, collision_weight=1.0):
        super(CustomLoss,self).__init__()
        self.collision_weight = collision_weight

    def forward(self, output, label):
        mse_loss = F.mse_loss(output, label)

        collision_loss = self.collision_weight * F.mse_loss(output[label == 1], label[label ==1])
        total_loss = mse_loss + collision_loss
        
        return total_loss

def train_model(no_epochs, patience=5):

    batch_size = 64
    data_loaders = Data_Loaders(batch_size)
    model = Action_Conditioned_FF(input_size=6, hidden_size=128, output_size=1)

   # loss_function = CustomLoss(collision_weight=5.0)
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

   # scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

    losses = []
    min_loss = float('inf')
    early_stopping_counter = 0 



    for epoch_i in range(no_epochs):
        model.train()
        epoch_loss = 0.0

        for idx, sample in enumerate(data_loaders.train_loader): # sample['input'] and sample['label']
            input, label = sample['input'], sample['label']

        optimizer.zero_grad()
        output = model(input) 
        loss = loss_function(output,label)
        loss.backward() 
        optimizer.step()

        epoch_loss += loss.item()
    
        avg_loss = epoch_loss / len(data_loaders.train_loader)
        print(f"Epoch {epoch_i + 1}/{no_epochs}, Train Loss: {avg_loss}")

        # Evaluate on the test set
        model.eval()
        test_loss = model.evaluate(model, data_loaders.test_loader, loss_function)
        print(f"Epoch {epoch_i + 1}/{no_epochs}, Test Loss: {test_loss}")

        losses.append(test_loss)

        # Learning rate scheduler step
       # scheduler.step(test_loss)

        if test_loss < min_loss: 
            min_loss = test_loss 
            torch.save(model.state_dict(), "saved/saved_model.pkl")
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
    
        if early_stopping_counter >= patience: 
            print(f"Stopping early. No improvement for {patience} consecutive epochs.")
            break
        
    plt.plot(losses, label='Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    no_epochs = 200
    train_model(no_epochs)