import torch
import torch.nn as nn

from Data_Loaders import Data_Loaders, DatasetWrapper

class Action_Conditioned_FF(nn.Module):
    def __init__(self, input_size=6, hidden_size=128, output_size=1):
# STUDENTS: __init__() must initiatize nn.Module and define your network's
# custom architecture
        super(Action_Conditioned_FF, self).__init__()
        self.fc1 = nn.Linear(in_features=input_size, out_features=hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(in_features=hidden_size, out_features=output_size)

    def forward(self, input):
# STUDENTS: forward() must complete a single forward pass through your network
# and return the output which should be a tensor
        x = self.fc1(input)
        x = self.relu(x)
        output = self.fc2(x)
        return output


    def evaluate(self, model, test_loader, loss_function):
# STUDENTS: evaluate() must return the loss (a value, not a tensor) over your testing dataset. Keep in
# mind that we do not need to keep track of any gradients while evaluating the
# model. loss_function will be a PyTorch loss function which takes as argument the model's
# output and the desired output.
        model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for batch in test_loader:
                input, label = batch['input'], batch['label']
                output = model(input)
                loss = loss_function(output, label)
                total_loss += loss.item()
        return total_loss / len(test_loader)

def main():

    input_size = 6
    hidden_size = 128
    output_size = 1
    batch_size = 64
    data_loaders = Data_Loaders(batch_size)
  
    model = Action_Conditioned_FF(input_size, hidden_size, output_size)
    test_loader = torch.utils.data.DataLoader(DatasetWrapper(data_loaders.test_loader.dataset.data), batch_size=64, shuffle=True)

    loss_function = nn.MSELoss()

    test_loss = model.evaluate(model, test_loader, loss_function)
    print(f'Test Loss: {test_loss}')

if __name__ == '__main__':
    main()
