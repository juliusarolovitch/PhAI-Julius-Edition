import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torch import nn
from models import MLPModel
import os


class PathDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_tensor, g_value, h_value, f_star_value = self.data[idx]
        input_tensor = input_tensor.float() / input_tensor.max()
        g_h_values = torch.tensor([g_value, h_value]).float()
        return input_tensor.float(), g_h_values.float(), torch.tensor(f_star_value).float()


def loss_function(inputs, output, target):
    mse_loss = F.mse_loss(output, target)
    # take the gradient of the output with respect to the input
    grad_output = torch.autograd.grad(outputs=output,
                                      inputs=inputs,
                                      grad_outputs=torch.ones_like(output),
                                      create_graph=True,
                                      retain_graph=True,
                                      only_inputs=True)[0]

    grad_x = grad_output[:, 1]
    grad_y = grad_output[:, 0]
    grad_0_loss = torch.clamp(-grad_x, min=0)
    grad_1_loss = torch.clamp(-grad_y, min=0)
    grad_01_loss = torch.clamp(grad_y - grad_x, min=0)
    grad_01_sum_loss = torch.clamp(grad_x + grad_y - 2, min=0)
    grad_loss = grad_0_loss + grad_1_loss + grad_01_loss + grad_01_sum_loss
    mse_loss += grad_loss.mean()

    return mse_loss


def train_model(model, train_loader, val_loader, device, model_path, epochs=100, lr=0.001, weight_decay=0.0, patience=10, verbose=True):
    optimizer = optim.Adam(model.parameters(), lr=lr,
                           weight_decay=weight_decay)
    criterion = loss_function
    val_crit = nn.MSELoss()
    if verbose:
        print('Training model with {} parameters...'.format(sum(p.numel()
              for p in model.parameters() if p.requires_grad)))

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in tqdm(range(epochs), desc="Training Epochs"):
        model.train()
        train_loss = 0
        for batch_idx, (data, g_h_values, target) in enumerate(train_loader):
            data, g_h_values, target = data.to(
                device), g_h_values.to(device), target.to(device)
            data.requires_grad = True
            optimizer.zero_grad()
            output = model(data, g_h_values)
            loss = criterion(data, output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data, g_h_values, target in val_loader:
                data, g_h_values, target = data.to(
                    device), g_h_values.to(device), target.to(device)
                output = model(data, g_h_values)
                val_loss += val_crit(output.squeeze(), target).item()
            val_loss /= len(val_loader.dataset)

        if verbose:
            print(
                f'Epoch: {epoch + 1}/{epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), model_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                if verbose:
                    print('Early stopping triggered')
                break

    return model


if __name__ == '__main__':
    dataset_paths = [os.path.join('datasets_cpp', f) for f in os.listdir(
        'datasets_cpp') if f.startswith('dataset_map_')]
    datasets = [torch.load(path) for path in dataset_paths]
    dataset = [item for sublist in datasets for item in sublist]

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        PathDataset(dataset), [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_size = (20 * 20 * 20 * 4) + 2
    output_size = 1
    model = MLPModel(input_size=input_size, output_size=output_size).to(device)
    model_path = 'model.pth'

    num_epochs = 15
    train_model(model, train_loader, test_loader, device, model_path,
                epochs=num_epochs, lr=0.001, weight_decay=0.0, patience=5, verbose=True)

    torch.save(model.state_dict(), model_path)
