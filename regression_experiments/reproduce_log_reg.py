import argparse
from sklearn.datasets import load_svmlight_file
import warnings
import os
import dadaptation
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import lr_scheduler
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore", category=UserWarning)

def main(args):
    FILE_PATH = args.filepath
    file_data = os.listdir(FILE_PATH)
    print(f"Running Logistic Regression on {args.filename}...")
    for file in file_data:
        if file == args.filename:
            data = load_svmlight_file(os.path.join(FILE_PATH, file))


    class MulticlassLogisticRegressionModel(nn.Module):
        def __init__(self, input_size, num_classes):
            super(MulticlassLogisticRegressionModel, self).__init__()
            self.linear = nn.Linear(input_size, num_classes)

        def forward(self, x):
            out = self.linear(x)
            return out

    def csr2tensor(sparse_matrix):
        coo = sparse_matrix.tocoo()
        values = coo.data
        indices = np.vstack((coo.row, coo.col))

        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        shape = coo.shape

        res = torch.sparse.FloatTensor(i, v, torch.Size(shape)).to_dense()

        return res

    def train_model(model, optimizer, scheduler, criterion, train_loader, X_tensor, y_tensor, epochs, device):
        model.to(device)
        criterion.to(device)
        X_tensor = X_tensor.to(device)
        y_tensor = y_tensor.to(device)    
        
        epoch_losses = []
        epoch_accuracies = []

        for epoch in range(epochs):
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                epoch_losses.append(loss.item())

            scheduler.step()

            with torch.no_grad():
                outputs = model(X_tensor)
                _, predicted = torch.max(outputs, 1)
                accuracy = accuracy_score(y_tensor, predicted.numpy())
                epoch_accuracies.append(accuracy)

        return epoch_losses, epoch_accuracies

    def print_debug(debug, *args):
        if debug:
            print(*args)

    def train_and_plot_learning_rates(data, epochs=100, debug=False,batch_size = 16
    ):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        label_encoder = LabelEncoder()
        num_classes = len(np.unique(data[1]))
        learning_rates = [0.01, 0.1, 1, 10]

        seeds = [random.randint(1, 1000) for _ in range(10)]
        print_debug(debug, f"Seeds: {seeds}")

        loss_curves = np.empty((0, epochs * int(np.ceil(data[0].shape[0]/batch_size) )))
        accuracy_curves = np.empty((0, epochs))
        lrs = {seed: 0 for seed in seeds}

        def get_model(input_size, num_classes):
            return MulticlassLogisticRegressionModel(input_size, num_classes)

        for seed in seeds:
            X, y = shuffle(data[0], data[1], random_state=seed)
            X_tensor = csr2tensor(X)
            y_tensor = torch.LongTensor(label_encoder.fit_transform(y))

            train_dataset = TensorDataset(X_tensor, y_tensor)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

            input_size = data[0].shape[1]
            model = get_model(input_size, num_classes)

            criterion = nn.CrossEntropyLoss()
            best_accuracy = 0
            best_lr = None
            loss_curves_lr = {lr: [] for lr in learning_rates}
            accuracy_curves_lr = {lr: [] for lr in learning_rates}

            for lr in learning_rates:
                model = get_model(input_size, num_classes)
                criterion = nn.CrossEntropyLoss()
                print_debug(debug, f"Seed: {seed}, LR: {lr}")

                optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0)
                scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[60, 80, 95], gamma=0.1)

                epoch_losses, epoch_accuracies = train_model(model, optimizer, scheduler, criterion, train_loader,
                                                            X_tensor, y_tensor, epochs, device)

                loss_curves_lr[lr].append(epoch_losses)
                accuracy_curves_lr[lr].append(epoch_accuracies)

                print_debug(debug, f"Accuracy: {epoch_accuracies[-1]}")
                if epoch_accuracies[-1] > best_accuracy:
                    best_accuracy = epoch_accuracies[-1]
                    best_lr = lr

            loss_curves = np.vstack([loss_curves, np.array(loss_curves_lr[best_lr])])
            accuracy_curves = np.vstack([accuracy_curves, np.array(accuracy_curves_lr[best_lr])])
            lrs[seed] = best_lr

            print_debug(debug, f"Best Learning Rate: {best_lr}")
            print_debug(debug, f"Training Accuracy: {best_accuracy * 100:.2f}%")

        dadapt_loss_curves = np.empty((0, epochs * int(np.ceil(data[0].shape[0]/batch_size))))
        dadapt_accuracy_curves = np.empty((0, epochs))
        lr = 1
        for seed in seeds:
            X, y = shuffle(data[0], data[1], random_state=seed)

            X_tensor = csr2tensor(X)
            y_tensor = torch.LongTensor(label_encoder.fit_transform(y))

            train_dataset = TensorDataset(X_tensor, y_tensor)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

            input_size = data[0].shape[1]
            model = get_model(input_size, num_classes)

            criterion = nn.CrossEntropyLoss()
            best_accuracy = 0
            loss_curves_lr = {lr: [] for lr in learning_rates}
            accuracy_curves_lr = {lr: [] for lr in learning_rates}

            model = get_model(input_size, num_classes)
            criterion = nn.CrossEntropyLoss()
  
            optimizer = dadaptation.dadapt_adam.DAdaptAdam(model.parameters(), lr=lr, weight_decay=0)
            scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[60, 80, 95], gamma=0.1)
            epoch_losses, epoch_accuracies = train_model(model, optimizer, scheduler, criterion, train_loader,
                                                        X_tensor, y_tensor, epochs, device)

            loss_curves_lr[lr].append(epoch_losses)
            accuracy_curves_lr[lr].append(epoch_accuracies)

            dadapt_loss_curves = np.vstack([dadapt_loss_curves, np.array(loss_curves_lr[lr])])
            dadapt_accuracy_curves = np.vstack([dadapt_accuracy_curves, np.array(accuracy_curves_lr[lr])])

        # Plot the mean loss and accuracy curves for each learning rate
        mean_accuracies = np.mean(accuracy_curves, axis=0)
        std_accuracies = np.std(accuracy_curves, axis=0)
        upper_bound = mean_accuracies + std_accuracies
        lower_bound = mean_accuracies - std_accuracies

        mean_loss = np.mean(loss_curves, axis=0)
        std_loss = np.std(loss_curves, axis=0)
        lower_bound_losses = mean_loss - std_loss
        upper_bound_losses = mean_loss + std_loss

        dadapt_mean_accuracies = np.mean(dadapt_accuracy_curves, axis=0)
        dadapt_std_accuracies = np.std(dadapt_accuracy_curves, axis=0)

        dadapt_upper_bound = dadapt_mean_accuracies + dadapt_std_accuracies
        dadapt_lower_bound = dadapt_mean_accuracies - dadapt_std_accuracies

        dadapt_mean_loss = np.mean(dadapt_loss_curves, axis=0)
        dadapt_std_loss = np.std(dadapt_loss_curves, axis=0)
        dadapt_lower_bound_losses = dadapt_mean_loss - dadapt_std_loss
        dadapt_upper_bound_losses = dadapt_mean_loss + dadapt_std_loss
            
        
        
        plt.rcParams['font.size'] = '12'
        fig_acc, ax_acc = plt.subplots(2, 1, figsize=(3, 3),sharex=True,sharey=True)
        fig_loss, ax_loss = plt.subplots(2, 1, figsize=(3, 3),sharex=True,sharey=True)

        ax_acc[0].plot(mean_accuracies, label=f'Adam', color='red', linestyle='dotted')
        ax_acc[0].fill_between(range(len(mean_accuracies)), lower_bound, upper_bound, alpha=0.1, color='red')
        ax_acc[1].plot(dadapt_mean_accuracies, label=f'D-adapt. Adam', color='black')
        ax_acc[1].fill_between(range(len(dadapt_mean_accuracies)),  np.maximum(0,dadapt_lower_bound), dadapt_upper_bound, alpha=0.1,
                        color='black')
        fig_acc.text(0.00, 0.5, 'Train Accuracy', va='center', rotation='vertical')
        ax_acc[1].set_xlabel('Epoch')
        ax_acc[0].legend(loc='lower right')
        ax_acc[1].legend(loc='lower right')
        fig_acc.tight_layout()

        savename = args.savename
        fig_acc.savefig(f'plots/{savename}_acc.png')


        ax_loss[0].plot(mean_loss, label=f'Adam', color='red', linestyle='dotted')
        ax_loss[0].fill_between(range(len(mean_loss)), np.maximum(0,lower_bound_losses), upper_bound_losses, alpha=0.1, color='red')
        ax_loss[1].plot(dadapt_mean_loss, label=f'D-adapt. Adam', color='black')
        ax_loss[1].fill_between(range(len(dadapt_mean_loss)),  np.maximum(0,dadapt_lower_bound_losses), dadapt_upper_bound_losses, alpha=0.1,
                        color='black')
        fig_loss.text(0.00, 0.5, 'Train Loss', va='center', rotation='vertical')
        ax_loss[1].set_xlabel('Step')
        ax_loss[0].legend(loc='upper right')
        ax_loss[1].legend(loc='upper right')
        ax_loss[0].set_ylim(bottom=np.min(dadapt_mean_loss)*0.9 - 1e-3,top=np.max(np.maximum(mean_loss,dadapt_mean_loss))*1.1)
        fig_loss.tight_layout()

        fig_loss.savefig(f'plots/{savename}_loss.png')
        return loss_curves, accuracy_curves, dadapt_loss_curves, dadapt_accuracy_curves

    train_and_plot_learning_rates(
        data,
        epochs=args.epochs,
        debug=args.debug,
        batch_size=args.batch_size
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and plot learning rates.")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--savename", type=str, default="placeholder", help="Name for saving plots")
    parser.add_argument("--filename", type=str, default="letter.scale", help="Name of the file to load")
    parser.add_argument("--filepath", type=str, default="../data/regression_experiments/raw/libsvm/", help="Path to the file")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    args = parser.parse_args()

    main(args)
    print(f"Done. Plot saved in plots/{args.savename}")
