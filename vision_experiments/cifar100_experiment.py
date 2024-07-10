import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100
from tqdm import tqdm
import pickle
import argparse
import numpy as np
from dadaptation.dadapt_sgd import DAdaptSGD
from torchvision.models.densenet import DenseNet
import csv



def parse_args():
    parser = argparse.ArgumentParser(description='CIFAR-100 Training Script')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate')
    parser.add_argument('--num_epochs', type=int, default=300, help='Number of training epochs')
    parser.add_argument('--batch_size_per_gpu', type=int, default=64, help='Batch size per GPU')
    parser.add_argument('--learning_rate', type=float, default=0.05, help='Learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for SGD')
    parser.add_argument('--weight_decay', type=float, default=0.0002, help='Weight decay for SGD')
    parser.add_argument('--seed_count', type=int, default=2, help='Number of random seeds to use')
    parser.add_argument('--confusion_matrix', action='store_true', help='Include confusion matrix in results')
    parser.add_argument('--dadaptation', action='store_true', help='Use d-adaptation.')

    return parser.parse_args()

def calculate_confusion_matrix(model, test_loader, device):
    # Function to calculate the confusion matrix in the GPU
    model.eval()
    num_classes = len(test_loader.dataset.classes)
    confusion_matrix = torch.zeros(num_classes, num_classes)

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)

            for t, p in zip(targets.view(-1), predicted.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1

    return confusion_matrix

lr_schedule = [150,225]
def adjust_learning_rate(optimizer, epoch, lr_schedule):
    if epoch in lr_schedule:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.1

def main():
    args = parse_args()

    # Set random seeds for reproducibility
    seeds = np.random.randint(1, 10000, size=args.seed_count)
    results = []

    for seed in seeds:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = DenseNet(num_classes=100)


        # Data preprocessing
        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (1, 1, 1)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (1, 1, 1)),
        ])

        # Load CIFAR-100 dataset
        train_dataset = CIFAR100(root='./data', train=True, download=True, transform=transform_train)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size_per_gpu, shuffle=True, num_workers=4)

        test_dataset = CIFAR100(root='./data', train=False, download=True, transform=transform_test)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size_per_gpu, shuffle=False, num_workers=4)

        # Model, loss function, and optimizer
        model.to(device)
        criterion = nn.CrossEntropyLoss()
        if args.dadaptation: 
            optimizer = DAdaptSGD(model.parameters(), lr=args.learning_rate*10, momentum=args.momentum, weight_decay=args.weight_decay)
        else:
            optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)


        train_losses = []
        test_accuracies = []
        conf_mats = []
        # Training loop
        for epoch in range(args.num_epochs):
            model.train()
            train_loss = 0.0
            correct = 0
            total = 0

            tqdm_bar = tqdm(train_loader)
            for i, (inputs, targets) in enumerate(tqdm_bar):
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                tqdm_bar.set_description('Epoch %d | Loss: %.3f | Acc: %.3f%% (%d/%d)'
                                         % (epoch + 1, train_loss / (args.batch_size_per_gpu * (i + 1)),
                                            100. * correct / total, correct, total))

            # Save the training loss for this epoch
            train_losses.append(train_loss / len(train_loader))

            # Evaluate test accuracy
            model.eval()
            test_loss = 0.0
            correct = 0
            total = 0

            with torch.no_grad():
                for inputs, targets in test_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)

                    test_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()

            # Calculate and save test accuracy for this epoch
            test_accuracy = 100. * correct / total
            test_accuracies.append(test_accuracy)
            if args.confusion_matrix:
                conf_mat = calculate_confusion_matrix(model, test_loader, device)
                conf_mats.append(conf_mat)
            adjust_learning_rate(optimizer, epoch, lr_schedule)

        # Save results for this run
        result = {
            'seed': seed,
            'train_losses': train_losses,
            'test_accuracies': test_accuracies,
            'confusion_matrices': conf_mats if args.confusion_matrix else None,
        }
        results.append(result)

        # Save results to a file
        result_file = f"cifar100_dadapt_results_dropout_{args.dropout}_epochs_{args.num_epochs}_batch_{args.batch_size_per_gpu}_lr_{args.learning_rate}_momentum_{args.momentum}_wd_{args.weight_decay}_seed_{seed}.pkl"
        with open(result_file, 'wb') as f:
            pickle.dump(results, f)

    	# Save results to a CSV file
        result_csv_file = f"cifar100_dadapt_results_dropout_{args.dropout}_epochs_{args.num_epochs}_batch_{args.batch_size_per_gpu}_lr_{args.learning_rate}_momentum_{args.momentum}_wd_{args.weight_decay}_seed_{seed}.csv"
        with open(result_csv_file, 'w', newline='') as csv_file:
            fieldnames = ['seed', 'train_losses', 'test_accuracies','confusion_matrices']
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        	#Write header
            writer.writeheader()

        	# Write data
            for result in results:
            	writer.writerow(result)
    
if __name__ == "__main__":
    main()
