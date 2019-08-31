from __future__ import print_function
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torchvision import datasets, transforms

from dac_loss import DACLoss


class ErrorDataset(Dataset):

    def __init__(self, n_error, *args, **kwargs):
        self.base = datasets.MNIST('../data', train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,))
                                   ]))

        self.distored_idx = torch.randint(0, len(self.base), (n_error, ))

    def __getitem__(self, index):
        x, t = self.base[index]
        if index in self.distored_idx:
            mod_t = t + 1 if t < 9 else 0
            return x, mod_t
        else:
            return x, t

    def __len__(self):
        return len(self.base)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 11)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5)
        x = self.fc2(x)
        #return F.log_softmax(x, dim=1)
        return x


def train(args, model, criterion, device, train_loader, optimizer, epoch):
    model.train()

    correct = 0
    total = 0
    abstain = 0

    with tqdm(enumerate(train_loader), total=len(train_loader), desc="epoch {:02d}".format(epoch)) as t:
        for batch_idx, (data, target) in t:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target, epoch)
            #loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(output, 1)
            total += target.size(0)
            correct += predicted.eq(target).cpu().sum().item()

            abstain += predicted.eq(10).sum().item()
            if total != abstain:
                abst_acc = 100. * correct / float(total - abstain)
            else:
                abst_acc = 0.

        #if batch_idx % args.log_interval == 0:
    print('Train epoch {:02d}: Loss {:.6f} abstained {:04d}, correct {:.3f}%, abst_acc {:.3f}%'.format(
          epoch, loss.item(), abstain, 100.*correct/float(total), abst_acc))
    print('alpha = {:.6f}'.format(criterion.alpha_var) if criterion.alpha_var is not None else 'alpha = None')


def test(args, model, device, test_loader, epoch):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('Test  epoch {:02d}: Average loss {:.4f}, Accuracy {}/{} ({:.3f}%)'.format(
        epoch, test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    #train_loader = torch.utils.data.DataLoader(
    #    datasets.MNIST('../data', train=True, download=True,
    #                   transform=transforms.Compose([
    #                       transforms.ToTensor(),
    #                       transforms.Normalize((0.1307,), (0.3081,))
    #                   ])),
    #    batch_size=args.batch_size, shuffle=True, **kwargs)
    train_loader = torch.utils.data.DataLoader(
        ErrorDataset(10000),
        batch_size=args.batch_size, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    model = Net().to(device)
    criterion = DACLoss(learn_epochs=5, device=device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    for epoch in range(1, args.epochs + 1):
        train(args, model, criterion, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader, epoch)

    if (args.save_model):
        torch.save(model.state_dict(),"mnist_cnn.pt")


if __name__ == '__main__':
    main()
