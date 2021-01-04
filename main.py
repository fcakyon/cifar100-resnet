import argparse
import os

import torch
from torchvision import transforms

import cifarresnet
from dataset import Cifar100Dataset
from utils import create_dir, import_class


def evaluate(dataloader, model, criterion, device):
    """
    Evaluate model performance over test data
    """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in dataloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print("Number of correct classifications:", correct)
    print("Number of total test samples:", total)
    print(
        "Accuracy of the network on the 10000 test images: %d %%"
        % (100 * correct / total)
    )


def save_checkpoint(state, filename):
    """
    Save the trained model state
    """
    create_dir(os.path.dirname(filename))
    torch.save(state, filename)


def train_one_epoch(trainloader, optimizer, criterion, model, epoch, device):
    """
    Train model for one epoch
    """
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(device), data[1].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # print statistics
    print(
        "epoch: %d, iter: %5d | loss: %.3f"
        % (epoch + 1, i + 1, running_loss / len(trainloader))
    )
    running_loss = 0.0


def main(args):
    create_dir(args.save_dir)
    input_size = 32

    # create model
    model_callback = import_class("cifarresnet.{}".format(args.arch))
    model = torch.nn.DataParallel(model_callback())

    # set transforms
    data_transforms = {
        "train": transforms.Compose(
            [
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
        "test": transforms.Compose(
            [
                transforms.Resize(input_size),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
    }

    # set cifar100 dataset
    cifar100_dataset = Cifar100Dataset(
        batch_size=args.batch_size,
        input_size=input_size,
        num_workers=args.workers,
        transforms=data_transforms,
    )

    # define loss function (criterion) and optimizer
    criterion = torch.nn.CrossEntropyLoss().cuda()

    if args.half:
        model.half()
        criterion.half()

    optimizer = torch.optim.SGD(
        params=model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[100, 150], last_epoch=-1
    )

    # set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device is set as:", device)

    model.to(device)
    criterion.to(device)

    if args.evaluate:
        model.load_state_dict(
            torch.load(args.evaluate, map_location=device)["state_dict"]
        )
        evaluate(
            dataloader=cifar100_dataset.testloader,
            model=model,
            criterion=criterion,
            device=device,
        )
        return

    # training loop
    for epoch in range(args.epochs):  # loop over the dataset multiple times

        # train
        train_one_epoch(
            trainloader=cifar100_dataset.trainloader,
            optimizer=optimizer,
            criterion=criterion,
            model=model,
            epoch=epoch,
            device=device,
        )

        # save checkpoint every save_every epochs
        if epoch > 0 and epoch % args.save_every == 0:
            save_checkpoint(
                state={
                    "epoch": epoch + 1,
                    "state_dict": model.state_dict(),
                },
                filename=os.path.join(args.save_dir, args.arch + "_checkpoint.th"),
            )

        lr_scheduler.step()

    # save final model
    save_checkpoint(
        state={
            "state_dict": model.state_dict(),
        },
        filename=os.path.join(args.save_dir, args.arch + "_final.th"),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ResNets for CIFAR100 in Pytorch")
    parser.add_argument(
        "--arch",
        "-a",
        metavar="ARCH",
        default="resnet32",
        choices=cifarresnet.__all__,
        help="model architecture: "
        + " | ".join(cifarresnet.__all__)
        + " (default: resnet32)",
    )
    parser.add_argument(
        "-j",
        "--workers",
        default=4,
        type=int,
        metavar="N",
        help="number of data loading workers (default: 4)",
    )
    parser.add_argument(
        "--epochs",
        default=200,
        type=int,
        metavar="N",
        help="number of total epochs to run",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        default=128,
        type=int,
        metavar="N",
        help="mini-batch size (default: 128)",
    )
    parser.add_argument(
        "--lr",
        "--learning-rate",
        default=0.1,
        type=float,
        metavar="LR",
        help="initial learning rate",
    )
    parser.add_argument(
        "--momentum", default=0.9, type=float, metavar="M", help="momentum"
    )
    parser.add_argument(
        "--weight-decay",
        "--wd",
        default=1e-4,
        type=float,
        metavar="W",
        help="weight decay (default: 1e-4)",
    )
    parser.add_argument(
        "--print-freq",
        "-p",
        default=50,
        type=int,
        metavar="N",
        help="print frequency (default: 50)",
    )
    parser.add_argument(
        "--half", dest="half", action="store_true", help="use half-precision(16-bit) "
    )
    parser.add_argument(
        "--save-dir",
        dest="save_dir",
        help="The directory used to save the trained models",
        default="save_temp",
        type=str,
    )
    parser.add_argument(
        "--save-every",
        dest="save_every",
        help="Saves checkpoints at every specified number of epochs",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--evaluate",
        default="",
        type=str,
        metavar="PATH",
        help="path to checkpoint to be used in evaluation (default: none)",
    )

    args = parser.parse_args()

    main(args)
