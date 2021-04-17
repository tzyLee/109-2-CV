import sys
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import torch.optim as optim
import matplotlib.pyplot as plt
from model import ConvNet, MyNet
from data import get_dataloader

if __name__ == "__main__":
    # Specifiy data folder path and model type(fully/conv)
    folder, model_type = sys.argv[1], sys.argv[2]

    # Get data loaders of training set and validation set
    train_loader, val_loader = get_dataloader(folder, batch_size=32)

    # Specify the type of model
    if model_type == "conv":
        model = ConvNet()
    elif model_type == "mynet":
        model = MyNet()

    # Set the type of gradient optimizer and the model it update
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    # Choose loss function
    criterion = nn.CrossEntropyLoss()

    # Check if GPU is available, otherwise CPU is used
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model.cuda()

    # Run any number of epochs you want
    train_acc = []
    train_loss = []
    val_acc = []
    val_loss = []
    ep = 20
    scheduler = optim.lr_scheduler.StepLR(optimizer, 5)
    for epoch in range(ep):
        print("Epoch:", epoch)
        ##############
        ## Training ##
        ##############
        # Record the information of correct prediction and loss
        correct_cnt, total_loss, total_cnt = 0, 0, 0

        model.train()
        # Load batch data from dataloader
        for batch, (x, label) in enumerate(train_loader, 1):
            # Set the gradients to zero (left by previous iteration)
            optimizer.zero_grad()
            # Put input tensor to GPU if it's available
            if use_cuda:
                x, label = x.cuda(), label.cuda()
            # Forward input tensor through your model
            out = model(x)
            # Calculate loss
            loss = criterion(out, label)
            # Compute gradient of each model parameters base on calculated loss
            loss.backward()
            # Update model parameters using optimizer and gradients
            optimizer.step()

            # Calculate the training loss and accuracy of each iteration
            total_loss += loss.item()
            _, pred_label = torch.max(out, 1)
            total_cnt += x.size(0)
            correct_cnt += (pred_label == label).sum().item()

            # Show the training information
            if batch % 500 == 0 or batch == len(train_loader):
                acc = correct_cnt / total_cnt
                ave_loss = total_loss / batch
                print(
                    "Training batch index: {}, train loss: {:.6f}, acc: {:.3f}".format(
                        batch, ave_loss, acc
                    )
                )
        train_acc.append(correct_cnt / total_cnt)
        train_loss.append(total_loss / total_cnt)

        scheduler.step()
        ################
        ## Validation ##
        ################
        # Record the information of correct prediction and loss
        correct_cnt, total_loss, total_cnt = 0, 0, 0

        model.eval()
        with torch.no_grad():
            # Load batch data from dataloader
            for batch, (x, label) in enumerate(val_loader, 1):
                # Put input tensor to GPU if it's available
                if use_cuda:
                    x, label = x.cuda(), label.cuda()
                # Forward input tensor through your model
                out = model(x)
                # Calculate loss
                loss = criterion(out, label)

                # Calculate the training loss and accuracy of each iteration
                total_loss += loss.item()
                _, pred_label = torch.max(out, 1)
                total_cnt += x.size(0)
                correct_cnt += (pred_label == label).sum().item()

                # Show the training information
                if batch % 100 == 0 or batch == len(val_loader):
                    acc = correct_cnt / total_cnt
                    ave_loss = total_loss / batch
                    print(
                        "Validation batch index: {}, valid loss: {:.6f}, acc: {:.3f}".format(
                            batch, ave_loss, acc
                        )
                    )
        val_acc.append(correct_cnt / total_cnt)
        val_loss.append(total_loss / total_cnt)
    # Save trained model
    torch.save(model.state_dict(), "./checkpoint/{}.pth".format(model.name()))

    # Plot Learning Curve
    x = list(range(1, ep + 1))
    x_tick = x[::2]
    plt.figure()
    plt.grid(True, linestyle="-.")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.plot(x, train_loss, label="Training Loss")
    plt.legend()
    plt.title(model_type)
    plt.tight_layout()
    plt.xticks(x_tick)
    plt.savefig("./checkpoint/{}_train_loss.png".format(model_type))

    plt.figure()
    plt.grid(True, linestyle="-.")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.plot(x, train_acc, label="Training Accuracy")
    plt.legend()
    plt.title(model_type)
    plt.tight_layout()
    plt.xticks(x_tick)
    plt.ylim(top=1)
    plt.savefig("./checkpoint/{}_train_acc.png".format(model_type))

    plt.figure()
    plt.grid(True, linestyle="-.")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.plot(x, val_loss, label="Validation Loss")
    plt.legend()
    plt.title(model_type)
    plt.tight_layout()
    plt.xticks(x_tick)
    plt.savefig("./checkpoint/{}_val_loss.png".format(model_type))

    plt.figure()
    plt.grid(True, linestyle="-.")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.plot(x, val_acc, label="Validation Accuracy")
    plt.legend()
    plt.title(model_type)
    plt.tight_layout()
    plt.xticks(x_tick)
    plt.ylim(top=1)
    plt.savefig("./checkpoint/{}_val_acc.png".format(model_type))
