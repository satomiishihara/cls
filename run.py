import torch
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import trainer
import tester
from model import VIT_model


imgsize = 224
epochs = 10
data_transforms = transforms.Compose([transforms.Resize([imgsize, imgsize]),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
train_data = datasets.CIFAR10(root="dataset/train",
                              train=True,
                              transform=data_transforms,
                              download=True)
test_data = datasets.CIFAR10(root="dataset/test",
                             train=False,
                             transform=data_transforms,
                             download=True)

batch_size = 16
train_dataloader = torch.utils.data.DataLoader(train_data,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=0)
test_dataloader = torch.utils.data.DataLoader(test_data,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              num_workers=0)
model = VIT_model(num_classes=10)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

for epoch in range(epochs):
    train_loss, train_acc = trainer.train_one_epoch(model=model,
                                                    optimizer=optimizer,
                                                    data_loader=train_dataloader,
                                                    epoch=epoch)

    test_loss, test_acc = tester.test_one_epoch(model=model,
                                                data_loader=test_dataloader,
                                                epoch=epoch)

torch.save({'model_state_dict': model.state_dict()},
            "savepath/model-{}-{}-last.pth".format(epoch, test_acc))

