import torch
import torch.nn as nn
from tqdm import tqdm
import sys




def train_one_epoch(model, optimizer, data_loader, epoch):

    model.cuda()
    model.train()

    accu_loss = torch.zeros(1).cuda()
    accu_num = torch.zeros(1).cuda()

    optimizer.zero_grad()

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)

    for step, data in enumerate(data_loader):
        images, labels = data
        images, labels = images.cuda(), labels.cuda()

        output = model(images)
        sample_num += images.shape[0]
        criterion = nn.CrossEntropyLoss()
        loss = criterion(output, labels)
        pred_classes = torch.max(output, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels).sum()


        loss.backward()
        accu_loss += loss.detach()

        data_loader.desc = "[train epoch {}] loss: {:.3f}, acc:{:.3f}".format(epoch,
                                                                              accu_loss.item() / (step + 1),
                                                                              accu_num.item() / (sample_num))

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)
        optimizer.step()
        optimizer.zero_grad()

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num