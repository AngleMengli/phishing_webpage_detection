import os
import torch.utils.data
import wandb
from PIL import Image
from datetime import datetime
from torch.nn import DataParallel
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

from config import BATCH_SIZE, PROPOSAL_NUM, SAVE_FREQ, LR, WD, resume, save_dir
from config import INPUT_SIZE
from core import model
from core.utils import init_log, progress_bar

if __name__ == "__main__":
    hyperparameter_defaults = dict(
        BATCH_SIZE=BATCH_SIZE,
        PROPOSAL_NUM=PROPOSAL_NUM,
        SAVE_FREQ=SAVE_FREQ,
        LR=LR,
        WD=WD,
        resume=resume,
        save_dir=save_dir,
        INPUT_SIZE=INPUT_SIZE
    )
    wandb.init(project="NTS-Net", entity="exception", config=hyperparameter_defaults)
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
    start_epoch = 1
    save_dir = os.path.join(save_dir, datetime.now().strftime('%Y%m%d_%H%M%S'))
    if not os.path.exists(save_dir):
        # raise NameError('model dir exists!')
        os.makedirs(save_dir)
    logging = init_log(save_dir)
    _print = logging.info

    # trainset = dataset.Phish(root=r"I:\wei\phish", is_train=True, data_len=5000)
    # trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, drop_last=False)

    # testset = dataset.Phish(root=r"I:\wei\phish", is_train=False, data_len=1000)
    # testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, drop_last=False)
    transform_train = transforms.Compose([
        transforms.Resize(INPUT_SIZE, Image.BILINEAR),
        # transforms.RandomCrop(INPUT_SIZE),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    # Augmentation is not done for test/validation data.
    transform_test = transforms.Compose([
        transforms.Resize(INPUT_SIZE, Image.BILINEAR),
        # transforms.CenterCrop(INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_ds = ImageFolder("dataset/train/", transform=transform_train)
    test_ds = ImageFolder("dataset/val/", transform=transform_test)
    pred_ds = ImageFolder("dataset/pred/", transform=transform_test)

    train_dl = DataLoader(train_ds, BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_dl = DataLoader(test_ds, BATCH_SIZE, num_workers=4, pin_memory=True)
    pred_dl = DataLoader(pred_ds, BATCH_SIZE, num_workers=4, pin_memory=True)

    # define model
    net = model.attention_net(topN=PROPOSAL_NUM)
    if resume:
        ckpt = torch.load(resume)
        net.load_state_dict(ckpt['net_state_dict'])
        start_epoch = ckpt['epoch'] + 1
    creterion = torch.nn.CrossEntropyLoss()

    # define optimizers
    raw_parameters = list(net.pretrained_model.parameters())
    part_parameters = list(net.proposal_net.parameters())
    concat_parameters = list(net.concat_net.parameters())
    partcls_parameters = list(net.partcls_net.parameters())

    raw_optimizer = torch.optim.SGD(raw_parameters, lr=LR, momentum=0.9, weight_decay=WD)
    concat_optimizer = torch.optim.SGD(concat_parameters, lr=LR, momentum=0.9, weight_decay=WD)
    part_optimizer = torch.optim.SGD(part_parameters, lr=LR, momentum=0.9, weight_decay=WD)
    partcls_optimizer = torch.optim.SGD(partcls_parameters, lr=LR, momentum=0.9, weight_decay=WD)
    schedulers = [MultiStepLR(raw_optimizer, milestones=[60, 100], gamma=0.1),
                  MultiStepLR(concat_optimizer, milestones=[60, 100], gamma=0.1),
                  MultiStepLR(part_optimizer, milestones=[60, 100], gamma=0.1),
                  MultiStepLR(partcls_optimizer, milestones=[60, 100], gamma=0.1)]
    net = net.cuda()
    net = DataParallel(net, device_ids=[0, 1])
    wandb.watch(net)
    for epoch in range(start_epoch, 500):
        for scheduler in schedulers:
            scheduler.step()

        # begin training
        _print('--' * 50)
        net.train()
        for i, data in enumerate(train_dl):
            img, label = data[0].cuda(), data[1].cuda()
            batch_size = img.size(0)
            raw_optimizer.zero_grad()
            part_optimizer.zero_grad()
            concat_optimizer.zero_grad()
            partcls_optimizer.zero_grad()

            raw_logits, concat_logits, part_logits, _, top_n_prob = net(img)
            part_loss = model.list_loss(part_logits.view(batch_size * PROPOSAL_NUM, -1),
                                        label.unsqueeze(1).repeat(1, PROPOSAL_NUM).view(-1)).view(batch_size,
                                                                                                  PROPOSAL_NUM)
            raw_loss = creterion(raw_logits, label)
            concat_loss = creterion(concat_logits, label)
            rank_loss = model.ranking_loss(top_n_prob, part_loss)
            partcls_loss = creterion(part_logits.view(batch_size * PROPOSAL_NUM, -1),
                                     label.unsqueeze(1).repeat(1, PROPOSAL_NUM).view(-1))

            total_loss = raw_loss + rank_loss + concat_loss + partcls_loss
            total_loss.backward()
            raw_optimizer.step()
            part_optimizer.step()
            concat_optimizer.step()
            partcls_optimizer.step()
            progress_bar(i, len(train_dl), 'train')

        if epoch % SAVE_FREQ == 0:
            train_loss = 0
            train_correct = 0
            total = 0
            net.eval()
            for i, data in enumerate(val_dl):
                with torch.no_grad():
                    img, label = data[0].cuda(), data[1].cuda()
                    batch_size = img.size(0)
                    _, concat_logits, _, _, _ = net(img)
                    # calculate loss
                    concat_loss = creterion(concat_logits, label)
                    # calculate accuracy
                    _, concat_predict = torch.max(concat_logits, 1)
                    total += batch_size
                    train_correct += torch.sum(concat_predict.data == label.data)
                    train_loss += concat_loss.item() * batch_size
                    progress_bar(i, len(val_dl), 'eval train set')

            train_acc = float(train_correct) / total
            train_loss = train_loss / total

            _print(
                'epoch:{} - train loss: {:.3f} and train acc: {:.3f} total sample: {}'.format(
                    epoch,
                    train_loss,
                    train_acc,
                    total))
            wandb.log({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc, "total": total})
            # evaluate on test set
            test_loss = 0
            test_correct = 0
            total = 0
            for i, data in enumerate(pred_dl):
                with torch.no_grad():
                    img, label = data[0].cuda(), data[1].cuda()
                    batch_size = img.size(0)
                    _, concat_logits, _, _, _ = net(img)
                    # calculate loss
                    concat_loss = creterion(concat_logits, label)
                    # calculate accuracy
                    _, concat_predict = torch.max(concat_logits, 1)
                    total += batch_size
                    test_correct += torch.sum(concat_predict.data == label.data)
                    test_loss += concat_loss.item() * batch_size
                    progress_bar(i, len(pred_dl), 'eval test set')

            test_acc = float(test_correct) / total
            test_loss = test_loss / total
            _print(
                'epoch:{} - test loss: {:.3f} and test acc: {:.3f} total sample: {}'.format(
                    epoch,
                    test_loss,
                    test_acc,
                    total))
            wandb.log({"epoch": epoch, "test_loss": test_loss, "test_acc": test_acc, "total sample": total})

            # save model
            net_state_dict = net.module.state_dict()
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            torch.save({
                'epoch': epoch,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'test_loss': test_loss,
                'test_acc': test_acc,
                'net_state_dict': net_state_dict},
                os.path.join(save_dir, '%03d.ckpt' % epoch))

    print('finishing training')
