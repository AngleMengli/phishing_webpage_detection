import json
import os
import torch.utils.data
# import wandb
from PIL import Image
from datetime import datetime
from torch.nn import DataParallel
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

from config import BATCH_SIZE, PROPOSAL_NUM, SAVE_FREQ, LR, WD, resume, save_dir, CAT_NUM
from config import INPUT_SIZE
from core import model, dataset, utils
from core.utils import init_log, progress_bar, batch_images
import numpy as np


def transform_invert(img_, transform_train):
    """
    将data 进行反transfrom操作
    :param img_: Tensor
    :param transform_train: torchvision.transforms
    :return: PIL image
    """
    if 'Normalize' in str(transform_train):
        norm_transform = list(filter(lambda x: isinstance(x, transforms.Normalize), transform_train.transforms))
        mean = torch.tensor(norm_transform[0].mean, dtype=img_.dtype, device=img_.device)
        std = torch.tensor(norm_transform[0].std, dtype=img_.dtype, device=img_.device)
        img_.mul_(std[:, None, None]).add_(mean[:, None, None])

    # img_ = img_.transpose(0, 2).transpose(0, 1)  # C*H*W --> H*W*C
    img_ = img_.permute(1, 2, 0)  # C*H*W --> H*W*C
    np.seterr(divide='ignore', invalid='ignore')
    if 'ToTensor' in str(transform_train):
        img_ = np.array(img_)  # 先把Tensor转换成numpy.darray
        # if np.min(img_) == np.min(img_):
        #     print(img_)
        img_ -= np.min(img_)
        # if np.max(img_) == 0:
        #     print(img_)
        img_ /= np.max(img_)
        img_ = img_ * 255

    # 再把numpy.darray转换成PIL.Image
    if img_.shape[2] == 3:
        img_ = Image.fromarray(img_.astype('uint8')).convert('RGB')
    elif img_.shape[2] == 1:
        img_ = Image.fromarray(img_.astype('uint8').squeeze())
    else:
        raise Exception("Invalid img shape, expected 1 or 3 in axis 2, but got {}!".format(img_.shape[2]))

    return img_


if __name__ == "__main__":
    hyperparameter_defaults = dict(
        BATCH_SIZE=BATCH_SIZE,
        PROPOSAL_NUM=PROPOSAL_NUM,
        CAT_NUM=CAT_NUM,
        SAVE_FREQ=SAVE_FREQ,
        LR=LR,
        WD=WD,
        resume=resume,
        save_dir=save_dir,
        INPUT_SIZE=INPUT_SIZE
    )
    # wandb.init(project="NTS-Net", entity="exception", config=hyperparameter_defaults)
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    start_epoch = 1
    save_dir = os.path.join(save_dir, datetime.now().strftime('%Y%m%d_%H%M%S'))
    if not os.path.exists(save_dir):
        # raise NameError('model dir exists!')
        os.makedirs(save_dir)
    logging = init_log(save_dir)
    _print = logging.info
    _print(hyperparameter_defaults)
    # trainset = dataset.Phish(root=r"I:\wei\phish", is_train=True, data_len=5000)
    # trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, drop_last=False)

    # testset = dataset.Phish(root=r"I:\wei\phish", is_train=False, data_len=1000)
    # testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, drop_last=False)
    transform_train = transforms.Compose([
        transforms.Resize(INPUT_SIZE, Image.BILINEAR),
        transforms.RandomCrop(INPUT_SIZE),
        transforms.Grayscale(num_output_channels=3),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        #这个归一化的均值和方差是在imagenet上的图片集中计算出来的
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        # 这个归一化的均值和方差是在trustaddphishing上的图片集中计算出来的,没有灰度直接计算出来的
        # transforms.Normalize([0.7639301, 0.77205104, 0.7801083], [0.2628308, 0.24516599, 0.23961127])
    ])
    # Augmentation is not done for test/validation data.
    transform_test = transforms.Compose([
        transforms.Resize(INPUT_SIZE, Image.BILINEAR),
        # transforms.CenterCrop(INPUT_SIZE),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        # transforms.Normalize((0.1307,), (0.3081,))
        # transforms.Normalize([0.7639301, 0.77205104, 0.7801083], [0.2628308, 0.24516599, 0.23961127])
    ])

    # train_ds = ImageFolder("I:/NTS-NET-Visualphish/dataset-split2/train", transform=transform_train)
    train_ds = ImageFolder("I:/Phishpedia-dataset/new_benign", transform=transform_train)
    pred_ds = ImageFolder("I:/Phishpedia-dataset/new_phish", transform=transform_test)
    # test_ds = ImageFolder("I://NTS-NET-Visualphish//val", transform=transform_test)
    # pred_ds = ImageFolder("I:/NTS-NET-Visualphish/dataset-split2/test", transform=transform_test)

    web_list = train_ds.class_to_idx
    cla_dict = dict((val, key) for key, val in web_list.items())
    # write dict into json file
    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    # train_dl = DataLoader(train_ds, BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True, collate_fn=dataset.my_collate)
    # val_dl = DataLoader(test_ds, BATCH_SIZE, num_workers=4, pin_memory=True, collate_fn=dataset.my_collate)
    # pred_dl = DataLoader(pred_ds, BATCH_SIZE, num_workers=4, pin_memory=True, collate_fn=dataset.my_collate)
    train_dl = DataLoader(train_ds, BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    # val_dl = DataLoader(test_ds, BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    pred_dl = DataLoader(pred_ds, BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # define model
    net = model.attention_net(topN=PROPOSAL_NUM)
    print(net)
    # 这里判断是否为继续训练，无用代码
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

    # wandb.watch(net)

    for epoch in range(start_epoch, 60):
        for scheduler in schedulers:
            scheduler.step()

        # begin training
        _print('--' * 50)
        net.train()
        for i, data in enumerate(train_dl):
            # img = utils.batch_images(data[0])
            img, label = data[0].cuda(), data[1].cuda()
            batch_size = img.size(0)

            raw_optimizer.zero_grad()
            part_optimizer.zero_grad()
            concat_optimizer.zero_grad()
            partcls_optimizer.zero_grad()
            raw_logits, concat_logits, part_logits, top_n_index, top_n_prob = net(img)
            #part_imgs_points[0, :, :]

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
            for i, data in enumerate(train_dl):
                with torch.no_grad():
                    img, label = data[0].cuda(), data[1].cuda()
                    batch_size = img.size(0)
                    _, concat_logits, _, _, _, = net(img)
                    # calculate loss
                    concat_loss = creterion(concat_logits, label)
                    # calculate accuracy
                    _, concat_predict = torch.max(concat_logits, 1)
                    total += batch_size
                    train_correct += torch.sum(concat_predict.data == label.data)
                    train_loss += concat_loss.item() * batch_size
                    progress_bar(i, len(train_dl), 'eval train set')

            train_acc = float(train_correct) / total
            train_loss = train_loss / total

            _print(
                'epoch:{} - train loss: {:.3f} and train acc: {:.3f} total sample: {}'.format(
                    epoch,
                    train_loss,
                    train_acc,
                    total))
            # wandb.log({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc})
            # evaluate on test set
            # test_loss = 0
            # test_correct = 0
            # total = 0
            # for i, data in enumerate(pred_dl):
            #     with torch.no_grad():
            #         img, label = data[0].cuda(), data[1].cuda()
            #         batch_size = img.size(0)
            #         _, concat_logits, _, _, _ = net(img)
            #         # calculate loss
            #         concat_loss = creterion(concat_logits, label)
            #         # calculate accuracy
            #         _, concat_predict = torch.max(concat_logits, 1)
            #         total += batch_size
            #         test_correct += torch.sum(concat_predict.data == label.data)
            #         test_loss += concat_loss.item() * batch_size
            #         progress_bar(i, len(pred_dl), 'eval test set')
            #
            # test_acc = float(test_correct) / total
            # test_loss = test_loss / total
            # _print(
            #     'epoch:{} - test loss: {:.3f} and test acc: {:.3f} total sample: {}'.format(
            #         epoch,
            #         test_loss,
            #         test_acc,
            #         total))
            # wandb.log({"epoch": epoch, "test_loss": test_loss, "test_acc": test_acc})

            # save model
            net_state_dict = net.module.state_dict()
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            torch.save({
                'epoch': epoch,
                'train_loss': train_loss,
                'train_acc': train_acc,
                # 'test_loss': test_loss,
                # 'test_acc': test_acc,
                'net_state_dict': net_state_dict},
                os.path.join(save_dir, '%03d.ckpt' % epoch))
    print('finishing training')

    net.eval()
    # columns = ["label", "prediction", "image", "truth"]
    table_data = []
    # my_table = wandb.Table()

    # for i, data in enumerate(pred_dl):
    #     with torch.no_grad():
    #         img, label = data[0].cuda(), data[1].cuda()
    #         batch_size = img.size(0)
    #         _, concat_logits, _, _, _ = net(img)
    #         # calculate loss
    #         concat_loss = creterion(concat_logits, label)
    #         # calculate accuracy
    #         _, concat_predict = torch.max(concat_logits, 1)
    #         total += batch_size
    #         test_correct += torch.sum(concat_predict.data == label.data)
    #         test_loss += concat_loss.item() * batch_size
    #         progress_bar(i, len(pred_dl), 'eval test set')
    #



            # my_table.add_column("image", img.cpu().numpy().tolist())
            # my_table.add_column("prediction", concat_predict.cpu().numpy().tolist())
            # my_table.add_column("label", label.cpu().numpy().tolist())
            # my_table.add_column("pro", concat_logits.cpu().numpy().tolist())
            # [my_table.add_data(l, pre, wandb.Image(transforms.ToPILImage()(im.to('cpu'))), pro) for l, pre, im, pro in zip(label, concat_predict, img, concat_logits)]
            # for l, p, im, prb in zip(label.cpu().numpy(), concat_predict.cpu().numpy(),
            #                          data[0], concat_logits.cpu().numpy()):
            #     im = np.transpose(im.numpy(), (1, 2, 0))
            #     table_data.append(
            #         [l.data, p.data, wandb.Image(im), prb]
            #     )
    #         for l, p, im, prb in zip(data[1], concat_predict.cpu().numpy(), data[0],
    #                                 concat_logits.cpu().numpy().tolist()):
    #
    #            try:
    #                table_data.append(
    #                    [l.item(), p, wandb.Image(transform_invert(im, transform_test)), max(prb)]
    #                )
    #            except Exception as e:
    #                print(e)
    #
    # my_table = wandb.Table(data=table_data, columns=["label", "prediction", "image", "truth"])
    # wandb.log({"my_table": my_table})
