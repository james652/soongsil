#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, argparse, torch, torch.nn as nn, torch.backends.cudnn as cudnn, torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from utils.misc import my_makedir, Logger, AverageMeter, setup_seed
from utils.test_helpers import test
from utils.model import resnet18

# ---- 스케줄러(로컬 정의) ----
def adjust_learning_rate(optimizer, epoch, args):
    lr = args.lr
    if epoch >= 36:
        lr *= 0.01
    elif epoch >= 31:
        lr *= 0.1
    for pg in optimizer.param_groups:
        pg['lr'] = lr

DATASETS = {
    "cifar10":  {"num_classes":10,  "mean":[0.4914,0.4822,0.4465], "std":[0.2470,0.2435,0.2616]},
    "cifar100": {"num_classes":100, "mean":[0.5071,0.4867,0.4408], "std":[0.2675,0.2565,0.2761]},
    "mnist":    {"num_classes":10,  "mean":[0.1307],               "std":[0.3081]},
}

def build_tf(name):
    if name == "mnist":
        m,s = DATASETS[name]["mean"], DATASETS[name]["std"]
        tr = transforms.Compose([
            transforms.Resize((32,32)), transforms.ToTensor(), transforms.Normalize(m,s),
            transforms.Lambda(lambda x: x.expand(3, *x.shape[1:])),  # 1ch -> 3ch
        ])
        te = transforms.Compose([
            transforms.Resize((32,32)), transforms.ToTensor(), transforms.Normalize(m,s),
            transforms.Lambda(lambda x: x.expand(3, *x.shape[1:])),
        ])
    else:
        m,s = DATASETS[name]["mean"], DATASETS[name]["std"]
        tr = transforms.Compose([
            transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
            transforms.ToTensor(), transforms.Normalize(m,s),
        ])
        te = transforms.Compose([transforms.ToTensor(), transforms.Normalize(m,s)])
    return tr, te

def load_ds(name, root, tr_tf, te_tf):
    if name == "cifar10":
        return datasets.CIFAR10(root, True, tr_tf, download=True), datasets.CIFAR10(root, False, te_tf, download=True)
    if name == "cifar100":
        return datasets.CIFAR100(root, True, tr_tf, download=True), datasets.CIFAR100(root, False, te_tf, download=True)
    if name == "mnist":
        return datasets.MNIST(root, True, tr_tf, download=True), datasets.MNIST(root, False, te_tf, download=True)
    raise ValueError(f"Unsupported dataset: {name}")

def train_epoch(model, loader, criterion, opt, device, clip=False, clipv=1.0):
    model.train()
    losses, accm = AverageMeter('Loss',':.4e'), AverageMeter('Acc@1',':6.2f')
    for x,y in loader:
        x,y = x.to(device), y.to(device)
        opt.zero_grad(); out = model(x); loss = criterion(out,y); loss.backward()
        if clip: torch.nn.utils.clip_grad_norm_(model.parameters(), clipv, 2.0)
        opt.step()
        losses.update(loss.item(), y.size(0))
        accm.update((out.argmax(1)==y).float().mean().item(), y.size(0))
    return losses.avg, accm.avg

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["cifar10","cifar100","mnist"])
    ap.add_argument("--data-root", default="./data")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--test-batch-size", type=int, default=256)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--use-bn", action="store_true")
    ap.add_argument("--group-norm", type=int, default=32)
    ap.add_argument("--clip-gradnorm", action="store_true")
    ap.add_argument("--clipvalue", type=float, default=1.0)
    ap.add_argument("--out-root", default="/home/jun/work/soongsil/Agent/checkpoints_base_bn_Agent")
    args = ap.parse_args()

    setup_seed(20)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cudnn.benchmark = True

    out_dir = os.path.join(args.out_root, args.dataset)
    os.makedirs(out_dir, exist_ok=True)

    # 로그는 에이전트 경로에 기록
    import sys
    sys.stdout = Logger(os.path.join(out_dir, 'log.txt'), mode='a')
    print(args)

    tr_tf, te_tf = build_tf(args.dataset)
    tr_ds, te_ds = load_ds(args.dataset, args.data_root, tr_tf, te_tf)
    tr_ld = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True,  num_workers=args.workers, pin_memory=True)
    te_ld = DataLoader(te_ds, batch_size=args.test_batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    num_classes = DATASETS[args.dataset]["num_classes"]
    norm_layer = None if args.use_bn else (lambda p: nn.GroupNorm(args.group_norm, p))
    net = resnet18(num_classes=num_classes, norm_layer=norm_layer).to(device)
    net = torch.nn.DataParallel(net)

    opt = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    crit = nn.CrossEntropyLoss().to(device)

    for epoch in range(1, args.epochs + 1):
        adjust_learning_rate(opt, epoch, args)
        tr_loss, tr_acc = train_epoch(net, tr_ld, crit, opt, device, args.clip_gradnorm, args.clipvalue)
        err_cls = test(te_ld, net)
        print(f"Epoch:{epoch:02d}\t train_loss:{tr_loss:.4f}\t train_acc:{tr_acc*100:.2f}%\t err_cls:{err_cls:.4f}")

        # 10단위 체크포인트 저장 (online_accu_train.py에서 요구하는 키 포함)
        if epoch % 10 == 0:
            torch.save(
                {"epoch": epoch, "args": args, "err_cls": err_cls, "optimizer": opt.state_dict(), "net": net.state_dict()},
                os.path.join(out_dir, f"epoch{epoch}.pth")
            )

if __name__ == "__main__":
    main()
