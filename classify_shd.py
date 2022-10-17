import torch
import torch.nn.functional as F
from torch.cuda import amp
from spikingjelly.activation_based import functional, neuron
from torch.utils.data import DataLoader
import time
import argparse
import datetime
import model
import data
import numpy as np


torch.backends.cudnn.benchmark = True
_seed_ = 202208
torch.manual_seed(_seed_)  
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(_seed_)


def load_data(args):
    if args.dataset == "SHD":
        train_ds = data.SHD(train=True, dt=args.dt, T=args.T)
        test_ds = data.SHD(train=False, dt=args.dt, T=args.T)
        train_dl = DataLoader(train_ds, shuffle=True, batch_size=args.batch_size, pin_memory=True)
        test_dl = DataLoader(test_ds, shuffle=False, batch_size=args.batch_size, pin_memory=True)
    return train_dl, test_dl


def main():
    # python ./classify_shd.py -dataset SHD -T 15 -dt 60 -device cuda:0 -batch_size 256 -epochs 1000 -opt adam -lr 0.0001 -loss MSE

    parser = argparse.ArgumentParser(description='Classify SHD')
    parser.add_argument("-dataset",type=str,default="SHD")
    parser.add_argument("-batch_size",type=int,default=256) 
    parser.add_argument("-T",type=int,default=15,help='simulating time-steps') 
    parser.add_argument("-dt",type=int,default=60,help='frame time-span') 
    parser.add_argument('-device', default='cuda:0', help='device')
    parser.add_argument('-epochs', default=200, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-amp', default=True, type=bool, help='automatic mixed precision training')
    parser.add_argument('-cupy', default=True, type=bool, help='use cupy backend')
    parser.add_argument('-opt', default="adam", type=str, help='use which optimizer. SDG or Adam')
    parser.add_argument('-momentum', default=0.9, type=float, help='momentum for SGD')
    parser.add_argument('-lr', default=0.0001, type=float, help='learning rate')
    parser.add_argument('-loss', default="MSE", type=str, help='loss function')

    args = parser.parse_args()
    print(args)

    net = model.SHD_STSC()

    functional.set_step_mode(net, 'm')
    if args.cupy:
        functional.set_backend(net, 'cupy', instance=neuron.LIFNode)


    print(net)
    net.to(args.device)

    train_data_loader, test_data_loader = load_data(args)  


    scaler = None
    if args.amp:
        scaler = amp.GradScaler()

    start_epoch = 0
    max_test_acc = -1

    optimizer = None
    if args.opt == 'sgd':
        optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum)
    elif args.opt == 'adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    else:
        raise NotImplementedError(args.opt)

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 1000)

    for epoch in range(start_epoch, args.epochs):
        start_time = time.time()
        net.train()
        train_loss = 0
        train_acc = 0
        train_samples = 0
        for frame, label in train_data_loader:
            optimizer.zero_grad()
            frame = frame.to(args.device)
            frame = frame.transpose(0, 1)  # [B, T, N] -> [T, B, N]
            label = label.to(args.device)
            label_onehot = F.one_hot(label.to(torch.int64), 20).float()

            if scaler is not None:
                with amp.autocast():
                    if args.loss == "MSE":
                        out_fr = net(frame).mean(0)
                        loss = F.mse_loss(out_fr, label_onehot)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                if args.loss == "MSE":
                    out_fr = net(frame).mean(0)
                    loss = F.mse_loss(out_fr, label_onehot)
                loss.backward()
                optimizer.step()

            train_samples += label.numel()
            train_loss += loss.item() * label.numel()
            train_acc += (out_fr.argmax(1) == label).float().sum().item()

            functional.reset_net(net)

        train_time = time.time()
        train_speed = train_samples / (train_time - start_time)
        train_loss /= train_samples
        train_acc /= train_samples

        lr_scheduler.step()

        net.eval()
        test_loss = 0
        test_acc = 0
        test_samples = 0
        with torch.no_grad():
            for frame, label in test_data_loader:
                frame = frame.to(args.device)
                frame = frame.transpose(0, 1)   # [B, T, N] -> [T, B, N]
                label = label.to(args.device)
                label_onehot = F.one_hot(label.to(torch.int64), 20).float()
                out_fr = None
                if args.loss == "MSE":
                    out_fr = net(frame).mean(0)
                    loss = F.mse_loss(out_fr, label_onehot)
                test_samples += label.numel()
                test_loss += loss.item() * label.numel()
                test_acc += (out_fr.argmax(1) == label).float().sum().item()
                functional.reset_net(net)
        test_time = time.time()
        test_speed = test_samples / (test_time - train_time)
        test_loss /= test_samples
        test_acc /= test_samples

        if test_acc > max_test_acc:
            max_test_acc = test_acc

        print(args)
        print(f'epoch = {epoch}, train_loss ={train_loss: .4f}, train_acc ={train_acc: .4f}, test_loss ={test_loss: .4f}, test_acc ={test_acc: .4f}, max_test_acc ={max_test_acc: .4f}')
        print(f'train speed ={train_speed: .4f} images/s, test speed ={test_speed: .4f} images/s')
        print(f'escape time = {(datetime.datetime.now() + datetime.timedelta(seconds=(time.time() - start_time) * (args.epochs - epoch))).strftime("%Y-%m-%d %H:%M:%S")}\n')


if __name__ == '__main__':
    main()