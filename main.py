from models.resnet import ResNet_50 as get_model
from utilis.data import dataset, dataloader
from utilis.output import write_output
from utilis.averageMeter import AverageMeter
from utilis.lr_scheduler import get_lr_scheduler
import deepspeed
from torch import nn
import torch
import argparse
import datetime
import time
import os


def add_argument():
    parser = argparse.ArgumentParser(description="main")
    parser.add_argument('--local_rank', type=int, default=-1, help='local rank passed from districuted launcher')
    parser.add_argument('--data_dir', type=str, help='data dir which contains train and val sub dir')
    parser.add_argument('--out_dir', type=str, help='output dir')

    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    return args


def train_epoch(model_engine, train_loader, criterions, lr_scheduler):
    for batch in train_loader:
        input, label = batch
        input = input.to(model_engine.local_rank)
        label = label.to(model_engine.local_rank)

        output = model_engine(input)

        loss = 0
        for criterion in criterions:
            loss += criterion["func"](output, label) * criterion["weight"]

        model_engine.backward(loss)

        model_engine.step()

    lr_scheduler.step()


def val_epoch(model_engine, val_loader, criterions, outdir):
    avgLoss = AverageMeter()
    avgAcc = AverageMeter()

    with torch.no_grad():
        for batch in val_loader:
            input, label = batch
            input = input.to(model_engine.local_rank)
            label = label.to(model_engine.local_rank)
    
            output = model_engine(input)

            loss = 0
            for criterion in criterions:
                loss += criterion["func"](output, label) * criterion["weight"]

            _, pred = torch.max(output, dim=1)
            correct = (pred == label)

            avgLoss.update(loss, label.size(0))
            avgAcc.update(correct)

    write_output(outdir, model_engine.local_rank, "loss: " + str(avgLoss))
    write_output(outdir, model_engine.local_rank, "accuracy: " + str(avgAcc))


def main():
    args = add_argument()
    model = get_model()

    train_dst = dataset(args.data_dir)

    model_engine, optimizer, train_loader, _ = deepspeed.initialize(args=args, model=model, model_parameters=model.parameters(), training_data=train_dst)
    # criterion = nn.CrossEntropyLoss()
    # criterions = [{"func": nn.CrossEntropyLoss(), 'weight': 0.2}, {"func": nn.KLDivLoss(), "weight": 0.8}]
    criterions = [{"func": nn.CrossEntropyLoss(), 'weight': 1}]

    lr_scheduler = get_lr_scheduler(optimizer)

    outdir = os.path.join(args.out_dir, datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    if model_engine.local_rank == 0:
        val_loader = dataloader(args.data_dir, int(model_engine.train_batch_size() / model_engine.world_size), False, False)

    if model_engine.local_rank == 0:
        write_output(outdir, model_engine.local_rank, f"before train")
        write_output(outdir, model_engine.local_rank, model.conv1.weight[0, 0])
        val_epoch(model_engine, val_loader, criterions, outdir)
        write_output(outdir, model_engine.local_rank, "")

    epoch = 100
    for i in range(epoch):
        write_output(outdir, model_engine.local_rank, f"learning rate of epoch {i}: {model_engine.get_lr()}")
        startTime = time.time()
        train_epoch(model_engine, train_loader, criterions, lr_scheduler)
        endTime = time.time()
        runTime = endTime - startTime
        write_output(outdir, model_engine.local_rank, f"running time of epoch {i} / {epoch}: {runTime}")
        if model_engine.local_rank == 0:
            write_output(outdir, model_engine.local_rank, f"epoch {i} / {epoch}")
            write_output(outdir, model_engine.local_rank, model.conv1.weight[0, 0])
            val_epoch(model_engine, val_loader, criterions, outdir)
            write_output(outdir, model_engine.local_rank, "")


if __name__ == '__main__':
    main()
