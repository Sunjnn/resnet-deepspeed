from torch.optim.lr_scheduler import SequentialLR, CosineAnnealingLR, LinearLR, ConstantLR

def get_lr_scheduler(optimizer):
    """5 epoch warmup and cosine scheduling to 600th epoch
    learning rate scheduler from https://deci.ai/blog/resnet50-how-to-achieve-sota-accuracy-on-imagenet/"""

    start_factor = 0.2
    total_iters=4
    T_max = 5
    verbose = False

    scheduler1 = LinearLR(optimizer, start_factor=start_factor, total_iters=total_iters, verbose=verbose)
    scheduler2 = CosineAnnealingLR(optimizer, T_max=T_max, verbose=verbose)
    scheduler3 = ConstantLR(optimizer, factor=1, total_iters=10)
    return SequentialLR(optimizer, [scheduler1, scheduler2, scheduler3], milestones=[5, 600])
