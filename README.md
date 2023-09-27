# resnet-deepspeed

Using deepspeed training resnet on imagenet on a system with multiple GPUs.

## run data parallel on single node

Specify the paths in the command below,
and run it.

``` bash
nohup deepspeed main.py --deepspeed --deepspeed_config /path/to/ds_config.json --data_dir /path/to/imagenet --out_dir /path/to/output > /path/to/out.log 2> /path/to/err.log &
```

## run data parallel on multiple nodes

Set IF names that nccl will use.

``` bash
export NCCL_SOCKET_IFNAME="all-names"
```

Specify the paths and nodes in the command below,
and run it.

``` bash
nohup deepspeed --hostfile=/path/to/hostfile --include="node-IP1:device-idxs[@node-IP2:devices-idxs]" main.py --deepspeed --deepspeed_config /path/to/ds_config.json --data_dir /path/to/imagenet --out_dir /path/to/output > /path/to/out.log 2> /path/to/err.log &
```
