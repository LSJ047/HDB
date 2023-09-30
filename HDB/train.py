import os
import sys
from typing import List
import argparse
import click
import numpy as np
import torch
import random
import torchinfo
import torch.distributed as dist
import torchvision.transforms as T
from PIL import ImageFile
from torch import nn, optim
from torch.utils import data, tensorboard
from tqdm import tqdm, trange
from src import configs
from src import data as lc_data
import timer
# from loss import compute_ac_loss_array
import src.model_LSB as model


def plot_bpsp(
        plotter: tensorboard.SummaryWriter, bits: network.Bits,
        inp_size: int, train_iter: int
) -> None:
    """ Plot bpsps for all keys on tensorboard.
        bpsp: bits per subpixel/bits per dimension
        There are 2 bpsps per key:
        self_bpsp: bpsp based on dimension of log-likelihood tensor.
            Measures bits if log-likelihood tensor is final scale.
        scaled_bpsp: bpsp based on dimension of original image.
            Measures how many bits we contribute to the total bpsp.

        param plotter: tensorboard logger
        param bits: bpsp aggregator
        param inp_size: product of dims of original image
        param train_iter: current training iteration
        returns: None
    """
    for key in bits.get_keys():
        plotter.add_scalar(
            f"{key}_self_bpsp", bits.get_self_bpsp(key).item(), train_iter)
        plotter.add_scalar(
            f"{key}_scaled_bpsp",
            bits.get_scaled_bpsp(key, inp_size).item(), train_iter)


def train_loop(
        MSB, LSB, compressor: nn.Module,
        optimizer: optim.Optimizer,  # type: ignore
        train_iter: int, plotter: tensorboard.SummaryWriter,
        plot_iters: int, clip: float,
        pixel_sums,
        mask_alpha=None
):
    """ Training loop for one batch. Computes grads and runs optimizer.
    """
    compressor.train()
    if not args.grad_acc:
        optimizer.zero_grad()
    # ltt
    bits = compressor(MSB, LSB, mask_alpha)
    bpd = bits.mean() / pixel_sums

    acc_steps = 8
    if args.grad_acc:
        bpd = bpd / acc_steps
    bpd.backward()
    if args.grad_acc:
        if train_iter % acc_steps == 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(compressor.parameters(), 5.0, norm_type=2)
            plotter.add_scalar("train/grad_norm", grad_norm, train_iter)
            optimizer.step()
            optimizer.zero_grad()
            plotter.add_scalar(
                "train/bpsp_iter", bpd, train_iter)
        return bpd.cpu().item() * acc_steps
    else:
        grad_norm = torch.nn.utils.clip_grad_norm_(compressor.parameters(), 5.0, norm_type=2)
        optimizer.step()  # 更新网络参数
        plotter.add_scalar("train/grad_norm", grad_norm, train_iter)
        # if train_iter % plot_iters == 0:
        plotter.add_scalar(
            "train/bpsp_iter", bpd, train_iter)
    # Plots gradident norm pre-clipping.

    # plotter.add_scalar(
    #     f"scaled_bpsp",
    #     bpd, train_iter)

    # plot_bpsp(plotter, bpd, inp_size, train_iter)
    return bpd.cpu().item()


def run_eval(
        eval_loader: data.DataLoader, compressor: nn.Module,
        train_iter: int, plotter: tensorboard.SummaryWriter,
        epoch: int,
        lr: float
) -> None:
    """ Runs entire eval epoch. """
    time_accumulator = timer.TimeAccumulator()
    compressor.eval()
    inp_size = 0
    cuda = torch.cuda.is_available()
    with open(os.path.join(args.plot, 'run_eval.txt'), 'a+') as flog:
        with torch.no_grad():
            # BitsKeeper is used to aggregates bits from all eval iterations.
            # bits_keeper = network.Bits()
            bpsp_sum = 0
            for _, DC_MSB, DC_LSB, shape in eval_loader:
                DC_MSB = DC_MSB.cuda()
                DC_LSB = DC_LSB.cuda()
                # for _, inputs,shape,dct_coe_idx,non_zeros_mask, mask in eval_loader:
                with time_accumulator.execute():
                    bits = compressor(DC_MSB, DC_LSB)
                    bpd = bits.cpu() / (shape[0] * shape[1])

                    # bpd_per_prior=[i.cpu()/(shape[0]*shape[1]) for i in bytes_each_slice]

                    # bpd_per_prior.extend(bpd_per_prior_exist)
                    bpsp_sum += bpd.cpu().item()

            eval_bpsp = bpsp_sum / len(eval_loader)
            flog.write(f"Iteration{train_iter} | epoch{epoch}  avg bpsp: {eval_bpsp}\n")
            plotter.add_scalar(
                "eval/avg_bpsp", eval_bpsp, epoch)
            plotter.add_scalar(
                "eval/batch_time", time_accumulator.mean_time_spent(), epoch)
            # plot_bpsp(plotter, bits_keeper, inp_size, train_iter)

            if configs.best_bpsp > eval_bpsp:
                configs.best_bpsp = eval_bpsp
                save(compressor, epoch, plot=args.plot, train_iter=train_iter, filename=f"best.pth", lr=lr)
            return eval_bpsp


def save(compressor,

         epoch: int,
         train_iter: int,
         plot: str,
         filename: str,
         lr:float,
        sampler_indices=None,
         index=None) -> None:
    if multiple_gpus:
        torch.save({
        "nets": compressor.module.state_dict(),
        "sampler_indices": sampler_indices,
        "index": index,
        "epoch": epoch,
            "lr":lr,
        "train_iter": train_iter,
        "best_bpsp": configs.best_bpsp,
    }, os.path.join(plot, filename))
    else:
        torch.save({
            "nets": compressor.state_dict(),
            "sampler_indices": sampler_indices,
            "index": index,
            "epoch": epoch,
             "lr":lr,
            "train_iter": train_iter,
            "best_bpsp": configs.best_bpsp,
        }, os.path.join(plot, filename))


parser = argparse.ArgumentParser(description='PyTorch Discrete Normalizing flows')
parser.add_argument("--name", '-n', type=str, default='HDB',
                    help="path to store tensorboard run data/plots.")
parser.add_argument("--train-path", type=str,
                    default='E:/study/pythonProject/HDB/data/png',
                    help="path to directory of training images.")
parser.add_argument("--eval-path", type=str, default='E:/study/pythonProject/HDB/data/png-test',
                    help="path to directory of eval images.")
parser.add_argument("--use_nonzeros", type=bool, default=False,
                    help="file for training image names.")
parser.add_argument("--use_position_attention", type=bool, default=False,
                    help="file for training image names.")
parser.add_argument("--group_num", type=int, default=64,
                    help="Number of train iterations before plotting data")
parser.add_argument("--code_size", type=int, default=64,
                    help="Number of train iterations before plotting data")
parser.add_argument("--batch", type=int, default=8, help="Batch size for training.")
parser.add_argument("--workers", type=int, default=0,
                    help="Number of worker threads to use in dataloader.")
# parser.add_argument("--plot", type=str,default='../plot/train_GID2_NonNorm',
#               help="path to store tensorboard run data/plots.")
parser.add_argument("--epochs", type=int, default=500,
                    help="Number of epochs to run.")
parser.add_argument("--grad_acc", type=int, default=0,
                    help="Number of epochs to run.")
parser.add_argument("--resblocks", type=int, default=5,
                    help="Number of resblocks to use.")
parser.add_argument("--model", type=str, default="scg",
                    help="Number of resblocks to use.")
parser.add_argument("--n_feats", type=int, default=64,
                    help="Size of feature vector/channel width.")
parser.add_argument("--scale", type=int, default=3,
                    help="Scale of downsampling")
parser.add_argument("--load", type=str,
                    default='K=3_logistic_nfeats64_lr=5e-05_grad_acc=0_batch8_crop=960/epoch0_lr5e-05_batch8_train_bpsp0.297312_eval_bpsp0.2168_crop=960.pth',
                    help="Path to load model")
parser.add_argument("--resume", type=bool, default=False,
                    help="Path to load model")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
parser.add_argument("--eval-iters", type=int, default=0,
                    help="Number of train iterations per evaluation. "
                         "If 0, then evaluate at the end of every epoch.")
parser.add_argument("--lr-epochs", type=int, default=10,
                    help="Number of epochs before multiplying learning rate by 0.75")
parser.add_argument("--plot-iters", type=int, default=1000,
                    help="Number of train iterations before plotting data")
parser.add_argument("--n_mixtures", type=int, default=10,
                    help="Number of clusters in logistic mixture model.")
parser.add_argument("--clip", type=float, default=5,
                    help="Norm to clip by for gradient clipping.")
parser.add_argument("--crop", type=int, default=960,
                    help="Size of image crops in training.")
parser.add_argument("--gd", type=click.Choice(["sgd", "adam", "rmsprop"]), default="adam",
                    help="Type of gd to use.")
parser.add_argument("--group_id", type=int, default=2,
                    help="group id.")
# parser.add_argument("--n_levels", type=int, default=2,
#               help="Number of clusters in logistic mixture model.")
# parser.add_argument("--n_flows", type=float, default=1,
#               help="Norm to clip by for gradient clipping.")
parser.add_argument("--input_size", type=tuple, default=(1, 256, 192),
                    help="Size of image crops in training.")
parser.add_argument("--img_size", type=tuple, default=(960, 960, 1),
                    help="Size of image crops in training.")
parser.add_argument("--n_channels", type=int, default=1,
                    help="Number of clusters in logistic mixture model.")

parser.add_argument("--n_bits", type=int, default=8,
                    help="Number of clusters in logistic mixture model.")
parser.add_argument('--variable_type', type=str, default='discrete',
                    help='variable type of data distribution: discrete/continuous',
                    choices=['discrete', 'continuous'])
parser.add_argument('--distribution_type', type=str, default='logistic',
                    choices=['logistic', 'normal', 'steplogistic', 'laplace'],
                    help='distribution type: logistic/normal')

parser.add_argument('--splitprior_type', type=str, default='none',
                    choices=['none', 'shallow', 'resnet', 'densenet'],
                    help='Type of splitprior. Use \'none\' for no splitprior')
parser.add_argument('--hard_round', dest='hard_round', action='store_true',
                    help='Rounding of translation in discrete models. Weird '
                         'probabilistic implications, only for experimental phase')
parser.add_argument('--no_hard_round', dest='hard_round', action='store_false')
parser.set_defaults(hard_round=True)

parser.add_argument('--round_approx', type=str, default='smooth',
                    choices=['smooth', 'stochastic'])

parser.add_argument('--temperature', default=1.0, type=float,
                    help='Temperature used for BackRound. It is used in '
                         'the the SmoothRound module. '
                         '(default=1.0')
# parser.add_argument('--local_rank', default=-1, type=int,
#                     help='node rank for distributed training')
args = parser.parse_args()
args.cuda = torch.cuda.is_available()


# dist.init_process_group(backend='nccl')
# torch.cuda.set_device(args.local_rank)
def main() -> None:
    torch.manual_seed(42)
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    configs.n_feats = args.n_feats
    # configs.scale = args.scale
    configs.resblocks = args.resblocks
    configs.K = args.n_mixtures
    cuda = args.cuda
    args.plot = f'K{args.n_mixtures}_{args.distribution_type}_{args.model}_nfeats{args.n_feats}_lr={args.lr}_grad_acc{args.grad_acc}_batch{args.batch}_codesize{args.code_size}_crop{args.crop}'
    configs.plot = args.plot
    print(sys.argv)

    device = torch.device('cuda:0') if cuda else torch.device('cpu')
    os.makedirs(args.plot, exist_ok=True)

    compressor = model.Compressor(args, args.n_channels)

    if args.resume:
        checkpoint = torch.load(args.load, map_location=device)
        print(f"Loaded model from {args.load}.")
        print("Epoch:", checkpoint["epoch"])
        if checkpoint.get("best_bpsp") is None:
            print("Warning: best_bpsp not found!")
        else:
            configs.best_bpsp = checkpoint["best_bpsp"]
            print("Best bpsp:", configs.best_bpsp)
        compressor.load_state_dict(checkpoint["nets"])
    else:
        checkpoint = {}

    print(compressor)

    # multiple gpu for training==================
    device_ids = [0]  # 你可以根据实际情况指定要使用的 GPU 设备 ID

    if multiple_gpus:
        compressor = nn.DataParallel(compressor.cuda(), device_ids=device_ids, output_device=device_ids[0])
    else:
        compressor = compressor.cuda()
    # 将模型移到多个 GPU 上并包装成 DataParallel
    # compressor = torch.nn.parallel.DistributedDataParallel(compressor, device_ids=[args.local_rank])

    # compressor = nn.DataParallel(compressor, device_ids=device_ids)

    optimizer: optim.Optimizer  # type: ignore
    if args.gd == "adam":
        # optimizer = optim.Adam(compressor.parameters(), lr=args.lr, weight_decay=0)
        optimizer = optim.Adam(
            compressor.parameters(), lr=args.lr,
            betas=(0.9, 0.999), eps=1e-08, weight_decay=0
        )
    elif args.gd == "sgd":
        optimizer = optim.SGD(compressor.parameters(), lr=args.lr,
                              momentum=0.9, nesterov=True)
    elif args.gd == "rmsprop":
        optimizer = optim.RMSprop(  # type: ignore
            compressor.parameters(), lr=args.lr)
    else:
        raise NotImplementedError(args.gd)

    # scheduler=optim.lr_scheduler.StepLR(optimizer,step_size=1,gamma=0.5)
    # lr_scheduler = optim.lr_scheduler.StepLR(optimizer, args.lr_epochs, gamma=0.75)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=30, factor=0.5, min_lr=3e-6)
    starting_epoch = checkpoint.get("epoch") or 0
    starting_epoch = 0

    train_dataset = lc_data.ImageFolder(
        args.train_path,
        [filename.strip() for filename in os.listdir(args.train_path)],
        args.scale,
        T.Compose([
            T.RandomHorizontalFlip(),
            # T.RandomCrop(args.crop),  #
            T.CenterCrop(args.crop)
        ]),
        # use_non_zeros_ac=args.use_nonzeros,
        # encode_block_size=args.code_size,
        # ac_coe_num=args.group_num,
        # ac_coe_channle_start=1 if args.group_num==63 else 0
    )
    dataset_index = checkpoint.get("index") or 0
    # train_sampler = lc_data.PreemptiveRandomSampler(
    #     checkpoint.get("sampler_indices") or torch.randperm(
    #         len(train_dataset)).tolist(),
    #     dataset_index,
    # )
    train_sampler = lc_data.PreemptiveRandomSampler(torch.randperm(
        len(train_dataset)).tolist(),
                                                    dataset_index,
                                                    )
    # train_sampler = torch.utils.data.distributed.DistributedSampler(
    #     train_dataset
    # )
    # train_loader = data.DataLoader(
    #     train_dataset, batch_size=batch, sampler=train_sampler,
    #     num_workers=workers, drop_last=True,
    # )
    train_loader = data.DataLoader(
        train_dataset, batch_size=args.batch,
        num_workers=args.workers, drop_last=True,
    )

    print(f"Loaded training dataset with {len(train_loader)} batches "
          f"and {len(train_loader.dataset)} images")
    train_loader = tqdm(train_loader, total=len(train_loader), leave=False)

    ##
    eval_dataset = lc_data.ImageFolder(
        args.eval_path, [filename.strip() for filename in os.listdir(args.eval_path)],
        args.scale,
        T.Compose([
            T.RandomHorizontalFlip(),
            T.CenterCrop(args.crop)
        ]),
        # use_non_zeros_ac=args.use_nonzeros,
        # encode_block_size=args.code_size,
        # ac_coe_num=args.group_num,
        # ac_coe_channle_start=1 if args.group_num==63 else 0
    )
    # 随机选择指定数量的图片
    num_selected_images = 20  # 指定要选择的图片数量
    random_indices = random.sample(range(len(eval_dataset)), num_selected_images)
    selected_images = [eval_dataset[i] for i in random_indices]

    eval_loader = data.DataLoader(
        selected_images, batch_size=1, shuffle=False,
        num_workers=args.workers, drop_last=False,
    )
    print(f"Loaded eval dataset with {len(eval_loader)} batches "
          f"and {len(eval_dataset)} images")
    eval_loader = tqdm(eval_loader, total=len(eval_loader), leave=False)

    # for _ in range(starting_epoch):
    #     lr_scheduler.step()  # type: ignore

    train_iter = 0  # checkpoint.get("train_iter") or 0
    # eval_iters=args.eval_iters
    # if args.eval_iters == 0:
    #     eval_iters = len(train_loader)

    for epoch in range(starting_epoch, args.epochs):
        with tensorboard.SummaryWriter(args.plot) as plotter:
            bpd_sum = 0.
            for _, MSB, LSB, shape in train_loader:
                MSB = MSB.cuda()
                LSB = LSB.cuda()
                train_iter += 1
                batch_size = args.batch
                pixel_sums = args.img_size[0] * args.img_size[1] * args.img_size[2]

                if args.use_position_attention:
                    # todo: position attention
                    pass
                # torchinfo.summary(compressor,input_data=[MSB,LSB])
                bpd = train_loop(MSB, LSB, compressor, optimizer, train_iter,
                                 plotter, args.plot_iters, args.clip, pixel_sums)
                bpd_sum += bpd
                # print(f'\n{train_iter}/{epoch}:{bpd}\t loss_nn:{NN_loss}')
                print(f'\n{train_iter} | {epoch}\t{bpd:.5f}\t')
                # 更新 tqdm 进度条
                train_loader.set_description(f'Training - bpd: {train_iter} | {epoch}\t{bpd:.4f}')
                # Increment dataset_index before checkpointing because
                # dataset_index is starting index of index of the FIRST
                # unseen piece of data.
                dataset_index += batch_size
                plotter.add_scalar(
                    "train/bpd",
                    bpd,  # type: ignore
                    train_iter)
                current_lr = optimizer.param_groups[0]['lr']
                # eval_bpsp=run_eval(
                #         eval_loader, compressor, train_iter,
                #         plotter, epoch,current_lr)
                # plotter.add_scalar(
                #         "train/loss_nn",
                #         NN_loss,  # type: ignore
                #         epoch)
                # if train_iter % args.plot_iters == 0:
                #     plotter.add_scalar(
                #         "train/lr",
                #         lr_scheduler.get_lr()[0],  # type: ignore
                #         epoch)
                #     save(compressor, train_sampler.indices, dataset_index,
                #          epoch, train_iter, args.plot, "train.pth")
                # if train_iter == 0:
                # if train_iter % eval_iters == 0:
                #     eval_bpsp=run_eval(
                #         eval_loader, compressor, train_iter,
                #         plotter, epoch)
            eval_bpsp = run_eval(
                eval_loader, compressor, train_iter,
                plotter, epoch, current_lr)
            train_avg_bpd = bpd_sum / len(train_loader)
            # save(compressor,epoch, train_iter, args.plot, f"epoch={epoch}_nonzeros={args.use_nonzeros}_posAttention={args.use_position_attention}_group={args.group_num}_lr{args.lr}_crop={args.crop}.pth",\
            # train_sampler.indices, train_sampler.index)
            lr_scheduler.step(eval_bpsp)  # type: ignore
            # save(compressor, epoch, train_iter, args.plot, f"epoch{epoch}_lr{optimizer.param_groups[0]['lr']}_batch{args.batch}_crop={args.crop}.pth",optimizer.param_groups[0]['lr'],
            #  train_sampler.indices, train_sampler.index)
            if train_iter % args.plot_iters == 0:
                plotter.add_scalar(
                    "train/lr",
                    optimizer.param_groups[0]['lr'],  # type: ignore
                    epoch)
            dataset_index = 0
            plotter.add_scalar(
                "train/avg_bpsp",
                train_avg_bpd,  # type: ignore
                epoch)
            # plotter.add_scalar(
            #             "eval/avg_bpsp",
            #            train_avg_bpd,  # type: ignore
            #             epoch)

        # with tensorboard.SummaryWriter(args.plot) as plotter:
        #     run_eval(eval_loader, compressor, train_iter,
        #              plotter, args.epoch)
        if epoch % 10 == 0:
            save(compressor, epoch, train_iter, args.plot,
                 f"epoch{epoch}_lr{optimizer.param_groups[0]['lr']}_batch{args.batch}_train_bpsp{train_avg_bpd:4f}_eval_bpsp{eval_bpsp:.4f}_crop={args.crop}.pth",
                 optimizer.param_groups[0]['lr'],
                 train_sampler.indices, train_sampler.index)
    print("training done")


# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
if __name__ == "__main__":
    # type: ignore
    multiple_gpus = False
    main()
