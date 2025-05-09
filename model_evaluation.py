from torch.utils.tensorboard import SummaryWriter

import Arguments
from MaskedAutoencoder import MaskedAutoencoder
from mae import engine_pretrain, engine_finetune
import timm.optim.optim_factory as optim_factory
import torch
from torch import nn
import mae.util.misc as misc
from helpers import *
from sklearn.model_selection import KFold
from functools import partial
import numpy as np
import mae.util.lr_decay as lrd
from mae.models_vit import VisionTransformer
from mae.util.pos_embed import interpolate_pos_embed
from timm.models.layers import trunc_normal_
from torchmetrics.classification import MulticlassAccuracy

def transform_ae(model, X):
    _, X_patches, _ = model(X.unsqueeze(0))
    return model.unpatchify(X_patches)

def autoencoder_validate(model, loader, device):
    loss = 0
    criterion = nn.CrossEntropyLoss().to(device)
    with torch.no_grad():
        for X, y in loader:
            y_pred = model(X.to(device))
            current_loss = criterion(y, y_pred)
            loss = loss + current_loss
    return loss

def model_validate(model, loader, criterion, device):
    loss = 0
    with torch.no_grad():
        for X, y in loader:
            y_pred = model(X.to(device))
            current_loss = criterion(y_pred, y.to(device))
            loss = loss + current_loss
    return loss

def short_dirname(trial):
    return "trial_" + str(trial.trial_id)

def select_hyperparams():
    args = Arguments.get_pretrain_args().parse_args(args=[])
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    search_space = {
        "embed_dim": [256, 512, 1024],
        "depth": [4, 8, 16],
        "num_heads": [4, 8, 16]
    }

    for embed_dim in search_space["embed_dim"]:
        for depth in search_space["depth"]:
            for num_heads in search_space["num_heads"]:
                model, loss = model_trial({"embed_dim": embed_dim, "num_heads": num_heads, "depth": depth})
                with open(f"results/embed-{embed_dim}_depth-{depth}_heads-{num_heads}.txt", "w") as f:
                    f.write(str(loss.cpu().detach().item()))
                torch.save(model, f"models/embed-{embed_dim}_depth-{depth}_heads-{num_heads}.pt")

    

def autoencoder_trial(config):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args = Arguments.get_pretrain_args().parse_args(args=[])
    for k, v in config.items():
        setattr(args, k, v)

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256
    model = MaskedAutoencoder(
        patch_size=args.patch_size,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads
    ).to(device)
    
    param_groups = optim_factory.add_weight_decay(model, args.weight_decay)
    scaler = misc.NativeScalerWithGradNormCount()
    dataset = get_train_dataset()
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    loss = 0
    for train_idx, valid_idx in kfold.split(dataset):
        log_writer = SummaryWriter(log_dir=args.log_dir)
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
        valid_subsampler = torch.utils.data.SubsetRandomSampler(valid_idx)

        train_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, sampler=train_subsampler)
        valid_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, sampler=valid_subsampler)

        optimizer = torch.optim.AdamW(param_groups, args.lr, betas=(0.9, 0.95))
        for i in range(args.epochs):
            engine_pretrain.train_one_epoch(model, train_loader, optimizer, device, i, scaler, log_writer=None, args=args)
        loss = loss + autoencoder_validate(model, valid_loader, device)

        if args.output_dir:
            misc.save_model(
                args=args, model=model, model_without_ddp=model, optimizer=optimizer,
                loss_scaler=scaler, epoch=args.epochs)


    return model, loss

def full_trial(config):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    pretrain_args = Arguments.get_pretrain_args().parse_args(args=[])
    finetune_args = Arguments.get_finetune_args().parse_args(args=[])
    for k, v in config.items():
        setattr(pretrain_args, k, v)
        setattr(finetune_args, k, v)

    print(pretrain_args)
    print(finetune_args)
    eff_batch_size = pretrain_args.batch_size * pretrain_args.accum_iter * misc.get_world_size()
    if pretrain_args.lr is None:  # only base_lr is specified
        pretrain_args.lr = pretrain_args.blr * eff_batch_size / 256

    
    dataset = get_train_dataset()
    print(len(dataset))
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    loss = 0
    for train_idx, valid_idx in kfold.split(dataset):
        ae = MaskedAutoencoder(
            patch_size=pretrain_args.patch_size,
            embed_dim=pretrain_args.embed_dim,
            depth=pretrain_args.depth,
            num_heads=pretrain_args.num_heads
        ).to(device)
        log_writer = SummaryWriter(log_dir=pretrain_args.log_dir)
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
        valid_subsampler = torch.utils.data.SubsetRandomSampler(valid_idx)

        train_loader = torch.utils.data.DataLoader(dataset, batch_size=pretrain_args.batch_size, sampler=train_subsampler)
        valid_loader = torch.utils.data.DataLoader(dataset, batch_size=pretrain_args.batch_size, sampler=valid_subsampler)

        param_groups = optim_factory.add_weight_decay(ae, pretrain_args.weight_decay)
        scaler = misc.NativeScalerWithGradNormCount()
        optimizer = torch.optim.AdamW(param_groups, pretrain_args.lr, betas=(0.9, 0.95))
        for i in range(pretrain_args.epochs):
            engine_pretrain.train_one_epoch(ae, train_loader, optimizer, device, i, scaler, log_writer=None, args=pretrain_args)

        ae = ae.to("cpu")
        model = VisionTransformer(img_size=(4096, 512), patch_size=finetune_args.patch_size, in_chans=1, num_classes=2, embed_dim=finetune_args.embed_dim, depth=finetune_args.depth, num_heads=finetune_args.num_heads, mlp_ratio=4, qkv_bias=False).to(device)
        state_dict = model.state_dict()
        ae_state = ae.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in ae_state and ae_state[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del ae[k]
        # interpolate position embedding
        interpolate_pos_embed(model, ae_state)

        # load pre-trained model
        msg = model.load_state_dict(ae_state, strict=False)
        print(msg.missing_keys)
        
        if finetune_args.lr is None:  # only base_lr is specified
            finetune_args.lr = finetune_args.blr * eff_batch_size / 256

        # manually initialize fc layer
        trunc_normal_(model.head.weight, std=2e-5)
        criterion = nn.CrossEntropyLoss().to(device)

        
        param_groups = lrd.param_groups_lrd(model, finetune_args.weight_decay,
            no_weight_decay_list=model.no_weight_decay(),
            layer_decay=finetune_args.layer_decay)
        optimizer = torch.optim.AdamW(param_groups, lr=finetune_args.lr, betas=(0.9, 0.95))

        for i in range(1, finetune_args.epochs):
            engine_finetune.train_one_epoch(model, criterion, train_loader, optimizer, device, i, scaler, args=finetune_args)
        loss = loss + model_validate(model, valid_loader, criterion, device)
    
    return model, loss


def model_trial(config):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    finetune_args = Arguments.get_finetune_args().parse_args(args=[])
    for k, v in config.items():
        setattr(finetune_args, k, v)


    
    dataset = get_train_dataset()
    loss = 0
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    for train_idx, valid_idx in kfold.split(dataset):
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
        valid_subsampler = torch.utils.data.SubsetRandomSampler(valid_idx)

        train_loader = torch.utils.data.DataLoader(dataset, batch_size=finetune_args.batch_size, sampler=train_subsampler)
        valid_loader = torch.utils.data.DataLoader(dataset, batch_size=finetune_args.batch_size, sampler=valid_subsampler)

        model = VisionTransformer(img_size=(4096, 512), patch_size=finetune_args.patch_size, in_chans=1, num_classes=2, embed_dim=finetune_args.embed_dim, depth=finetune_args.depth, num_heads=finetune_args.num_heads, mlp_ratio=4, qkv_bias=False).to(device)

        eff_batch_size = finetune_args.batch_size * finetune_args.accum_iter * misc.get_world_size()
        if finetune_args.lr is None:  # only base_lr is specified
            finetune_args.lr = finetune_args.blr * eff_batch_size / 256

        
        criterion = nn.CrossEntropyLoss().to(device)

        
        param_groups = lrd.param_groups_lrd(model, finetune_args.weight_decay,
            no_weight_decay_list=model.no_weight_decay(),
            layer_decay=finetune_args.layer_decay)
        scaler = misc.NativeScalerWithGradNormCount()
        optimizer = torch.optim.AdamW(param_groups, lr=finetune_args.lr)

        for i in range(1, finetune_args.epochs):
            engine_finetune.train_one_epoch(model, criterion, train_loader, optimizer, device, i, scaler, args=finetune_args)
        loss = loss + model_validate(model, valid_loader, criterion, device)
    
    return model, loss
    