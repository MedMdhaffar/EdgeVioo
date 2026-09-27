from torch.utils.data import DataLoader
import torch.optim as optim
import torch
import time
import numpy as np
import random
import os
from pathlib import Path
from model import Model
from dataset import Dataset
from train import train
from test import test
import option


def init_wandb(args, model):
    if args.wandb_mode == 'disabled':
        return None, None

    try:
        import wandb
    except ImportError as error:
        raise RuntimeError(
            'Weights & Biases logging was requested, but wandb is not installed. '
            'Install it with: pip install wandb'
        ) from error

    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name or args.model_name,
        mode=args.wandb_mode,
        config=vars(args),
        job_type='train',
        tags=[args.dataset_name, args.modality],
    )
    if args.wandb_watch:
        run.watch(model, log='all', log_freq=100)
    return wandb, run


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def make_profiler(args):
    if args.profile_steps <= 0:
        raise ValueError('--profile-steps must be positive')
    profile_dir = Path(args.profile_dir)
    profile_dir.mkdir(parents=True, exist_ok=True)
    return torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(
            wait=1,
            warmup=1,
            active=args.profile_steps,
            repeat=1,
        ),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(
            str(profile_dir), worker_name='xdviodet'
        ),
        record_shapes=True,
        profile_memory=True,
        with_flops=True,
    )

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    # setup_seed(2333)
    args = option.parser.parse_args()
    device = torch.device("cuda")
    train_loader = DataLoader(Dataset(args, test_mode=False),
                              batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True,
                              drop_last=True)
    test_loader = DataLoader(Dataset(args, test_mode=True),
                              batch_size=5, shuffle=False,
                              num_workers=args.workers, pin_memory=True)
    torch.set_float32_matmul_precision("high")

    device = torch.device('cuda:{}'.format(args.gpus) if args.gpus != '-1' else 'cpu')
    eager_model = Model(args).to(device)
    compiled_model = torch.compile(eager_model) 
    
    wandb, wandb_run = init_wandb(args, eager_model)

    for name, value in eager_model.named_parameters():
        print(name)
    approximator_param = list(map(id, eager_model.approximator.parameters()))
    approximator_param += list(map(id, eager_model.conv1d_approximator.parameters()))
    base_param = filter(lambda p: id(p) not in approximator_param, eager_model.parameters())

    if not os.path.exists('./ckpt'):
        os.makedirs('./ckpt')
    optimizer = optim.Adam([{'params': base_param},
                            {'params': eager_model.approximator.parameters(), 'lr': args.lr / 2},
                            {'params': eager_model.conv1d_approximator.parameters(), 'lr': args.lr / 2},
                            ],
                            lr=args.lr, weight_decay=0.000)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10], gamma=0.1)
    criterion = torch.nn.BCELoss()

    is_topk = True
    gt = np.load(args.gt)
    pr_auc, pr_auc_online = test(test_loader, eager_model, device, gt)
    print('Random initalization: offline pr_auc:{0:.4}; online pr_auc:{1:.4}\n'.format(pr_auc, pr_auc_online))
    if wandb_run is not None:
        wandb_run.log({
            'epoch': -1,
            'test/offline_pr_auc': pr_auc,
            'test/online_pr_auc': pr_auc_online,
        })

    try:
        for epoch in range(args.max_epoch):
            st = time.time()
            if args.profile and epoch == 0:
                with make_profiler(args) as profiler:
                    train_metrics = train(
                        train_loader, compiled_model, optimizer, criterion,
                        device, is_topk, profiler=profiler
                    )
                print(profiler.key_averages().table(
                    sort_by='self_cuda_time_total', row_limit=25
                ))
                print('Profiler trace written to: {}'.format(
                    Path(args.profile_dir).resolve()
                ))
            else:
                train_metrics = train(
                    train_loader, compiled_model, optimizer, criterion,
                    device, is_topk
                )
            if epoch % 2 == 0 and not epoch == 0:
                torch.save(eager_model.state_dict(), './ckpt/'+args.model_name+'{}.pkl'.format(epoch))

            pr_auc, pr_auc_online = test(test_loader, eager_model, device, gt)
            epoch_seconds = time.time() - st
            print(
                'Epoch {0}/{1}: loss:{2:.4}; offline pr_auc:{3:.4}; '
                'online pr_auc:{4:.4}; time:{5:.1f}s\n'.format(
                    epoch,
                    args.max_epoch,
                    train_metrics['train/total_loss'],
                    pr_auc,
                    pr_auc_online,
                    epoch_seconds,
                )
            )

            if wandb_run is not None:
                wandb_run.log({
                    'epoch': epoch,
                    **train_metrics,
                    'test/offline_pr_auc': pr_auc,
                    'test/online_pr_auc': pr_auc_online,
                    'train/learning_rate': optimizer.param_groups[0]['lr'],
                    'system/epoch_seconds': epoch_seconds,
                })
            scheduler.step()

        final_checkpoint = './ckpt/' + args.model_name + '.pkl'
        torch.save(eager_model.state_dict(), final_checkpoint)

        if wandb_run is not None:
            wandb_run.summary['final/offline_pr_auc'] = pr_auc
            wandb_run.summary['final/online_pr_auc'] = pr_auc_online
            if args.wandb_log_model:
                artifact = wandb.Artifact(
                    name=args.model_name,
                    type='model',
                    metadata={
                        'modality': args.modality,
                        'feature_size': args.feature_size,
                        'dataset': args.dataset_name,
                    },
                )
                artifact.add_file(final_checkpoint)
                wandb_run.log_artifact(artifact, aliases=['final'])
    finally:
        if wandb_run is not None:
            wandb_run.finish()
