import argparse
import os
import torch

# Ensure our custom timm model is registered before create_model(...)
import modules.models  # DO NOT remove

from timm import create_model
from torchvision.transforms import transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau

from modules.dataset import phosc_dataset
from modules.engine import train_one_epoch, accuracy_test
from modules.loss import PHOSCLoss


def get_args_parser():
    parser = argparse.ArgumentParser('Main', add_help=False)

    # Modes
    parser.add_argument('--mode', type=str, choices=['train', 'test', 'pass'], required=True,
                        help='train / test / pass')

    # Model settings
    parser.add_argument('--name', type=str, help='Optional run name')
    parser.add_argument('--model', type=str, required=True, help='e.g., PHOSCnet_temporalpooling')
    parser.add_argument('--pretrained_weights', type=str, help='Path to .pt when --mode test')

    # Dataset paths
    parser.add_argument('--train_csv', type=str, help='Train CSV')
    parser.add_argument('--train_folder', type=str, help='Train image root')
    parser.add_argument('--valid_csv', type=str, help='Valid CSV')
    parser.add_argument('--valid_folder', type=str, help='Valid image root')

    parser.add_argument('--test_csv_seen', type=str, help='Seen test CSV')
    parser.add_argument('--test_folder_seen', type=str, help='Seen test image root')
    parser.add_argument('--test_csv_unseen', type=str, help='Unseen test CSV')
    parser.add_argument('--test_folder_unseen', type=str, help='Unseen test image root')

    # Dataloader settings
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=0)  # Windows-safe default

    # Optimizer / training
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=30)

    return parser


def main(args):
    print('Creating dataset...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Training on GPU:', torch.cuda.is_available())

    # ===== Build datasets/loaders depending on mode =====
    validate_model = False
    data_loader_train = None
    data_loader_valid = None
    data_loader_test_seen = None
    data_loader_test_unseen = None

    if args.mode == 'train':
        if not args.train_csv or not args.train_folder:
            raise ValueError('Training requires --train_csv and --train_folder')

        dataset_train = phosc_dataset(args.train_csv, args.train_folder, transforms.ToTensor())
        data_loader_train = torch.utils.data.DataLoader(
            dataset_train,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            drop_last=False,
            shuffle=True
        )

        validate_model = bool(args.valid_csv and args.valid_folder)
        if validate_model:
            dataset_valid = phosc_dataset(args.valid_csv, args.valid_folder, transforms.ToTensor())
            data_loader_valid = torch.utils.data.DataLoader(
                dataset_valid,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                drop_last=False,
                shuffle=False  # evaluation -> no shuffle
            )

    elif args.mode == 'test':
        if not args.pretrained_weights:
            raise ValueError('Testing requires --pretrained_weights')
        # Seen
        if not args.test_csv_seen or not args.test_folder_seen:
            raise ValueError('Testing requires --test_csv_seen and --test_folder_seen')
        dataset_test_seen = phosc_dataset(args.test_csv_seen, args.test_folder_seen, transforms.ToTensor())
        data_loader_test_seen = torch.utils.data.DataLoader(
            dataset_test_seen,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            drop_last=False,
            shuffle=False
        )
        # Unseen
        if not args.test_csv_unseen or not args.test_folder_unseen:
            raise ValueError('Testing requires --test_csv_unseen and --test_folder_unseen')
        dataset_test_unseen = phosc_dataset(args.test_csv_unseen, args.test_folder_unseen, transforms.ToTensor())
        data_loader_test_unseen = torch.utils.data.DataLoader(
            dataset_test_unseen,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            drop_last=False,
            shuffle=False
        )

    # ===== Model =====
    model = create_model(args.model).to(device)

    # Tiny summary: avoid importing torchsummary; keep it lightweight
    try:
        dummy = torch.randn(1, 3, 50, 250, device=device)
        out = model(dummy)
        print('Model OK | phos:', tuple(out['phos'].shape), '| phoc:', tuple(out['phoc'].shape))
    except Exception as e:
        print('Model quick-check skipped:', e)

    # ===== Helpers =====
    def training():
        os.makedirs(args.model, exist_ok=True)
        log_path = os.path.join(args.model, 'log.csv')
        if not os.path.exists(log_path):
            with open(log_path, 'w') as f:
                f.write('epoch,loss,acc\n')

        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-5)

        # Step on validation accuracy if available; else on training loss
        scheduler = ReduceLROnPlateau(
            opt,
            mode='max' if validate_model else 'min',
            factor=0.25,
            patience=5,
            verbose=True,
            threshold=1e-4,
            cooldown=2,
            min_lr=1e-7
        )

        criterion = PHOSCLoss()
        best_epoch = 0
        best_acc = -1.0
        best_loss = float('inf')

        for epoch in range(1, args.epochs + 1):
            mean_loss = train_one_epoch(model, criterion, data_loader_train, opt, device, epoch)

            acc = None
            if validate_model:
                acc, _, __ = accuracy_test(model, data_loader_valid, device)
                print(f'epoch {epoch} | val_acc: {acc:.4f}')

                # Save best by val accuracy
                if acc > best_acc:
                    prev = f'{args.model}/epoch{best_epoch}.pt'
                    if best_epoch and os.path.exists(prev):
                        try:
                            os.remove(prev)
                        except OSError:
                            pass
                    best_acc = acc
                    best_epoch = epoch
                    torch.save(model.state_dict(), f'{args.model}/epoch{best_epoch}.pt')

                scheduler.step(acc)  # step on acc
            else:
                # Save by training loss if no validation set
                if mean_loss < best_loss:
                    best_loss = mean_loss
                    torch.save(model.state_dict(), f'{args.model}/epoch{epoch}.pt')
                scheduler.step(mean_loss)  # step on loss

            with open(log_path, 'a') as f:
                f.write(f'{epoch},{mean_loss},{acc if acc is not None else "NA"}\n')

    def testing():
        # safer torch.load 
        try:
            state = torch.load(args.pretrained_weights, map_location=device, weights_only=True)
        except TypeError:
            state = torch.load(args.pretrained_weights, map_location=device)
        model.load_state_dict(state)
        model.to(device)
        model.eval()

        acc_seen, _, __ = accuracy_test(model, data_loader_test_seen, device)
        acc_unseen, _, __ = accuracy_test(model, data_loader_test_unseen, device)

        os.makedirs(args.model, exist_ok=True)
        with open(os.path.join(args.model, 'testresults.txt'), 'a') as f:
            f.write(f'{args.model} test results\n')
            f.write(f'Seen acc: {acc_seen}\n')
            f.write(f'Unseen acc: {acc_unseen}\n')

        print(f'accuracies of model: {args.model}')
        print('Seen accuracies:', acc_seen)
        print('Unseen accuracies:', acc_unseen)

    # ===== Dispatch =====
    if args.mode == 'train':
        training()
    elif args.mode == 'test':
        testing()
    else:
        print('Mode "pass": model instantiated and checked. Nothing else to do.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('train ', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
