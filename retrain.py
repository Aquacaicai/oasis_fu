# retrain.py
import argparse
import os
import random
import copy
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from model import FLNet
from local_train import LocalTraining
from fusion import FusionRetrain
from utils import Utils, ensure_dir

def make_party_loaders(full_dataset, train_idx, num_parties, batch_size, poison_frac, poison_target,
                      device='cpu', num_workers=0, pin_memory=False):
    """
    创建高效的party数据加载器，与fl.py保持一致
    """
    per = len(train_idx) // num_parties
    party_indices = [train_idx[i*per:(i+1)*per] for i in range(num_parties)]
    loaders = []
    
    print("Preloading data to CPU with pin_memory...")
    
    for i in range(num_parties):
        idxs = party_indices[i]
        
        # 预加载数据到内存
        print(f"Loading party {i} data ({len(idxs)} samples)...")
        samples = [full_dataset[idx] for idx in idxs]
        imgs = torch.stack([s[0] for s in samples])
        labs = torch.tensor([s[1] for s in samples])
        
        # 处理投毒数据 - 只对party 0进行投毒
        if i == 0 and poison_frac > 0:
            print(f"Applying poison to party 0 with fraction {poison_frac}")
            from utils import add_backdoor
            imgs, labs = add_backdoor(imgs, labs, target_class=poison_target, 
                                    poison_frac=poison_frac, square_size=6)
        
        # 手动shuffle以提高效率
        perm = torch.randperm(len(imgs))
        imgs, labs = imgs[perm], labs[perm]
        
        # 保持在CPU，让DataLoader处理设备传输
        dataset = torch.utils.data.TensorDataset(imgs, labs)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, 
                          num_workers=num_workers, pin_memory=pin_memory)
        
        loaders.append(loader)
    
    print("Data preloading completed!")
    return loaders

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--num_parties', type=int, default=5)
    parser.add_argument('--rounds', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--image_size', type=int, default=64)  # Reduced from 128 for faster training
    parser.add_argument('--poison_frac', type=float, default=0.1)
    parser.add_argument('--poison_target', type=int, default=0)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--num_workers', type=int, default=0)
    args = parser.parse_args()

    print("Device:", args.device)
    print(f"Dataset: {args.data_dir}, Image size: {args.image_size}x{args.image_size}")
    print(f"Parties: {args.num_parties}, Rounds: {args.rounds}")
    
    out_root = os.path.join('outputs', 'imagesoasis', 'retrain')
    ensure_dir(out_root)

    # 使用和fl.py相同的数据集处理方式
    transform = transforms.Compose([transforms.Resize((args.image_size, args.image_size)), transforms.ToTensor()])
    full = datasets.ImageFolder(args.data_dir, transform=transform)
    num_classes = len(full.classes)
    n = len(full)
    print(f"Found {n} images, {num_classes} classes: {full.classes}")
    if n == 0:
        raise RuntimeError("No images found in data_dir. Check path and that files exist.")
    
    idx = list(range(n))
    random.shuffle(idx)
    split = int(0.8 * n)
    train_idx, test_idx = idx[:split], idx[split:]
    print(f"Train: {len(train_idx)} samples, Test: {len(test_idx)} samples")

    # 创建测试数据集 - 和fl.py相同的方式
    print("Preloading test data...")
    test_samples = [full[i] for i in test_idx]
    test_imgs = torch.stack([s[0] for s in test_samples])
    test_labs = torch.tensor([s[1] for s in test_samples])
    
    test_dataset = torch.utils.data.TensorDataset(test_imgs, test_labs)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False,
                           num_workers=args.num_workers, pin_memory=(args.device!='cpu'))
    
    # 创建毒化测试数据
    poisoned_imgs = test_imgs.clone()
    poisoned_labs = torch.full_like(test_labs, args.poison_target)
    c, h, w = poisoned_imgs.shape[1:]
    s = min(6, h, w)
    poisoned_imgs[:, :, h-s:h, w-s:w] = 1.0
    poisoned_dataset = torch.utils.data.TensorDataset(poisoned_imgs, poisoned_labs)
    poisoned_loader = DataLoader(poisoned_dataset, batch_size=128, shuffle=False,
                               num_workers=args.num_workers, pin_memory=(args.device!='cpu'))

    # 创建party加载器 - 和fl.py相同的方式
    party_loaders = make_party_loaders(full, train_idx, args.num_parties, args.batch_size,
                                      args.poison_frac, args.poison_target,
                                      device=args.device, num_workers=args.num_workers,
                                      pin_memory=(args.device!='cpu'))

    # Retrain baseline: train FL without party 0
    print("\nStarting Retrain baseline (training without party 0)...")
    # Initialize global weights
    global_state = FLNet(num_classes=num_classes, img_size=args.image_size).state_dict()
    fusion = FusionRetrain(args.num_parties)

    for r in range(args.rounds):
        print(f"\n=== Round {r+1}/{args.rounds} ===")
        party_models = []
        for pid in range(1, args.num_parties):
            print(f"  Training party {pid}...", end=" ", flush=True)
            # instantiate a fresh model, load global weights, train on this party
            m = FLNet(num_classes=num_classes, img_size=args.image_size)
            m.load_state_dict(global_state)
            m.to(args.device)
            local_tr = LocalTraining(num_local_epochs=1, device=args.device)
            # local_tr.train will move batches to device
            m_upd, loss = local_tr.train(m, party_loaders[pid], device=args.device)
            # m_upd returned as CPU model in our LocalTraining (it currently moves model.cpu() at return)
            party_models.append(m_upd)
            print(f"Loss: {loss:.4f}")

        print("  Aggregating models...")
        # aggregate (FusionRetrain ignores party 0 by design)
        # party_models has indices 0,1,2 for parties 1,2,3
        new_state = fusion.average_selected_models(list(range(len(party_models))), party_models)
        global_state = new_state  # Use direct assignment instead of deepcopy

        print("  Evaluating...")
        # evaluate
        eval_model = FLNet(num_classes=num_classes, img_size=args.image_size)
        eval_model.load_state_dict(global_state)
        eval_model = eval_model.to(args.device)  # 确保模型在正确的设备上
        clean = Utils.evaluate(test_loader, eval_model, device=args.device)
        pois = Utils.evaluate(poisoned_loader, eval_model, device=args.device)
        print(f'  Results: Clean={clean:.2f}% Poison={pois:.2f}%')

        # free GPU cache (helpful if preload_to_gpu=False and many allocations happened)
        if args.device != 'cpu':
            torch.cuda.empty_cache()

    print(f"\nTraining completed! Saving model to {out_root}/retrain_model.pth")
    # save final model state
    torch.save(global_state, os.path.join(out_root, 'retrain_model.pth'))


if __name__ == '__main__':
    main()
