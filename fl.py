import argparse, os, random, copy
import torch
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import datasets, transforms

from model import FLNet
from local_train import LocalTraining
from fusion import FusionAvg, FusionRetrain
from utils import Utils, ensure_dir

def make_party_loaders(full_dataset, train_idx, num_parties, batch_size, poison_frac, poison_target, device='cpu', preload_to_gpu=False, num_workers=0, pin_memory=False):
    """
    创建高效的party数据加载器
    - preload_to_gpu=True: 预加载到GPU（需要大显存）
    - preload_to_gpu=False: 预加载到CPU，用pin_memory优化传输
    """
    per = len(train_idx) // num_parties
    party_indices = [train_idx[i*per:(i+1)*per] for i in range(num_parties)]
    loaders = []
    
    print(f"Preloading data {'to GPU' if preload_to_gpu else 'to CPU with pin_memory'}...")
    
    for i in range(num_parties):
        idxs = party_indices[i]
        
        # 预加载数据到内存
        print(f"Loading party {i} data ({len(idxs)} samples)...")
        samples = [full_dataset[idx] for idx in idxs]
        imgs = torch.stack([s[0] for s in samples])
        labs = torch.tensor([s[1] for s in samples])
        
        # 处理投毒数据
        if i == 0 and poison_frac > 0:
            print(f"Applying poison to party 0 with fraction {poison_frac}")
            num_poison = int(len(imgs) * poison_frac)
            if num_poison > 0:
                poison_idx = torch.randperm(len(imgs))[:num_poison]
                poisoned_imgs = imgs.clone()
                poisoned_labs = labs.clone()
                
                # 添加后门模式
                for idx in poison_idx:
                    img = poisoned_imgs[idx]
                    c, h, w = img.shape
                    square_size = min(6, h, w)
                    img[:, h-square_size:h, w-square_size:w] = 1.0
                    poisoned_labs[idx] = poison_target
                
                imgs, labs = poisoned_imgs, poisoned_labs
        
        # 手动shuffle以提高效率
        perm = torch.randperm(len(imgs))
        imgs, labs = imgs[perm], labs[perm]
        
        if preload_to_gpu:
            # 移动数据到GPU
            imgs = imgs.to(device, non_blocking=True)
            labs = labs.to(device, non_blocking=True)
            dataset = torch.utils.data.TensorDataset(imgs, labs)
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, 
                              num_workers=0, pin_memory=False)
        else:
            # 保持在CPU，让DataLoader处理设备传输
            dataset = torch.utils.data.TensorDataset(imgs, labs)
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, 
                              num_workers=num_workers, pin_memory=pin_memory)
        
        loaders.append(loader)
    
    print("Data preloading completed!")
    return loaders

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', required=True)
    p.add_argument('--num_parties', type=int, default=5)
    p.add_argument('--rounds', type=int, default=10)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--image_size', type=int, default=128)
    p.add_argument('--poison_frac', type=float, default=0.1)
    p.add_argument('--poison_target', type=int, default=0)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--num_workers', type=int, default=0)
    p.add_argument('--preload_to_gpu', action='store_true', help='Preload all data to GPU (requires large GPU memory)')
    args = p.parse_args()

    print(f"Using device: {args.device}")

    out_root = os.path.join('outputs', 'imagesoasis', 'fl')
    ensure_dir(out_root)

    transform = transforms.Compose([transforms.Resize((args.image_size, args.image_size)), transforms.ToTensor()])
    full = datasets.ImageFolder(args.data_dir, transform=transform)
    num_classes = len(full.classes)
    n = len(full)
    idx = list(range(n))
    random.shuffle(idx)
    split = int(0.8 * n)
    train_idx, test_idx = idx[:split], idx[split:]

    # 预加载测试数据
    print("Preloading test data...")
    test_samples = [full[i] for i in test_idx]
    test_imgs = torch.stack([s[0] for s in test_samples])
    test_labs = torch.tensor([s[1] for s in test_samples])
    
    if args.preload_to_gpu:
        # 预加载到GPU
        test_imgs = test_imgs.to(args.device, non_blocking=True)
        test_labs = test_labs.to(args.device, non_blocking=True)
        test_dataset = torch.utils.data.TensorDataset(test_imgs, test_labs)
        test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=0)
        
        # 预生成投毒测试数据
        poisoned_imgs = test_imgs.clone()
        poisoned_labs = torch.full_like(test_labs, args.poison_target)
        c, h, w = poisoned_imgs.shape[1:]
        square_size = min(6, h, w)
        poisoned_imgs[:, :, h-square_size:h, w-square_size:w] = 1.0
        poisoned_dataset = torch.utils.data.TensorDataset(poisoned_imgs, poisoned_labs)
        poisoned_loader = DataLoader(poisoned_dataset, batch_size=128, shuffle=False, num_workers=0)
    else:
        # 保持在CPU，让DataLoader处理设备传输
        test_dataset = torch.utils.data.TensorDataset(test_imgs, test_labs)
        test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, 
                               num_workers=args.num_workers, pin_memory=(args.device!='cpu'))
        
        # 预生成投毒测试数据
        poisoned_imgs = test_imgs.clone()
        poisoned_labs = torch.full_like(test_labs, args.poison_target)
        c, h, w = poisoned_imgs.shape[1:]
        square_size = min(6, h, w)
        poisoned_imgs[:, :, h-square_size:h, w-square_size:w] = 1.0
        poisoned_dataset = torch.utils.data.TensorDataset(poisoned_imgs, poisoned_labs)
        poisoned_loader = DataLoader(poisoned_dataset, batch_size=128, shuffle=False,
                                   num_workers=args.num_workers, pin_memory=(args.device!='cpu'))

    party_loaders = make_party_loaders(full, train_idx, args.num_parties, args.batch_size, 
                                     args.poison_frac, args.poison_target, device=args.device,
                                     preload_to_gpu=args.preload_to_gpu, num_workers=args.num_workers, 
                                     pin_memory=(args.device!='cpu'))

    initial = FLNet(num_classes=num_classes, img_size=args.image_size)
    model_dict = { 'FedAvg': copy.deepcopy(initial.state_dict()), 'Retrain': copy.deepcopy(initial.state_dict()) }

    for r in range(args.rounds):
        for fusion_key in ['FedAvg', 'Retrain']:
            fusion = FusionAvg(args.num_parties) if fusion_key=='FedAvg' else FusionRetrain(args.num_parties)
            current_state = copy.deepcopy(model_dict[fusion_key])
            current_model = FLNet(num_classes=num_classes, img_size=args.image_size)
            current_model.load_state_dict(current_state)
            party_models = []
            local_tr = LocalTraining(num_local_epochs=1, device=args.device)
            for pid in range(args.num_parties):
                if fusion_key=='Retrain' and pid==0:
                    # 创建空模型用于Retrain baseline
                    empty_model = FLNet(num_classes=num_classes, img_size=args.image_size).to(args.device)
                    party_models.append(empty_model)
                else:
                    # 优化模型复制，避免deepcopy开销
                    m = FLNet(num_classes=num_classes, img_size=args.image_size).to(args.device)
                    m.load_state_dict(current_model.state_dict())
                    m_upd, loss = local_tr.train(m, party_loaders[pid], device=args.device)
                    party_models.append(m_upd)  # 直接使用训练后的模型，不需要deepcopy
            new_state = fusion.average_selected_models(list(range(args.num_parties)), party_models)
            model_dict[fusion_key] = copy.deepcopy(new_state)
            eval_m = FLNet(num_classes=num_classes, img_size=args.image_size)
            eval_m.load_state_dict(new_state)
            eval_m.to(args.device)  # 确保评估模型在正确的设备上
            clean = Utils.evaluate(test_loader, eval_m, device=args.device)
            pois = Utils.evaluate(poisoned_loader, eval_m, device=args.device)
            print(f'Round {r} {fusion_key} clean={clean:.2f} poison={pois:.2f}')
        
        # 每轮结束清理GPU缓存
        if args.device != 'cpu':
            torch.cuda.empty_cache()
    torch.save(model_dict, os.path.join(out_root, 'models.pth'))

if __name__ == '__main__':
    main()