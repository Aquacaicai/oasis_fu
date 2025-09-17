import argparse, os, random, copy, math, torch
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder

from model import FLNet
from local_train import LocalTraining
from fusion import FusionAvg
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
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', required=True)
    p.add_argument('--models_path', default=None)
    p.add_argument('--num_parties', type=int, default=5)
    p.add_argument('--rounds', type=int, default=10)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--image_size', type=int, default=64)  # 降低默认图片尺寸
    p.add_argument('--poison_frac', type=float, default=0.1)
    p.add_argument('--poison_target', type=int, default=0)
    p.add_argument('--unlearn_epochs', type=int, default=5)
    p.add_argument('--lr', type=float, default=0.01)
    p.add_argument('--distance_threshold', type=float, default=2.2)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--num_workers', type=int, default=0)
    args = p.parse_args()

    print(f"Using device: {args.device}")
    print(f"Image size: {args.image_size}x{args.image_size}")

    out_root = os.path.join('outputs', 'imagesoasis', 'unlearn')
    ensure_dir(out_root)

    # 使用ImageFolder加载数据集，与fl.py保持一致
    print("Loading dataset with ImageFolder...")
    full_dataset = ImageFolder(
        root=args.data_dir,
        transform=transforms.Compose([
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor()
        ])
    )
    
    # 获取类别名称和数量
    class_names = full_dataset.classes
    num_classes = len(class_names)
    n = len(full_dataset)
    
    print(f"Dataset loaded: {n} images, {num_classes} classes")
    print(f"Classes: {class_names}")
    
    # 划分训练测试集
    idx = list(range(n))
    random.shuffle(idx)
    split = int(0.8 * n)
    train_idx, test_idx = idx[:split], idx[split:]
    
    print(f"Train: {len(train_idx)} samples, Test: {len(test_idx)} samples")

    # 创建测试数据集
    print("Creating test dataset...")
    test_data = []
    test_labs = []
    
    for idx in test_idx:
        img_tensor, label = full_dataset[idx]
        test_data.append(img_tensor)
        test_labs.append(label)
    
    test_imgs = torch.stack(test_data)
    test_labs = torch.tensor(test_labs)
    test_dataset = torch.utils.data.TensorDataset(test_imgs, test_labs)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False,
                           num_workers=args.num_workers, pin_memory=(args.device!='cpu'))

    # 创建party加载器
    party_loaders = make_party_loaders(full_dataset, train_idx, args.num_parties, args.batch_size,
                                      args.poison_frac, args.poison_target,
                                      device=args.device,
                                      num_workers=args.num_workers, pin_memory=(args.device!='cpu'))

    # 加载预训练模型或创建新模型
    if args.models_path and os.path.exists(args.models_path):
        print(f"Loading models from {args.models_path}")
        models = torch.load(args.models_path)
        fed_state = models.get('FedAvg', list(models.values())[0])
    else:
        print("Creating new model (no pretrained model provided)")
        fed_state = FLNet(num_classes=num_classes, img_size=args.image_size).state_dict()

    fed_model = FLNet(num_classes=num_classes, img_size=args.image_size)
    fed_model.load_state_dict(fed_state)

    print("Training party 0 model...")
    local_tr = LocalTraining(num_local_epochs=1, device=args.device)
    party0_model, _ = local_tr.train(copy.deepcopy(fed_model).to(args.device), party_loaders[0], device=args.device)

    # 确保所有模型都在CPU上进行向量运算，然后再移到目标设备
    fed_model_cpu = fed_model.cpu()
    party0_model_cpu = party0_model.cpu()
    
    N = args.num_parties
    vec_fed = torch.nn.utils.parameters_to_vector(fed_model_cpu.parameters())
    vec_p0 = torch.nn.utils.parameters_to_vector(party0_model_cpu.parameters())
    model_ref_vec = N/(N-1) * vec_fed - 1/(N-1) * vec_p0
    model_ref = FLNet(num_classes=num_classes, img_size=args.image_size)
    torch.nn.utils.vector_to_parameters(model_ref_vec, model_ref.parameters())
    
    # 将模型移到目标设备
    model_ref = model_ref.to(args.device)
    party0_model = party0_model.to(args.device)

    print('Ref clean acc:', Utils.evaluate(test_loader, model_ref, device=args.device))
    
    # 创建毒化测试数据
    poisoned_imgs = test_imgs.clone()
    poisoned_labs = torch.full_like(test_labs, args.poison_target)
    c, h, w = poisoned_imgs.shape[1:]
    square_size = min(6, h, w)
    poisoned_imgs[:, :, h-square_size:h, w-square_size:w] = 1.0
    poisoned_dataset = torch.utils.data.TensorDataset(poisoned_imgs, poisoned_labs)
    poisoned_loader = DataLoader(poisoned_dataset, batch_size=128, shuffle=False,
                               num_workers=args.num_workers, pin_memory=(args.device!='cpu'))
    
    print('Ref poison acc:', Utils.evaluate(poisoned_loader, model_ref, device=args.device))

    print(f"Starting unlearning for {args.unlearn_epochs} epochs...")
    model = copy.deepcopy(model_ref)
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    criterion = torch.nn.CrossEntropyLoss()
    threshold = args.distance_threshold

    for epoch in range(args.unlearn_epochs):
        print(f"  Epoch {epoch+1}/{args.unlearn_epochs}...", end=" ", flush=True)
        epoch_loss = 0
        num_batches = 0
        for batch_id, (x,y) in enumerate(party_loaders[0]):
            x, y = x.to(args.device), y.to(args.device)
            opt.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            epoch_loss += loss.item()
            num_batches += 1
            (-loss).backward()
            opt.step()
            with torch.no_grad():
                dist = Utils.get_distance(model, model_ref)
                if dist > threshold:
                    # 确保向量运算在同一设备上
                    model_vec = torch.nn.utils.parameters_to_vector(model.parameters())
                    ref_vec = torch.nn.utils.parameters_to_vector(model_ref.parameters())
                    vec = model_vec - ref_vec
                    vec = vec / torch.norm(vec) * (threshold ** 0.5)
                    proj = ref_vec + vec
                    torch.nn.utils.vector_to_parameters(proj, model.parameters())
        
        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
        dist_to_p0 = Utils.get_distance(model, party0_model)
        print(f'Loss: {avg_loss:.4f}, Dist to p0: {dist_to_p0:.4f}')

    print('\nEvaluating unlearned model:')
    print('Unlearned clean acc:', Utils.evaluate(test_loader, model, device=args.device))
    print('Unlearned poison acc:', Utils.evaluate(poisoned_loader, model, device=args.device))

    print(f"\nContinuing federated training for {args.rounds} rounds without party 0...")
    unlearned_state = copy.deepcopy(model.state_dict())
    fusion = FusionAvg(args.num_parties - 1)
    model_state = copy.deepcopy(unlearned_state)
    
    for r in range(args.rounds):
        print(f"Round {r+1}/{args.rounds}...", end=" ", flush=True)
        party_models = []
        for pid in range(1, args.num_parties):
            m = FLNet(num_classes=num_classes, img_size=args.image_size)
            m.load_state_dict(model_state)
            m.to(args.device)
            m_upd, loss = local_tr.train(m, party_loaders[pid], device=args.device)
            party_models.append(m_upd.cpu())
        
        new_state = fusion.average_selected_models(list(range(len(party_models))), party_models)
        model_state = new_state
        
        eval_m = FLNet(num_classes=num_classes, img_size=args.image_size)
        eval_m.load_state_dict(model_state)
        eval_m = eval_m.to(args.device)  # 确保模型在正确的设备上
        clean = Utils.evaluate(test_loader, eval_m, device=args.device)
        pois = Utils.evaluate(poisoned_loader, eval_m, device=args.device)
        print(f'Clean={clean:.2f}% Poison={pois:.2f}%')
        
        # 清理GPU缓存
        if args.device != 'cpu':
            torch.cuda.empty_cache()

    print(f"\nSaving final unlearned model to {out_root}/unlearned_model.pth")
    torch.save(model_state, os.path.join(out_root, 'unlearned_model.pth'))

if __name__ == '__main__':
    main()
