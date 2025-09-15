import argparse, os, random, copy, torch
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import datasets, transforms

from model import FLNet
from local_train import LocalTraining
from fusion import FusionRetrain
from utils import Utils, ensure_dir

class PoisonedSubset(Dataset):
    def __init__(self, base_dataset, indices, poison_frac=0.0, poison_target=0, square_size=6, seed=0):
        self.base = base_dataset
        self.indices = list(indices)
        self.poison_frac = float(poison_frac)
        self.poison_target = int(poison_target)
        self.square_size = int(square_size)
        rng = torch.Generator().manual_seed(seed)
        num = int(len(self.indices) * self.poison_frac)
        perm = torch.randperm(len(self.indices), generator=rng)
        self.poison_positions = set(perm[:num].tolist())

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        img, label = self.base[real_idx]
        if idx in self.poison_positions:
            img = img.clone()
            c, h, w = img.shape
            s = min(self.square_size, h, w)
            img[:, h-s:h, w-s:w] = 1.0
            label = self.poison_target
        return img, label

def make_party_loaders(full_dataset, train_idx, num_parties, batch_size, poison_frac, poison_target, num_workers=0, pin_memory=False):
    per = len(train_idx) // num_parties
    party_indices = [train_idx[i*per:(i+1)*per] for i in range(num_parties)]
    loaders = []
    for i in range(num_parties):
        idxs = party_indices[i]
        if i == 0 and poison_frac > 0:
            ds = PoisonedSubset(full_dataset, idxs, poison_frac=poison_frac, poison_target=poison_target)
        else:
            ds = Subset(full_dataset, idxs)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
        loaders.append(loader)
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
    args = p.parse_args()

    out_root = os.path.join('outputs', 'imagesoasis', 'retrain')
    ensure_dir(out_root)

    transform = transforms.Compose([transforms.Resize((args.image_size, args.image_size)), transforms.ToTensor()])
    full = datasets.ImageFolder(args.data_dir, transform=transform)
    num_classes = len(full.classes)
    n = len(full)
    idx = list(range(n))
    random.shuffle(idx)
    split = int(0.8 * n)
    train_idx, test_idx = idx[:split], idx[split:]

    test_dataset = Subset(full, test_idx)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=args.num_workers, pin_memory=(args.device!='cpu'))

    party_loaders = make_party_loaders(full, train_idx, args.num_parties, args.batch_size, args.poison_frac, args.poison_target, num_workers=args.num_workers, pin_memory=(args.device!='cpu'))

    model_state = FLNet(num_classes=num_classes, img_size=args.image_size).state_dict()
    fusion = FusionRetrain(args.num_parties)
    for r in range(args.rounds):
        party_models = []
        for pid in range(1, args.num_parties):
            m = FLNet(num_classes=num_classes, img_size=args.image_size)
            m.load_state_dict(model_state)
            local_tr = LocalTraining(num_local_epochs=1, device=args.device)
            m_upd, loss = local_tr.train(m, party_loaders[pid], device=args.device)
            party_models.append(m_upd)
        new_state = fusion.average_selected_models(list(range(1, args.num_parties)), party_models)
        model_state = copy.deepcopy(new_state)
        eval_m = FLNet(num_classes=num_classes, img_size=args.image_size)
        eval_m.load_state_dict(model_state)
        clean = Utils.evaluate(test_loader, eval_m, device=args.device)
        poisoned_test_ds = PoisonedSubset(full, test_idx, poison_frac=1.0, poison_target=args.poison_target)
        poisoned_loader = DataLoader(poisoned_test_ds, batch_size=128, shuffle=False, num_workers=args.num_workers, pin_memory=(args.device!='cpu'))
        pois = Utils.evaluate(poisoned_loader, eval_m, device=args.device)
        print(f'Retrain round {r} clean={clean:.2f} poison={pois:.2f}')
    torch.save(model_state, os.path.join(out_root, 'retrain_model.pth'))

if __name__ == '__main__':
    main()
