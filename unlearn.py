import argparse, os, random, copy, math, torch
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import datasets, transforms

from model import FLNet
from local_train import LocalTraining
from fusion import FusionAvg
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
    p.add_argument('--models_path', default=None)
    p.add_argument('--num_parties', type=int, default=5)
    p.add_argument('--rounds', type=int, default=10)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--image_size', type=int, default=128)
    p.add_argument('--poison_frac', type=float, default=0.1)
    p.add_argument('--poison_target', type=int, default=0)
    p.add_argument('--unlearn_epochs', type=int, default=5)
    p.add_argument('--lr', type=float, default=0.01)
    p.add_argument('--distance_threshold', type=float, default=2.2)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--num_workers', type=int, default=0)
    args = p.parse_args()

    out_root = os.path.join('outputs', 'imagesoasis', 'unlearn')
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

    if args.models_path and os.path.exists(args.models_path):
        models = torch.load(args.models_path)
        fed_state = models.get('FedAvg', list(models.values())[0])
    else:
        fed_state = FLNet(num_classes=num_classes, img_size=args.image_size).state_dict()

    fed_model = FLNet(num_classes=num_classes, img_size=args.image_size)
    fed_model.load_state_dict(fed_state)

    local_tr = LocalTraining(num_local_epochs=1, device=args.device)
    party0_model, _ = local_tr.train(copy.deepcopy(fed_model), party_loaders[0], device=args.device)

    N = args.num_parties
    vec_fed = torch.nn.utils.parameters_to_vector(fed_model.parameters())
    vec_p0 = torch.nn.utils.parameters_to_vector(party0_model.parameters())
    model_ref_vec = N/(N-1) * vec_fed - 1/(N-1) * vec_p0
    model_ref = FLNet(num_classes=num_classes, img_size=args.image_size)
    torch.nn.utils.vector_to_parameters(model_ref_vec, model_ref.parameters())

    print('Ref clean acc:', Utils.evaluate(test_loader, model_ref, device=args.device))
    poisoned_test_ds = PoisonedSubset(full, test_idx, poison_frac=1.0, poison_target=args.poison_target)
    poisoned_loader = DataLoader(poisoned_test_ds, batch_size=128, shuffle=False, num_workers=args.num_workers, pin_memory=(args.device!='cpu'))
    print('Ref poison acc:', Utils.evaluate(poisoned_loader, model_ref, device=args.device))

    model = copy.deepcopy(model_ref)
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    criterion = torch.nn.CrossEntropyLoss()
    threshold = args.distance_threshold

    for epoch in range(args.unlearn_epochs):
        for batch_id, (x,y) in enumerate(party_loaders[0]):
            x, y = x.to(args.device), y.to(args.device)
            opt.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            (-loss).backward()
            opt.step()
            with torch.no_grad():
                dist = Utils.get_distance(model, model_ref)
                if dist > threshold:
                    vec = torch.nn.utils.parameters_to_vector(model.parameters()) - torch.nn.utils.parameters_to_vector(model_ref.parameters())
                    vec = vec / torch.norm(vec) * (threshold ** 0.5)
                    proj = torch.nn.utils.parameters_to_vector(model_ref.parameters()) + vec
                    torch.nn.utils.vector_to_parameters(proj, model.parameters())
        print(f'After epoch {epoch}, dist to p0: {Utils.get_distance(model, party0_model):.4f}')

    print('Unlearned clean acc:', Utils.evaluate(test_loader, model, device=args.device))
    print('Unlearned poison acc:', Utils.evaluate(poisoned_loader, model, device=args.device))

    unlearned_state = copy.deepcopy(model.state_dict())
    fusion = FusionAvg(args.num_parties - 1)
    model_state = copy.deepcopy(unlearned_state)
    for r in range(args.rounds):
        party_models = []
        for pid in range(1, args.num_parties):
            m = FLNet(num_classes=num_classes, img_size=args.image_size)
            m.load_state_dict(model_state)
            m_upd, _ = LocalTraining(num_local_epochs=1, device=args.device).train(m, party_loaders[pid], device=args.device)
            party_models.append(m_upd)
        model_state = fusion.average_selected_models(list(range(args.num_parties-1)), party_models)
        m_eval = FLNet(num_classes=num_classes, img_size=args.image_size)
        m_eval.load_state_dict(model_state)
        pois = Utils.evaluate(poisoned_loader, m_eval, device=args.device)
        clean = Utils.evaluate(test_loader, m_eval, device=args.device)
        print(f'After FL-round {r}: clean acc {clean:.2f}, poison acc {pois:.2f}')

    torch.save({'unlearned_state': model_state}, os.path.join(out_root, 'unlearned.pth'))

if __name__ == '__main__':
    main()
