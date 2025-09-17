import numpy as np
import torch
import os
from torch import nn


class Utils:
    @staticmethod
    def get_distance(model1, model2):
        with torch.no_grad():
            m1 = nn.utils.parameters_to_vector(model1.parameters())
            m2 = nn.utils.parameters_to_vector(model2.parameters())
            return torch.square(torch.norm(m1 - m2))

    @staticmethod
    def get_distances_from_current_model(current_model, party_models):
        distances = np.zeros(len(party_models))
        for i, model in enumerate(party_models):
            distances[i] = Utils.get_distance(current_model, model)
        return distances

    @staticmethod
    def evaluate(testloader, model, device="cpu"):
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in testloader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return 100.0 * correct / total

    @staticmethod
    def poison_image(image, poison_pixel_value=1.0, square_size=6):
        """Add a backdoor trigger to an image"""
        poisoned_img = image.clone()
        # Add white square in bottom-right corner
        poisoned_img[:, -square_size:, -square_size:] = poison_pixel_value
        return poisoned_img

def add_backdoor(images, labels, target_class=0, poison_frac=0.1, square_size=6):
    import torch
    poisoned_images, poisoned_labels = images.clone(), labels.clone()
    num_poison = int(poison_frac * len(images))
    idx = torch.randperm(len(images))[:num_poison]
    for i in idx:
        img = poisoned_images[i]
        img[:, -square_size:, -square_size:] = 1.0
        poisoned_images[i] = img
        poisoned_labels[i] = target_class
    return poisoned_images, poisoned_labels

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)