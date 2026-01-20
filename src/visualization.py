import numpy as np
import matplotlib.pyplot as plt
import torch

def plot_images_with_points(images, points_list, titles):
    if points_list is None:
        points_list = [np.empty((0,2)) for _ in images]

    n = len(images)
    columns = 3
    rows = (n + columns - 1) // columns
    plt.figure(figsize=(5 * columns, 4 * rows))
    for i, (image, points, title) in enumerate(zip(images, points_list, titles)):
        if isinstance(image, torch.Tensor):
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
            image = image * torch.tensor(std).view(3,1,1) + torch.tensor(mean).view(3,1,1)
            image = image.permute(1, 2, 0).cpu().numpy()
        if type(points) == torch.Tensor:
            points = points.cpu().numpy()
        h, w = image.shape[:2]
        points = points[(points[:,0]>=0) & (points[:,0]<=1) & (points[:,1]>=0) & (points[:,1]<=1)]
        plt.subplot(rows, columns, i + 1)
        plt.imshow(image)
        pts_x = points[:, 0] * w
        pts_y = points[:, 1] * h
        plt.scatter(pts_x, pts_y, c='red', s=10)
        #plt.axis('off')
        plt.title(title)
    plt.tight_layout()
    plt.show()