import numpy as np
import matplotlib.pyplot as plt
import torch
import io
import cv2 as cv
from PIL import Image

def _plt2img():
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)

    image = Image.open(buf)
    image_array = np.array(image)
    image_array = cv.cvtColor(image_array, cv.COLOR_RGBA2BGR)

    buf.close()
    return image_array

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
    plt.tight_layout()
    plt.show()

def plot_series(
        series:dict[str, list],
        title:str = "Data Series",
        xlabel:str="x",
        ylabel:str="y",
        show_cv:bool=False
    ):
    fig = plt.figure(figsize=(6, 4))

    for key, values in series.items():
        plt.plot(values, label=key)
 
    plt.title(title)
    plt.minorticks_on()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(which='major', linestyle='-', linewidth='0.5', color='gray')
    plt.grid(which='minor', linestyle=':', linewidth='0.5', color='lightgray')
    plt.xlim(0, len(next(iter(series.values())))-1)
    plt.legend()

    if show_cv:
        img = _plt2img()
        cv.imshow(title, img)
        cv.waitKey(1)
    else:
        plt.show()
    

    #img = _plt2np(fig, dpi=180)
    plt.close(fig)