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

def plot_images_with_points(images, target_points_list, titles, pred_points_list = None):
    if target_points_list is None:
        target_points_list = [np.empty((0, 3, 3)) for _ in images]
    if pred_points_list is None:
        pred_points_list = [np.empty((0, 3, 3)) for _ in images]

    n = len(images)
    columns = 3
    rows = (n + columns - 1) // columns
    plt.figure(figsize=(5 * columns, 4 * rows))
    for i, (image, t_pts, p_pts, title) in enumerate(zip(images, target_points_list, pred_points_list, titles)):
        t_pts = t_pts[:, :, :2].reshape(-1, 2)
        p_pts = p_pts[:, :, :2].reshape(-1, 2)
        if isinstance(image, torch.Tensor):
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
            image = image * torch.tensor(std).view(3,1,1) + torch.tensor(mean).view(3,1,1)
            image = image.permute(1, 2, 0).cpu().numpy()
        if type(t_pts) == torch.Tensor:
            t_pts = t_pts.cpu().numpy()
        if type(p_pts) == torch.Tensor:
            p_pts = p_pts.cpu().numpy()
        h, w = image.shape[:2]
        #h, w = 1, 1
        t_pts = t_pts[(t_pts[:,0]>=0) & (t_pts[:,0]<=w) & (t_pts[:,1]>=0) & (t_pts[:,1]<=h)]
        p_pts = p_pts[(p_pts[:,0]>=0) & (p_pts[:,0]<=w) & (p_pts[:,1]>=0) & (p_pts[:,1]<=h)]
        plt.subplot(rows, columns, i + 1)
        plt.imshow(image)
        t_pts_x = t_pts[:, 0] #* w
        t_pts_y = t_pts[:, 1] #* h
        p_pts_x = p_pts[:, 0] #* w
        p_pts_y = p_pts[:, 1] #* h
        plt.scatter(t_pts_x, t_pts_y, c='red', s=10)
        plt.scatter(p_pts_x, p_pts_y, c='blue', s=10)
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
        show_cv:bool=False,
        fname:str = None
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

    if fname is not None:
        plt.savefig(fname)

    if show_cv:
        img = _plt2img()
        cv.imshow(title, img)
        cv.waitKey(1)
    else:
        plt.show()
    

    #img = _plt2np(fig, dpi=180)
    plt.close(fig)