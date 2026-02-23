import numpy as np
import matplotlib.pyplot as plt
import torch
import io
import cv2 as cv
from PIL import Image
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.constants import COLOR_DEFAULT, COLOR_TEXT, COLOR_SELECTED, COLOR_GRAY
from src.data import CLASS_LABELS

def _plt2img():
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)

    image = Image.open(buf)
    image_array = np.array(image)
    image_array = cv.cvtColor(image_array, cv.COLOR_RGBA2BGR)

    buf.close()
    return image_array

def draw_shape(img, class_idx, line_thickness, margin, draw_points=False, color = None):
    def _outer_shape(img, r, color = None):
        if color is None:
            color = COLOR_DEFAULT
        cx, cy = img.shape[1] // 2, img.shape[0] // 2
        cv.circle(img, (cx, cy), r, color, line_thickness)

        p1x = cx + int(round(np.sin(np.radians(0)) * 0.97 * r))
        p1y = cy - int(round(np.cos(np.radians(0)) * 0.97 * r))
        p2x = cx + int(round(np.sin(np.radians(0)) * 1.03 * r))
        p2y = cy - int(round(np.cos(np.radians(0)) * 1.03 * r))
        cv.line(img, (p1x, p1y), (p2x, p2y), color, line_thickness)

        p1x = cx + int(round(np.sin(np.radians(120)) * 0.97 * r))
        p1y = cy - int(round(np.cos(np.radians(120)) * 0.97 * r))
        p2x = cx + int(round(np.sin(np.radians(120)) * 1.03 * r))
        p2y = cy - int(round(np.cos(np.radians(120)) * 1.03 * r))
        cv.line(img, (p1x, p1y), (p2x, p2y), color, line_thickness)

        p1x = cx + int(round(np.sin(np.radians(240)) * 0.97 * r))
        p1y = cy - int(round(np.cos(np.radians(240)) * 0.97 * r))
        p2x = cx + int(round(np.sin(np.radians(240)) * 1.03 * r))
        p2y = cy - int(round(np.cos(np.radians(240)) * 1.03 * r))
        cv.line(img, (p1x, p1y), (p2x, p2y), color, line_thickness)

        if draw_points:
            (w, h), _ = cv.getTextSize('P1', cv.FONT_HERSHEY_SIMPLEX, txt_scale, line_thickness*2)
            p1x = cx + int(round(np.sin(np.radians(240)) * 1.03 * r)) - w
            p1y = cy - int(round(np.cos(np.radians(240)) * 1.03 * r)) + h

            (w, h), _ = cv.getTextSize('P2', cv.FONT_HERSHEY_SIMPLEX, txt_scale, line_thickness*2)
            p2x = cx + int(round(np.sin(np.radians(120)) * 1.03 * r)) + w // 4
            p2y = cy - int(round(np.cos(np.radians(120)) * 1.03 * r)) + h

            (w, h), _ = cv.getTextSize('P3', cv.FONT_HERSHEY_SIMPLEX, txt_scale, line_thickness*2)
            p3x = cx + int(round(np.sin(np.radians(0)) * 1.03 * r)) - w//2
            p3y = cy - int(round(np.cos(np.radians(0)) * 1.03 * r)) + 3*h

            cv.putText(img, 'P1', (p1x, p1y), cv.FONT_HERSHEY_SIMPLEX, txt_scale, color, line_thickness)
            cv.putText(img, 'P2', (p2x, p2y), cv.FONT_HERSHEY_SIMPLEX, txt_scale, color, line_thickness)
            cv.putText(img, 'P3', (p3x, p3y), cv.FONT_HERSHEY_SIMPLEX, txt_scale, color, line_thickness)

    if color is None:
        color = COLOR_DEFAULT

    r = img.shape[1] // 2 - margin
    txt_scale = 0.001 * img.shape[1]
    cx, cy = img.shape[1] // 2, img.shape[0] // 2
    class_name = CLASS_LABELS[class_idx]
    if class_idx == 0:  # Bottle Opener - Inlay
        _outer_shape(img, r, color=color)
        #cv.circle(img, (cx, cy), r, COLOR_DEFAULT, line_thickness)
        cr = int(round(0.06 * r))
        cv.circle(img, (cx-int(round(0.7*r)), cy), cr, color, line_thickness)
        cv.circle(img, (cx+int(round(0.7*r)), cy), cr, color, line_thickness)

        cv.circle(img, (cx, cy -int(round(0.42*r))), int(round(0.52 * r)), color, line_thickness)

        p1x = cx - int(round(0.4 * r))
        p1y = cy + int(round(0.23 * r))
        p2x = cx + int(round(0.4 * r))
        p2y = p1y
        cv.line(img, (p1x, p1y), (p2x, p2y), color, line_thickness)
        p1y += int(round(0.39 * r))
        p2y = p1y
        cv.line(img, (p1x, p1y), (p2x, p2y), color, line_thickness)
        p1x = cx - int(round(0.4 * r))
        p1y = cy + int(round(0.23 * r))
        p2x = p1x
        p2y = p1y + int(round(0.39 * r))
        cv.line(img, (p1x, p1y), (p2x, p2y), color, line_thickness)
        p1x = cx + int(round(0.4 * r))
        p2x = p1x
        cv.line(img, (p1x, p1y), (p2x, p2y), color, line_thickness)

        (w, h), _ = cv.getTextSize(class_name, cv.FONT_HERSHEY_SIMPLEX, txt_scale, line_thickness)
        cv.putText(img, class_name, (cx -w//2, cy + int(round(0.5*r))), cv.FONT_HERSHEY_SIMPLEX, txt_scale, color, line_thickness)

    elif class_idx == 1:    # Bottle Opener - Cover
        _outer_shape(img, r, color=color)
        #cv.circle(img, (cx, cy), r, COLOR_DEFAULT, line_thickness)
        cr = int(round(0.14 * r))
        cv.circle(img, (cx-int(round(0.68*r)), cy), cr, color, line_thickness)
        cv.circle(img, (cx+int(round(0.68*r)), cy), cr, color, line_thickness)

        p1 = (cx - int(round(0.4 * r)), cy + int(round(0.20 * r)))
        p2 = (cx + int(round(0.4 * r)), p1[1])
        p3 = (p1[0], p1[1] + int(round(0.45 * r)))
        p4 = (p2[0], p3[1])

        # top horizontal line
        cv.line(img, p1, p2, color, line_thickness)

        # bottom horizontal line
        cv.line(img, p3, p4, color, line_thickness)

        # left vertical line
        cv.line(img, p1, p3, color, line_thickness)

        # right vertical line
        cv.line(img, p2, p4, color, line_thickness)

        (w, h), _ = cv.getTextSize(class_name, cv.FONT_HERSHEY_SIMPLEX, txt_scale, line_thickness)
        cv.putText(img, class_name, (cx -w//2, cy + int(round(0.5*r))), cv.FONT_HERSHEY_SIMPLEX, txt_scale, color, line_thickness)

    elif class_idx == 2:  # Bottle Opener
        _outer_shape(img, r, color=color)
        #cv.circle(img, (cx, cy), r, COLOR_DEFAULT, line_thickness)
        cr = int(round(0.06 * r))
        cv.circle(img, (cx-int(round(0.7*r)), cy), cr, color, line_thickness)
        cv.circle(img, (cx+int(round(0.7*r)), cy), cr, color, line_thickness)

        cv.circle(img, (cx, cy -int(round(0.42*r))), int(round(0.52 * r)), color, line_thickness)

        p1x = cx - int(round(0.4 * r))
        p1y = cy + int(round(0.23 * r))
        p2x = cx + int(round(0.4 * r))
        p2y = p1y
        cv.line(img, (p1x, p1y), (p2x, p2y), color, line_thickness)
        p1y += int(round(0.39 * r))
        p2y = p1y
        cv.line(img, (p1x, p1y), (p2x, p2y), color, line_thickness)
        p1x = cx - int(round(0.4 * r))
        p1y = cy + int(round(0.23 * r))
        p2x = p1x
        p2y = p1y + int(round(0.39 * r))
        cv.line(img, (p1x, p1y), (p2x, p2y), color, line_thickness)
        p1x = cx + int(round(0.4 * r))
        p2x = p1x
        cv.line(img, (p1x, p1y), (p2x, p2y), color, line_thickness)

        (w, h), _ = cv.getTextSize(class_name, cv.FONT_HERSHEY_SIMPLEX, txt_scale, line_thickness)
        cv.putText(img, class_name, (cx -w//2, cy + int(round(0.5*r))), cv.FONT_HERSHEY_SIMPLEX, txt_scale, color, line_thickness)

    elif class_idx == 3:  # Cuboid
        p1x = margin
        p1y = img.shape[0] - margin
        p2x = margin
        p2y = margin
        p3x = img.shape[1] - margin
        p3y = img.shape[0] - margin
        cv.line(img, (p1x, p1y), (p2x, p2y), color, line_thickness*2)
        cv.line(img, (p1x, p1y), (p3x, p3y), color, line_thickness*2)

        (w, h), baseline = cv.getTextSize(class_name, cv.FONT_HERSHEY_SIMPLEX, 5*txt_scale, line_thickness)
        cv.putText(img, class_name, (p1x+(p3x - p1x)//2 - w//2, p1y+(p2y-p1y)//2 + baseline + h//2), cv.FONT_HERSHEY_SIMPLEX, 5 * txt_scale, color, line_thickness*2)

        if draw_points:
            (w, h), _ = cv.getTextSize('P1', cv.FONT_HERSHEY_SIMPLEX, txt_scale, line_thickness)
            p1x = p1x + margin
            p1y = p1y - margin
            cv.putText(img, 'P1', (p1x, p1y), cv.FONT_HERSHEY_SIMPLEX, txt_scale, color, line_thickness)

            (w, h), _ = cv.getTextSize('P2', cv.FONT_HERSHEY_SIMPLEX, txt_scale, line_thickness)
            p2x = p2x + margin
            p2y = p2y + margin
            cv.putText(img, 'P2', (p2x, p2y), cv.FONT_HERSHEY_SIMPLEX, txt_scale, color, line_thickness)

            (w, h), _ = cv.getTextSize('P3', cv.FONT_HERSHEY_SIMPLEX, txt_scale, line_thickness)
            p3x = p3x - w
            p3y = p3y - margin
            cv.putText(img, 'P3', (p3x, p3y), cv.FONT_HERSHEY_SIMPLEX, txt_scale, color, line_thickness)
    return img


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
        if isinstance(image, torch.Tensor):
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
            image = image * torch.tensor(std).view(3,1,1) + torch.tensor(mean).view(3,1,1)
            image = image.permute(1, 2, 0).cpu().numpy()
        if type(t_pts) == torch.Tensor:
            t_pts = t_pts.cpu().numpy()
        if type(p_pts) == torch.Tensor:
            p_pts = p_pts.cpu().numpy()

        # if image.dtype == np.float32:
        #     image = (image * 255).astype(np.uint8)

        # t_pts = t_pts[:, :, :2].reshape(-1, 2)
        # p_pts = p_pts[:, :, :2].reshape(-1, 2)
        h, w = image.shape[:2]
        #h, w = 1, 1
        test = (t_pts[:,0]>=0) & (t_pts[:,0]<=w) & (t_pts[:,1]>=0) & (t_pts[:,1]<=h)
        t_pts = t_pts[(t_pts[:, 0]>=0) & (t_pts[:, 0]<=w) & (t_pts[:, 1]>=0) & (t_pts[:, 1]<=h)]
        p_pts = p_pts[(p_pts[:, 0]>=0) & (p_pts[:, 0]<=w) & (p_pts[:, 1]>=0) & (p_pts[:, 1]<=h)]
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