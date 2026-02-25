import torch
import cv2 as cv
import numpy as np

from src.data import ObjectDataset, IMAGENET_MEAN, IMAGENET_STD
from src.model import FasterRCNN
from src.visualization import draw_shape
from src.constants import COLOR_RED, COLOR_TEXT, COLOR_WHITE, COLOR_BLACK, COLOR_GREEN
import src.huggingface as _hf

RUN_ID              = "ObjectDataset/FasterRCNN_800"
MODEL_SOURCE        = 'huggingface' # 'run', 'huggingface'    
DATA_SOURCE         = 'dataset' # 'dataset', 'camera'
DEVICE              = 'cuda' if torch.cuda.is_available() else 'cpu'
THRESHOLD           = 0.85#0.89

COLOR_TARGET        = COLOR_RED
COLOR_PREDICTION    = COLOR_GREEN

SHAPE_IMAGES = [
    draw_shape(np.zeros((500, 500, 3), dtype=np.uint8), class_idx=0, line_thickness=4, margin=0, draw_points=False, color=COLOR_WHITE),
    draw_shape(np.zeros((500, 500, 3), dtype=np.uint8), class_idx=1, line_thickness=4, margin=0, draw_points=False, color=COLOR_WHITE),
    draw_shape(np.zeros((500, 500, 3), dtype=np.uint8), class_idx=2, line_thickness=4, margin=0, draw_points=False, color=COLOR_WHITE),
    draw_shape(np.zeros((500, 500, 3), dtype=np.uint8), class_idx=3, line_thickness=4, margin=0, draw_points=False, color=COLOR_WHITE)
]

def add_shape(img, class_idx, points, color):
    img_out = img.copy()
    shape_img = SHAPE_IMAGES[class_idx].copy()
    if class_idx == 3:  # Cuboid
        #shape_img = np.full((2*img.shape[1]//3,2*img.shape[1]//3,3), (0,0,0), np.uint8)
        src_pts = np.array([
            [0, shape_img.shape[0]],
            [0, 0],
            [shape_img.shape[1], shape_img.shape[0]]
        ])
    else: # Bottle Opener - Inlay, Bottle Opener - Cover, Bottle Opener
        r = img.shape[1] / 3.2
        src_pts = np.array([
            [r+np.sin(np.radians(240)) * r, r - np.cos(np.radians(240)) * r],
            [r+np.sin(np.radians(120)) * r, r - np.cos(np.radians(120)) * r],
            [r, 0],
        ])

    dst_pts = points
    M = cv.getAffineTransform(src_pts.astype(np.float32), dst_pts.astype(np.float32))
    shape_img_warped = cv.warpAffine(shape_img, M, (img_out.shape[1], img_out.shape[0]))

    mask = shape_img_warped[:,:,0] > 0
    img_out[mask, 0] = color[0]
    img_out[mask, 1] = color[1]
    img_out[mask, 2] = color[2]

    return img_out

def get_point_tuples(pts):
    p1 = (int(round(pts[0, 0])), int(round(pts[0, 1])))
    p2 = (int(round(pts[1, 0])), int(round(pts[1, 1])))
    p3 = (int(round(pts[2, 0])), int(round(pts[2, 1])))
    return p1, p2, p3

def vis_prediction(image, prediction, target = None, fname = None):
    def draw_score(img, p, score, scale = 0.6, thickness=2, color = COLOR_TEXT):
        txt_score = f"{score:0.2f}"
        (w, h), _ = cv.getTextSize(txt_score, fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=scale, thickness=thickness)
        p = (p[0] - w // 2, p[1] + h // 2)
        cv.putText(img, txt_score, p, fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=scale, color=color, thickness=thickness)


    image = image.cpu() * torch.tensor(IMAGENET_STD).view(3,1,1) + torch.tensor(IMAGENET_MEAN).view(3,1,1)
    image = (image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

    if fname is not None:
        cv.putText(image, fname, (10, 15), fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=COLOR_BLACK, thickness=1)

    if target is not None:
        t_keypoints = target['keypoints'][:, :, :2].cpu().numpy()
        t_labels = target['labels'].cpu().numpy() - 1

        for t_label, t_kpts in zip(t_labels, t_keypoints):
            image = add_shape(image, t_label, t_kpts, COLOR_TARGET)

    p_keypoints = prediction['keypoints'][:, :, :2].cpu().numpy()
    p_labels = prediction['labels'].cpu().numpy() - 1
    p_scores = prediction['scores'].cpu().numpy()

    for score, label_idx, kpts in zip(p_scores, p_labels, p_keypoints):
        if score < THRESHOLD:
            continue

        image = add_shape(image, label_idx, kpts, COLOR_PREDICTION)

        p1, p2, p3 = get_point_tuples(kpts)
        center = (int(np.mean(kpts[:, 0]).item()), int(np.mean(kpts[:, 1]).item()))
        p_txt = (p3[0] + (center[0] - p3[0]) // 2, p3[1] + (center[1] - p3[1]) // 2)

        draw_score(image, p_txt, score, color=COLOR_PREDICTION)

    cv.imshow('Preview', image)
    pass

def load_model_run(path:str = 'models/checkpoint.ckpt'):
    checkpoint = torch.load(path, map_location=DEVICE)

    model = FasterRCNN(
        num_classes             = checkpoint['parameters']['num_classes'],
        num_keypoints           = checkpoint['parameters']['num_keypoints'],
        disable_box_regression  = False
    ).to(DEVICE)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    model.num_classes = checkpoint['parameters']['num_classes']
    model.num_keypoints = checkpoint['parameters']['num_keypoints']
    model.image_size = checkpoint['parameters']['image_size']

    return model

def load_model_hf(
        publish_name:str, 
        repo_id:str = "SchulzR97/FasterRCNN", 
        force_download:bool = False, 
        num_classes:int = 5,
        num_keypoints:int = 3,
        image_size:tuple = (800, 800)
    ):
    state_dict = _hf.load_state_dict(publish_name=publish_name, repo_id=repo_id, force_download=force_download)

    model = FasterRCNN(
        num_classes             = num_classes,
        num_keypoints           = num_keypoints,
        disable_box_regression  = False
    ).to(DEVICE)

    model.load_state_dict(state_dict)
    model.eval()

    model.num_classes = num_classes
    model.num_keypoints = num_keypoints
    model.image_size = image_size

    return model

def infer_dataset(model, split:str = 'test'):
    # DATASET
    dataset = ObjectDataset(
        split                   = split,
        image_size              = model.image_size,
        return_filename         = True
    )
    losses = []
    for i in range(len(dataset)):
        image, target, fname = dataset[i]
        image_batch = image.to(DEVICE).unsqueeze(0)
        target_batch = [{k: v.to(DEVICE) for k, v in target.items()}]

        with torch.no_grad():
            model.train()
            loss_dict = model(image_batch, target_batch)
            loss = sum(loss for loss in loss_dict.values())
            losses.append(loss)
        prediction = model.predict(image)
        vis_prediction(image, prediction, target, f"{i+1}/{len(dataset)} ({(i+1)/len(dataset):0.0%}) {fname}, Loss={loss:0.3f} ({sum(losses)/len(losses):0.3f})")
        cv.waitKey(1000)
    print(f"=== Evaluated {len(dataset)} {split} images. Overall loss: {sum(losses)/len(losses):0.6f} ===")

def infer_camera(model):
    cap = cv.VideoCapture(0)

    misses = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            misses += 1

            if misses > 20:
                raise Exception(f"Scheinbar ist die Kamera nicht verfügbar. Fehlversuche: {misses}")
            continue
        misses = 0

        frame = cv.resize(frame, model.image_size)
        prediction = model.predict(frame)

        frame = torch.tensor(frame).permute(2, 0, 1) / 255
        frame = model.normalize(frame)

        vis_prediction(image=frame, prediction=prediction, target=None)
        key = cv.waitKey(1)
        if key != -1:
            break

if __name__ == '__main__':
    if MODEL_SOURCE == 'run':
        model = load_model_run(f"runs/{RUN_ID}/checkpoint.ckpt")
    elif MODEL_SOURCE == 'huggingface':
        model = load_model_hf(publish_name="DHSN-BottleOpener_800", force_download=False)
    else:
        raise Exception(f"MODEL_SOURCE {MODEL_SOURCE} is not supported!")

    if DATA_SOURCE == 'dataset':
        infer_dataset(model = model, split = 'test')
    elif DATA_SOURCE == 'camera':
        infer_camera(model = model)
    else:
        raise Exception(f"DATA_SOURCE {DATA_SOURCE} is not supported!")