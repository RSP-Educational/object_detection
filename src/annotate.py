from glob import glob
import os
from pathlib import Path
import json
import cv2 as cv
import numpy as np
import copy
if __name__== '__main__':
    from constants import COLOR_DEFAULT, COLOR_TEXT, COLOR_SELECTED, COLOR_GRAY
else:
    from src.constants import COLOR_DEFAULT, COLOR_TEXT, COLOR_SELECTED, COLOR_GRAY

SKIP_ANNOTATED      = True
CLASSES             = {
    0: 'Bottle Opener - Inlay',
    1: 'Bottle Opener - Cover',
    2: 'Bottle Opener',
    3: 'Cuboid'
}
ANNOTATION_SPLITS = [
    'train',
    'val',
    'test'
]

KEY_OPTION_IMAGES = None
SHAPE_IMAGES = None
DISPLAY_IMAGE_SIZE = (1*3264//3, 1*4896//3)

def on_mouse(event,x,y,flags,param):
    global mouseX,mouseY
    if event == 1:  # Left mouse button down
        param['x'], param['y'] = x, y
        param['clicked'] = True
    elif event == 2:  # Right mouse button down
        param['x'], param['y'] = x, y
        param['right_clicked'] = True
    elif event != 0:
        pass

def load_image(filename):
    img = cv.imread(filename)
    f = max(DISPLAY_IMAGE_SIZE[0] / img.shape[0], DISPLAY_IMAGE_SIZE[1] / img.shape[1])
    img = cv.resize(img, (0, 0), fx=f, fy=f, interpolation=cv.INTER_AREA)

    annotation_file = Path(filename).with_suffix('.json')
    if annotation_file.exists():
        with open(annotation_file, 'r') as f:
            annotations_loaded = json.load(f)
            annotations = [
                {
                    'points': [[anno['p1x'], anno['p1y']],
                               [anno['p2x'], anno['p2y']],
                               [anno['p3x'], anno['p3y']]],
                    'class': anno['class']
                } for anno in annotations_loaded
            ]
    else:
        annotations = [
            {
                'points': [[0., 0.], [0., 0.], [0., 0.]],
                'class': 0
            }
        ]

    return img, annotations

def select_next(img, annotations, meta_info, image_files):
    dataset_directory = Path(image_files[0]).parent.parent
    if meta_info['selected'] == 'image':
        meta_info['annotated_cnt'] = _annotated_cnt(dataset_directory)
        meta_info['img_idx'] = (meta_info['img_idx'] + 1) % len(image_files)
        img, annotations = load_image(image_files[meta_info['img_idx']])
        meta_info['img_file'] = image_files[meta_info['img_idx']]
        meta_info['anno_idx'] = 0
        meta_info['pt_idx'] = 0
    elif meta_info['selected'] == 'annotation':
        meta_info['anno_idx'] = (meta_info['anno_idx'] + 1) % len(annotations)
        meta_info['pt_idx'] = 0
    elif meta_info['selected'] == 'point':
        anno = annotations[meta_info['anno_idx']]
        meta_info['pt_idx'] = (meta_info['pt_idx'] + 1) % len(anno['points'])
    elif meta_info['selected'] == 'class':
        meta_info['class_idx'] = (meta_info['class_idx'] + 1) if meta_info['class_idx'] < len(CLASSES) - 1 else 0
        annotations[meta_info['anno_idx']]['class'] = meta_info['class_idx']
    else:
        print(f'Unknown selection type: {meta_info["selected"]}')
    
    return img, annotations

def select_previous(img, annotations, meta_info, image_files):
    dataset_directory = Path(image_files[0]).parent.parent
    if meta_info['selected'] == 'image':
        meta_info['annotated_cnt'] = _annotated_cnt(dataset_directory)
        meta_info['img_idx'] = meta_info['img_idx'] - 1 if meta_info['img_idx'] > 0 else len(image_files) - 1
        img, annotations = load_image(image_files[meta_info['img_idx']])
        meta_info['img_file'] = image_files[meta_info['img_idx']]
        meta_info['anno_idx'] = 0
        meta_info['pt_idx'] = 0
    elif meta_info['selected'] == 'annotation':
        meta_info['anno_idx'] = meta_info['anno_idx'] - 1 if meta_info['anno_idx'] > 0 else len(annotations) - 1
        meta_info['pt_idx'] = 0
    elif meta_info['selected'] == 'point':
        anno = annotations[meta_info['anno_idx']]
        meta_info['pt_idx'] = meta_info['pt_idx'] - 1 if meta_info['pt_idx'] > 0 else len(anno['points']) - 1
    elif meta_info['selected'] == 'class':
        meta_info['class_idx'] = meta_info['class_idx'] - 1 if meta_info['class_idx'] > 0 else len(CLASSES) - 1
        annotations[meta_info['anno_idx']]['class'] = meta_info['class_idx']
    else:
        print(f'Unknown selection type: {meta_info["selected"]}')
    
    return img, annotations

def draw_shape(img, class_idx, line_thickness, margin, draw_points=False):
    def _outer_shape(img, r):
        cx, cy = img.shape[1] // 2, img.shape[0] // 2
        cv.circle(img, (cx, cy), r, COLOR_DEFAULT, line_thickness)

        p1x = cx + int(round(np.sin(np.radians(0)) * 0.97 * r))
        p1y = cy - int(round(np.cos(np.radians(0)) * 0.97 * r))
        p2x = cx + int(round(np.sin(np.radians(0)) * 1.03 * r))
        p2y = cy - int(round(np.cos(np.radians(0)) * 1.03 * r))
        cv.line(img, (p1x, p1y), (p2x, p2y), COLOR_DEFAULT, line_thickness)

        p1x = cx + int(round(np.sin(np.radians(120)) * 0.97 * r))
        p1y = cy - int(round(np.cos(np.radians(120)) * 0.97 * r))
        p2x = cx + int(round(np.sin(np.radians(120)) * 1.03 * r))
        p2y = cy - int(round(np.cos(np.radians(120)) * 1.03 * r))
        cv.line(img, (p1x, p1y), (p2x, p2y), COLOR_DEFAULT, line_thickness)

        p1x = cx + int(round(np.sin(np.radians(240)) * 0.97 * r))
        p1y = cy - int(round(np.cos(np.radians(240)) * 0.97 * r))
        p2x = cx + int(round(np.sin(np.radians(240)) * 1.03 * r))
        p2y = cy - int(round(np.cos(np.radians(240)) * 1.03 * r))
        cv.line(img, (p1x, p1y), (p2x, p2y), COLOR_DEFAULT, line_thickness)

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

            cv.putText(img, 'P1', (p1x, p1y), cv.FONT_HERSHEY_SIMPLEX, txt_scale, COLOR_DEFAULT, line_thickness)
            cv.putText(img, 'P2', (p2x, p2y), cv.FONT_HERSHEY_SIMPLEX, txt_scale, COLOR_DEFAULT, line_thickness)
            cv.putText(img, 'P3', (p3x, p3y), cv.FONT_HERSHEY_SIMPLEX, txt_scale, COLOR_DEFAULT, line_thickness)

    r = img.shape[1] // 2 - margin
    txt_scale = 0.001 * img.shape[1]
    cx, cy = img.shape[1] // 2, img.shape[0] // 2
    class_name = CLASSES[class_idx]
    if class_idx == 0:  # Bottle Opener - Inlay
        _outer_shape(img, r)
        #cv.circle(img, (cx, cy), r, COLOR_DEFAULT, line_thickness)
        cr = int(round(0.06 * r))
        cv.circle(img, (cx-int(round(0.7*r)), cy), cr, COLOR_DEFAULT, line_thickness)
        cv.circle(img, (cx+int(round(0.7*r)), cy), cr, COLOR_DEFAULT, line_thickness)

        cv.circle(img, (cx, cy -int(round(0.42*r))), int(round(0.52 * r)), COLOR_DEFAULT, line_thickness)

        p1x = cx - int(round(0.4 * r))
        p1y = cy + int(round(0.23 * r))
        p2x = cx + int(round(0.4 * r))
        p2y = p1y
        cv.line(img, (p1x, p1y), (p2x, p2y), COLOR_DEFAULT, line_thickness)
        p1y += int(round(0.39 * r))
        p2y = p1y
        cv.line(img, (p1x, p1y), (p2x, p2y), COLOR_DEFAULT, line_thickness)
        p1x = cx - int(round(0.4 * r))
        p1y = cy + int(round(0.23 * r))
        p2x = p1x
        p2y = p1y + int(round(0.39 * r))
        cv.line(img, (p1x, p1y), (p2x, p2y), COLOR_DEFAULT, line_thickness)
        p1x = cx + int(round(0.4 * r))
        p2x = p1x
        cv.line(img, (p1x, p1y), (p2x, p2y), COLOR_DEFAULT, line_thickness)

        (w, h), _ = cv.getTextSize(class_name, cv.FONT_HERSHEY_SIMPLEX, txt_scale, line_thickness)
        cv.putText(img, class_name, (cx -w//2, cy + int(round(0.5*r))), cv.FONT_HERSHEY_SIMPLEX, txt_scale, COLOR_DEFAULT, line_thickness)

    elif class_idx == 1:    # Bottle Opener - Cover
        _outer_shape(img, r)
        #cv.circle(img, (cx, cy), r, COLOR_DEFAULT, line_thickness)
        cr = int(round(0.14 * r))
        cv.circle(img, (cx-int(round(0.68*r)), cy), cr, COLOR_DEFAULT, line_thickness)
        cv.circle(img, (cx+int(round(0.68*r)), cy), cr, COLOR_DEFAULT, line_thickness)

        p1 = (cx - int(round(0.4 * r)), cy + int(round(0.20 * r)))
        p2 = (cx + int(round(0.4 * r)), p1[1])
        p3 = (p1[0], p1[1] + int(round(0.45 * r)))
        p4 = (p2[0], p3[1])

        # top horizontal line
        cv.line(img, p1, p2, COLOR_DEFAULT, line_thickness)

        # bottom horizontal line
        cv.line(img, p3, p4, COLOR_DEFAULT, line_thickness)

        # left vertical line
        cv.line(img, p1, p3, COLOR_DEFAULT, line_thickness)

        # right vertical line
        cv.line(img, p2, p4, COLOR_DEFAULT, line_thickness)

        (w, h), _ = cv.getTextSize(class_name, cv.FONT_HERSHEY_SIMPLEX, txt_scale, line_thickness)
        cv.putText(img, class_name, (cx -w//2, cy + int(round(0.5*r))), cv.FONT_HERSHEY_SIMPLEX, txt_scale, COLOR_DEFAULT, line_thickness)

    elif class_idx == 2:  # Bottle Opener
        _outer_shape(img, r)
        #cv.circle(img, (cx, cy), r, COLOR_DEFAULT, line_thickness)
        cr = int(round(0.06 * r))
        cv.circle(img, (cx-int(round(0.7*r)), cy), cr, COLOR_DEFAULT, line_thickness)
        cv.circle(img, (cx+int(round(0.7*r)), cy), cr, COLOR_DEFAULT, line_thickness)

        cv.circle(img, (cx, cy -int(round(0.42*r))), int(round(0.52 * r)), COLOR_DEFAULT, line_thickness)

        p1x = cx - int(round(0.4 * r))
        p1y = cy + int(round(0.23 * r))
        p2x = cx + int(round(0.4 * r))
        p2y = p1y
        cv.line(img, (p1x, p1y), (p2x, p2y), COLOR_DEFAULT, line_thickness)
        p1y += int(round(0.39 * r))
        p2y = p1y
        cv.line(img, (p1x, p1y), (p2x, p2y), COLOR_DEFAULT, line_thickness)
        p1x = cx - int(round(0.4 * r))
        p1y = cy + int(round(0.23 * r))
        p2x = p1x
        p2y = p1y + int(round(0.39 * r))
        cv.line(img, (p1x, p1y), (p2x, p2y), COLOR_DEFAULT, line_thickness)
        p1x = cx + int(round(0.4 * r))
        p2x = p1x
        cv.line(img, (p1x, p1y), (p2x, p2y), COLOR_DEFAULT, line_thickness)

        (w, h), _ = cv.getTextSize(class_name, cv.FONT_HERSHEY_SIMPLEX, txt_scale, line_thickness)
        cv.putText(img, class_name, (cx -w//2, cy + int(round(0.5*r))), cv.FONT_HERSHEY_SIMPLEX, txt_scale, COLOR_DEFAULT, line_thickness)

    elif class_idx == 3:  # Cuboid
        p1x = margin
        p1y = img.shape[0] - margin
        p2x = margin
        p2y = margin
        p3x = img.shape[1] - margin
        p3y = img.shape[0] - margin
        cv.line(img, (p1x, p1y), (p2x, p2y), COLOR_DEFAULT, line_thickness*2)
        cv.line(img, (p1x, p1y), (p3x, p3y), COLOR_DEFAULT, line_thickness*2)

        (w, h), baseline = cv.getTextSize(class_name, cv.FONT_HERSHEY_SIMPLEX, 5*txt_scale, line_thickness)
        cv.putText(img, class_name, (p1x+(p3x - p1x)//2 - w//2, p1y+(p2y-p1y)//2 + baseline + h//2), cv.FONT_HERSHEY_SIMPLEX, 5 * txt_scale, COLOR_DEFAULT, line_thickness*2)

        if draw_points:
            (w, h), _ = cv.getTextSize('P1', cv.FONT_HERSHEY_SIMPLEX, txt_scale, line_thickness)
            p1x = p1x + margin
            p1y = p1y - margin
            cv.putText(img, 'P1', (p1x, p1y), cv.FONT_HERSHEY_SIMPLEX, txt_scale, COLOR_DEFAULT, line_thickness)

            (w, h), _ = cv.getTextSize('P2', cv.FONT_HERSHEY_SIMPLEX, txt_scale, line_thickness)
            p2x = p2x + margin
            p2y = p2y + margin
            cv.putText(img, 'P2', (p2x, p2y), cv.FONT_HERSHEY_SIMPLEX, txt_scale, COLOR_DEFAULT, line_thickness)

            (w, h), _ = cv.getTextSize('P3', cv.FONT_HERSHEY_SIMPLEX, txt_scale, line_thickness)
            p3x = p3x - w
            p3y = p3y - margin
            cv.putText(img, 'P3', (p3x, p3y), cv.FONT_HERSHEY_SIMPLEX, txt_scale, COLOR_DEFAULT, line_thickness)
    pass

def _add_overlay(img, overlay, position, alpha=1.0):
    x, y = position
    h, w = overlay.shape[0], overlay.shape[1]

    for c in range(0, 3):
        sx = x
        ex = x + w if x + w < img.shape[1] else img.shape[1]
        sy = y
        ey = y + h if y + h < img.shape[0] else img.shape[0]

        if ex <= sx or ey <= sy:
            continue

        img[sy:ey, sx:ex, c] = (alpha * overlay[0:ey-sy, 0:ex-sx, c] +
                                (1 - alpha) * img[sy:ey, sx:ex, c])
    return img

def draw_key(img, key:str, position, color, key_size=100):
    w, h = 200, 200
    sub_img = np.full((h, w, 3), 255, np.uint8)
    line_thickness = max(2, int(round(w / 20)))
    txt_scale = 3.
    txt_thickness = max(1, int(round(w / 20)))
    r = w // 10

    p1 = (r + line_thickness // 2, r + line_thickness // 2)
    p2 = (w - r - line_thickness // 2, r + line_thickness // 2)
    p3 = (w - r - line_thickness // 2, h - r - line_thickness // 2)
    p4 = (r + line_thickness // 2, h - r - line_thickness // 2)

    cv.ellipse(sub_img, p1, (r, r), 180, 0, 90, color, line_thickness)
    cv.ellipse(sub_img, p2, (r, r), 270, 0, 90, color, line_thickness)
    cv.ellipse(sub_img, p3, (r, r), 0, 0, 90, color, line_thickness)
    cv.ellipse(sub_img, p4, (r, r), 90, 0, 90, color, line_thickness)

    cv.line(sub_img, (r, line_thickness//2), (w-r, line_thickness//2), color, line_thickness)
    cv.line(sub_img, (w - line_thickness//2, r), (w - line_thickness//2, h - r), color, line_thickness)
    cv.line(sub_img, (w - r, h - line_thickness//2), (r, h - line_thickness//2), color, line_thickness)
    cv.line(sub_img, (line_thickness//2, h - r), (line_thickness//2, r), color, line_thickness)

    if key == 'arrow left':
        pts = np.array([
            (w//2 + 30, h//2 - 60),
            (w//2 - 35, h//2),
            (w//2 + 30, h//2 + 60)
        ])
        cv.fillPoly(sub_img, [pts], color)
    elif key == 'arrow right':
        pts = np.array([
            (w//2 - 30, h//2 - 60),
            (w//2 + 35, h//2),
            (w//2 - 30, h//2 + 60)
        ])
        cv.fillPoly(sub_img, [pts], color)
    elif key == 'arrow up':
        pts = np.array([
            (w//2 - 60, h//2 + 30),
            (w//2, h//2 - 35),
            (w//2 + 60, h//2 + 30)
        ])
        cv.fillPoly(sub_img, [pts], color)
    elif key == 'arrow down':
        pts = np.array([
            (w//2 - 60, h//2 - 30),
            (w//2, h//2 + 35),
            (w//2 + 60, h//2 - 30)
        ])
        cv.fillPoly(sub_img, [pts], color)
    elif len(key) > 1:
        txt_scale = 1.7
        (tw, th), _ = cv.getTextSize(key, cv.FONT_HERSHEY_SIMPLEX, txt_scale, txt_thickness)
        cv.putText(sub_img, key, (w//2 - tw//2, h//2 + th//2), cv.FONT_HERSHEY_SIMPLEX, txt_scale, color, txt_thickness)
    else:
        (tw, th), _ = cv.getTextSize(key, cv.FONT_HERSHEY_SIMPLEX, txt_scale, txt_thickness)
        cv.putText(sub_img, key, (w//2 - tw//2, h//2 + th//2), cv.FONT_HERSHEY_SIMPLEX, txt_scale, color, txt_thickness)
        

    sub_img = cv.resize(sub_img, (key_size, key_size), interpolation=cv.INTER_AREA)
    img = _add_overlay(img, sub_img, position, alpha=1.0)
    return img

def draw_mouse(img, button, position, color, size=100):
    w, h = 120, 200
    sub_img = np.full((h, w, 3), 255, np.uint8)

    line_thickness = max(2, int(round(h / 20)))
    txt_scale = 3.
    txt_thickness = max(1, int(round(h / 20)))

    r1 = w // 2 - line_thickness // 2
    r2 = int(0.8 * r1)
    c1 = (w // 2, r2 + line_thickness // 2)
    c2 = (w // 2, h - r2 - line_thickness // 2)

    color_marked = (255, 200, 200)
    if button == 'left':
        cv.ellipse(sub_img, c1, (r1, r2), 0, 180, 270, color_marked, -1)
        cv.rectangle(sub_img, (c1[0]-r1, c1[1]), (c2[0], h//2), color_marked, -1)
    elif button == 'right':
        cv.ellipse(sub_img, c1, (r1, r2), 0, 270, 360, color_marked, -1)
        cv.rectangle(sub_img, (c1[0], c1[1]), (c2[0]+r1, h//2), color_marked, -1)

    cv.ellipse(sub_img, c1, (r1, r2), 0, 180, 360, color, line_thickness)
    cv.ellipse(sub_img, c2, (r1, r2), 0, 0, 180, color, line_thickness)

    cv.line(sub_img, (line_thickness//2, c1[1]), (line_thickness//2, c2[1]), color, line_thickness)
    cv.line(sub_img, (w-line_thickness//2, c1[1]), (w-line_thickness//2, c2[1]), color, line_thickness)

    cv.line(sub_img, (w//2, line_thickness//2), (w//2, h//2), color, line_thickness)
    cv.line(sub_img, (w//2, 4*line_thickness), (w//2, h//2-4*line_thickness), color, 3*line_thickness)

    side = np.full((h, (h-w)//2, 3), 255, np.uint8)
    sub_img = cv.hconcat([side, sub_img, side])

    sub_img = cv.resize(sub_img, (0, 0), fx=size/h, fy=size/h, interpolation=cv.INTER_AREA)

    img = _add_overlay(img, sub_img, position, alpha=1.0)
    return img

def _key_options_image(width, margin, selected, txt_scale, txt_thickness):
    img_keyoptions = np.full((int(round(0.15 * width)), width, 3), (255, 255, 255), np.uint8)

    def _draw_key_info(img, key, info, position, color, txt_scale):
        (tw, th), baseline = cv.getTextSize(info, cv.FONT_HERSHEY_SIMPLEX, txt_scale, txt_thickness)
        key_size = int(1.8 * (th + baseline))
        img = draw_key(img, key, position, color, key_size=key_size)
        img = cv.putText(img, info, (position[0] + key_size + 10, position[1] + key_size // 2 + th // 2), cv.FONT_HERSHEY_SIMPLEX, txt_scale, color, txt_thickness)
        #img = cv.rectangle(img, (position[0], position[1]), (position[0]+key_size+10+tw, position[1]+key_size), COLOR_SELECTED, 1)
        return key_size
    
    def _draw_mouse_info(img, button, info, position, color, key_size):
        img = draw_mouse(img, button, position, color, size=key_size)
        (tw, th), baseline = cv.getTextSize(info, cv.FONT_HERSHEY_SIMPLEX, txt_scale, txt_thickness)
        img = cv.putText(img, info, (position[0] + key_size + 10, position[1] + key_size // 2 + th // 2), cv.FONT_HERSHEY_SIMPLEX, txt_scale, color, txt_thickness)
        #img = cv.rectangle(img, (position[0], position[1]), (position[0]+key_size+10+tw, position[1]+key_size), COLOR_SELECTED, 1)
        return key_size

    ox = img_keyoptions.shape[1] // 5
    px, py = margin, margin
    key_size = _draw_key_info(img_keyoptions, 'Q', 'Quit Annotation Tool', (px, py), COLOR_TEXT, txt_scale)
    
    px += ox
    py = margin
    key_size = _draw_key_info(img_keyoptions, '1', 'Select Image Level', (px, py), COLOR_TEXT if selected != 'image' else COLOR_SELECTED, txt_scale)
    py += key_size + margin
    key_size = _draw_key_info(img_keyoptions, '2', 'Select Annotation Level', (px, py), COLOR_TEXT if selected != 'annotation' else COLOR_SELECTED, txt_scale)
    py += key_size + margin
    key_size = _draw_key_info(img_keyoptions, '3', 'Select Point Level', (px, py), COLOR_TEXT if selected != 'point' else COLOR_SELECTED, txt_scale)
    py += key_size + margin
    key_size = _draw_key_info(img_keyoptions, '4', 'Select Class Level', (px, py), COLOR_TEXT if selected != 'class' else COLOR_SELECTED, txt_scale)
    
    px += ox
    py = margin

    key_size = _draw_key_info(img_keyoptions, 'B', 'Select Next Annotation', (px, py), COLOR_TEXT, txt_scale)
    px += ox
    key_size = _draw_key_info(img_keyoptions, 'C', 'Copy Annotations to next Image', (px, py), COLOR_TEXT, txt_scale)
    px -= ox

    px += key_size + margin
    py += 2 * (key_size + margin)
    key_size = _draw_key_info(img_keyoptions, 'W', '', (px, py), COLOR_TEXT if selected == 'point' else COLOR_GRAY, txt_scale)
    px -= key_size + margin
    py += key_size + margin
    key_size = _draw_key_info(img_keyoptions, 'A', '', (px, py), COLOR_TEXT if selected == 'point' else COLOR_GRAY, txt_scale)
    px += key_size + margin
    key_size = _draw_key_info(img_keyoptions, 'S', '', (px, py), COLOR_TEXT if selected == 'point' else COLOR_GRAY, txt_scale)
    px += key_size + margin
    key_size = _draw_key_info(img_keyoptions, 'D', 'Move Point', (px, py), COLOR_TEXT if selected == 'point' else COLOR_GRAY, txt_scale)
    px -= 2 * (key_size + margin)
    
    px += ox
    py = margin + key_size + margin
    key_size = _draw_key_info(img_keyoptions, '+', 'Add Annotation', (px, py), COLOR_TEXT, txt_scale)
    px += ox
    key_size = _draw_key_info(img_keyoptions, '-', 'Remove Annotation', (px, py), COLOR_TEXT, txt_scale)
    px -= ox

    py += key_size + margin
    _draw_mouse_info(img_keyoptions, 'left', 'Set point', (px, py), COLOR_TEXT if selected == 'point' else COLOR_GRAY, key_size)
    px += ox
    _draw_mouse_info(img_keyoptions, 'right', f'Next {selected}', (px, py), COLOR_TEXT, key_size)

    px -= ox
    py += key_size + margin
    key_size = _draw_key_info(img_keyoptions, 'N', f'Previous {selected}', (px, py), COLOR_TEXT, txt_scale)
    px += ox
    key_size = _draw_key_info(img_keyoptions, 'M', f'Next {selected}', (px, py), COLOR_TEXT, txt_scale)
    return img_keyoptions

def render(img, annotations, meta_info, window_name):
    img_out = img.copy()
    img_info = np.full((img_out.shape[0], int(round(img.shape[1] * 0.4)), 3), (255, 255, 255), np.uint8)
    img_file = os.path.basename(meta_info['img_file'])
    
    txt_thickness = max(2, int(round(img.shape[1] / 500)))
    txt_scale = 0.0015 * img_info.shape[1]
    line_thickness = max(1, int(round(img.shape[1] / 500)))
    margin = int(round(0.01 * img.shape[1]))

    global KEY_OPTION_IMAGES, SHAPE_IMAGES
    if KEY_OPTION_IMAGES is None:
        KEY_OPTION_IMAGES = {
            'image': _key_options_image(img.shape[1] + img_info.shape[1], margin, 'image', txt_scale, txt_thickness),
            'annotation': _key_options_image(img.shape[1] + img_info.shape[1], margin, 'annotation', txt_scale, txt_thickness),
            'point': _key_options_image(img.shape[1] + img_info.shape[1], margin, 'point', txt_scale, txt_thickness),
            'class': _key_options_image(img.shape[1] + img_info.shape[1], margin, 'class', txt_scale, txt_thickness)
        }
    if SHAPE_IMAGES is None:
        SHAPE_IMAGES = {}
        for class_idx in CLASSES.keys():
            if class_idx == 3:  # Cuboid
                shape_img = np.full((2*img.shape[1]//3,2*img.shape[1]//3,3), (0,0,0), np.uint8)
            else:
                r = img.shape[1] // 3
                shape_img = np.full((2*r,2*r,3), (0,0,0), np.uint8)
            draw_shape(shape_img, class_idx, line_thickness, 0)
            SHAPE_IMAGES[class_idx] = shape_img

    py, dy = int(round(0.025 * img.shape[1])), int(round(0.025 * img.shape[1]))

    color = COLOR_TEXT
    cv.putText(img_info, f'Selected: {meta_info["selected"]}', (10, py), cv.FONT_HERSHEY_SIMPLEX, txt_scale, color, txt_thickness)
    py += dy

    color = COLOR_SELECTED if meta_info['selected'] == 'image' else COLOR_TEXT
    cv.putText(img_info, f'1 - Image: {meta_info["img_idx"]+1}/{meta_info["file_cnt"]} ({img_file})', (10, py), cv.FONT_HERSHEY_SIMPLEX, txt_scale, color, txt_thickness)
    py += dy

    color = COLOR_SELECTED if meta_info['selected'] == 'annotation' else COLOR_TEXT
    cv.putText(img_info, f'2 - Annotation: {meta_info["anno_idx"]+1}/{len(annotations)}', (10, py), cv.FONT_HERSHEY_SIMPLEX, txt_scale, color, txt_thickness)
    py += dy

    color = COLOR_SELECTED if meta_info['selected'] == 'point' else COLOR_TEXT
    cv.putText(img_info, f'3 - Point: {meta_info["pt_idx"]+1}/{len(annotations[meta_info["anno_idx"]]["points"])}', (10, py), cv.FONT_HERSHEY_SIMPLEX, txt_scale, color, txt_thickness)
    py += dy

    color = COLOR_SELECTED if meta_info['selected'] == 'class' else COLOR_TEXT
    class_idx = meta_info['class_idx']
    cv.putText(img_info, f'4 - Class: {class_idx} ({CLASSES[class_idx]})', (10, py), cv.FONT_HERSHEY_SIMPLEX, txt_scale, color, txt_thickness)
    py += dy

    percentage = meta_info['annotated_cnt'] / meta_info['total_cnt'] * 100 if meta_info['total_cnt'] > 0 else 0.0
    cv.putText(img_info, f'{meta_info["annotated_cnt"]}/{meta_info["total_cnt"]} ({percentage:.2f}%) annotated', (10, py), cv.FONT_HERSHEY_SIMPLEX, txt_scale, color, txt_thickness)
    py += dy

    # preview image
    preview_img = np.full((img.shape[0] // 3, img.shape[1] // 3, 3), (0,0,0), np.uint8)
    for anno_idx, anno in enumerate(annotations):
        color_anno = COLOR_SELECTED if anno_idx == meta_info['anno_idx'] and meta_info['selected'] == 'annotation' else COLOR_DEFAULT

        shape_img = SHAPE_IMAGES[meta_info['class_idx']].copy()
        if meta_info['class_idx'] in list(CLASSES.keys())[0:3]: # Bottle Opener - Inlay, Bottle Opener - Cover, Bottle Opener
            r = img.shape[1] // 3
            #shape_img = np.full((2*r,2*r,3), (0,0,0), np.uint8)
            src_pts = np.array([
                [r+np.sin(np.radians(240)) * r, r - np.cos(np.radians(240)) * r],
                [r+np.sin(np.radians(120)) * r, r - np.cos(np.radians(120)) * r],
                [r, 0],
            ])
        elif meta_info['class_idx'] == 3:  # Cuboid
            #shape_img = np.full((2*img.shape[1]//3,2*img.shape[1]//3,3), (0,0,0), np.uint8)
            src_pts = np.array([
                [0, shape_img.shape[0]],
                [0, 0],
                [shape_img.shape[1], shape_img.shape[0]]
            ])

        #draw_shape(shape_img, anno['class'], line_thickness, 0)

        dst_pts = np.array([
            rel2abs(img_out, *anno['points'][0]),
            rel2abs(img_out, *anno['points'][1]),
            rel2abs(img_out, *anno['points'][2])
        ])
        M = cv.getAffineTransform(src_pts.astype(np.float32), dst_pts.astype(np.float32))
        shape_img_warped = cv.warpAffine(shape_img, M, (img_out.shape[1], img_out.shape[0]))
        #txt_img_warped = cv.warpAffine(txt_img, M, (img_out.shape[1], img_out.shape[0]))

        mask = shape_img_warped[:,:,0] > 0
        img_out[mask, 0] = color_anno[0]
        img_out[mask, 1] = color_anno[1]
        img_out[mask, 2] = color_anno[2]

        if anno_idx == meta_info['anno_idx']:
            margin = 60
            if meta_info['class_idx'] in list(CLASSES.keys())[0:3]: # Bottle Opener - Inlay, Bottle Opener - Cover, Bottle Opener  
                o1x = -np.sin(np.deg2rad(240)) * margin
                o1y = +np.cos(np.deg2rad(240)) * margin

                o2x = -np.sin(np.deg2rad(120)) * margin
                o2y = +np.cos(np.deg2rad(120)) * margin

                o3x = 0.
                o3y = margin
            elif meta_info['class_idx'] == 3:  # Cuboid
                o1x = margin
                o1y = -margin

                o2x = margin
                o2y = margin

                o3x = -margin
                o3y = -margin

            src_pts[0, 0] += o1x
            src_pts[0, 1] += o1y
            src_pts[1, 0] += o2x
            src_pts[1, 1] += o2y
            src_pts[2, 0] += o3x
            src_pts[2, 1] += o3y


            M = cv.getAffineTransform(dst_pts.astype(np.float32), src_pts.astype(np.float32))
            preview_img = cv.warpAffine(img.copy(), M, (shape_img.shape[1], shape_img.shape[0]))
            cv.putText(preview_img, 'Preview', (10, 80), cv.FONT_HERSHEY_SIMPLEX, 3, COLOR_GRAY, line_thickness)

            draw_shape(preview_img, anno['class'], line_thickness, margin)
                #cv.imshow('preview', preview_img)
            pass

        for pt_idx, pt in enumerate(anno['points']):
            pt = rel2abs(img_out, pt[0], pt[1])
            if meta_info['selected'] == 'point' and meta_info['anno_idx'] == anno_idx and meta_info['pt_idx'] == pt_idx:
                cv.circle(img_out, (pt[0], pt[1]), 15, COLOR_SELECTED, line_thickness)
            
    w, h = img_info.shape[1] - 2 * margin, img_info.shape[1] - 2 * margin
    preview_img = cv.resize(preview_img, (w, h), img_info)

    img_out = cv.hconcat([img_out, img_info])
    img_out[-h:, -w-margin:-margin, :] = preview_img

    # key options
    key_option_img = KEY_OPTION_IMAGES[meta_info['selected']]
    img_out = cv.vconcat([img_out, key_option_img])

    cv.imshow(window_name, img_out)
    key = cv.waitKey(1)

    return key

def set_point(annotations, meta_info, x, y):
    anno_idx = meta_info['anno_idx']
    pt_idx = meta_info['pt_idx']
    annotations[anno_idx]['points'][pt_idx][0] = x
    annotations[anno_idx]['points'][pt_idx][1] = y

def save(img_file, annotations):
    if len(annotations) == 1:
        ann = annotations[0]
        if all(pt[0] == 0 and pt[1] == 0 for pt in ann['points']):
            return  # Do not save annotations with all points at (0,0)
    annotation_file = Path(img_file).with_suffix('.json')
    with open(annotation_file, 'w') as f:
        annotations_save = [
            {
                'class': ann['class'],
                'p1x': ann['points'][0][0],
                'p1y': ann['points'][0][1],
                'p2x': ann['points'][1][0],
                'p2y': ann['points'][1][1],
                'p3x': ann['points'][2][0],
                'p3y': ann['points'][2][1]
            } for ann in annotations
        ]
        json.dump(annotations_save, f, indent=4)

def rel2abs(img, x_rel, y_rel):
    x_abs = int(round(x_rel * img.shape[1]))
    y_abs = int(round(y_rel * img.shape[0]))
    return x_abs, y_abs

def abs2rel(img, x_abs, y_abs):
    x_rel = x_abs / img.shape[1]
    y_rel = y_abs / img.shape[0]
    return x_rel, y_rel

def _annotated_cnt(dataset_directory):
    # cnt = len(list(glob(f'{dataset_directory}/train/*.json'))) +\
    #       len(list(glob(f'{dataset_directory}/val/*.json'))) +\
    #       len(list(glob(f'{dataset_directory}/test/*.json')))
    cnt = 0
    for split in ANNOTATION_SPLITS:
        cnt += len(list(glob(f'{dataset_directory}/{split}/*.json')))
    return cnt

def annotate_dataset(dataset_directory):
    """
    Annotate the dataset located in the specified directory.
    Args:
        dataset_directory (str or Path): Path to the dataset directory.
    """
    # img_files = sorted(
    #     list(glob(os.path.join(dataset_directory, 'train', '*.JPG'))) + 
    #     list(glob(os.path.join(dataset_directory, 'val', '*.JPG'))) +
    #     list(glob(os.path.join(dataset_directory, 'test', '*.JPG')))
    # )
    img_files = []
    for split in ANNOTATION_SPLITS:
        img_files += list(glob(os.path.join(dataset_directory, split, '*.JPG')))
    img_files = sorted(img_files)

    meta_info = {
        'selected': 'point',  # image, annotation, point, class
        'img_file': img_files[0],
        'file_cnt': len(img_files),
        'img_idx': 0,
        'anno_idx': 0,
        'pt_idx': 0,
        'class_idx': 0,
        'total_cnt': len(img_files),
        'annotated_cnt': _annotated_cnt(dataset_directory)
    }
    meta_info['img_file'] = img_files[meta_info['img_idx']]
    mouse = {
        'clicked': False,
        'right_clicked': False,
        'x': -1,
        'y': -1
    }

    window_name = 'Annotation'
    cv.namedWindow(window_name)
    cv.setMouseCallback(window_name, on_mouse, param=mouse)

    if SKIP_ANNOTATED:
        while Path(meta_info['img_file']).with_suffix('.json').exists() == True and meta_info['img_idx'] < meta_info['file_cnt'] - 1:
            meta_info['img_idx'] += 1
            meta_info['img_file'] = img_files[meta_info['img_idx']]

    # START_IMG = '20251219_095505(2).JPG'
    # while Path(meta_info['img_file']).name != START_IMG and meta_info['img_idx'] < meta_info['file_cnt'] - 1:
    #     meta_info['img_idx'] += 1
    #     meta_info['img_file'] = img_files[meta_info['img_idx']]

    img, annotations = load_image(img_files[meta_info['img_idx']])

    meta_info['class_idx'] = annotations[meta_info['anno_idx']]['class']

    while True:
        #img = img_org.copy()
        key = render(img, annotations, meta_info, window_name)

        if mouse['clicked']:
            mouse['clicked'] = False

            if meta_info['selected'] == 'point':
                set_point(annotations, meta_info, mouse['x'] / img.shape[1], mouse['y'] / img.shape[0])
                #save(meta_info['img_file'], annotations)

        elif key == ord('1'):
            meta_info['selected'] = 'image'
        elif key == ord('2'):
            meta_info['selected'] = 'annotation'
        elif key == ord('3'):
            meta_info['selected'] = 'point'
        elif key == ord('4'):
            meta_info['selected'] = 'class'
        elif key == ord('+'):
            annotations.insert(meta_info['anno_idx'], copy.deepcopy(annotations[meta_info['anno_idx']]))
            meta_info['selected'] = 'point'
            meta_info['anno_idx'] = meta_info['anno_idx'] + 1
            meta_info['pt_idx'] = 0
            #save(meta_info['img_file'], annotations)
        elif key == ord('-'):
            if len(annotations) > 1:
                annotations.pop(meta_info['anno_idx'])
                meta_info['anno_idx'] = meta_info['anno_idx'] - 1 if meta_info['anno_idx'] > 0 else 0
            #save(meta_info['img_file'], annotations)
        elif meta_info['selected'] == 'point' and key in [ord('w'), ord('a'), ord('s'), ord('d')]:
            pt = annotations[meta_info['anno_idx']]['points'][meta_info['pt_idx']]
            if key == ord('w'):
                annotations[meta_info['anno_idx']]['points'][meta_info['pt_idx']][1] = pt[1] - 1/1000
            elif key == ord('s'):
                annotations[meta_info['anno_idx']]['points'][meta_info['pt_idx']][1] = pt[1] + 1/1000
            elif key == ord('a'):
                annotations[meta_info['anno_idx']]['points'][meta_info['pt_idx']][0] = pt[0] - 1/1000
            elif key == ord('d'):
                annotations[meta_info['anno_idx']]['points'][meta_info['pt_idx']][0] = pt[0] + 1/1000
            #save(meta_info['img_file'], annotations)
        elif key == 3 or key == ord('m') or mouse['right_clicked']:  # Right arrow key
            mouse['right_clicked'] = False
            if meta_info['selected'] == 'image':
                save(meta_info['img_file'], annotations)
            img, annotations = select_next(img, annotations, meta_info, img_files)
            meta_info['class_idx'] = annotations[meta_info['anno_idx']]['class']
        elif key == 2 or key == ord('n'):  # Left arrow key
            if meta_info['selected'] == 'image':
                save(meta_info['img_file'], annotations)
            img, annotations = select_previous(img, annotations, meta_info, img_files)
            meta_info['class_idx'] = annotations[meta_info['anno_idx']]['class']
        elif key == ord('b') and meta_info['selected'] == 'point':
            meta_info['anno_idx'] = (meta_info['anno_idx'] + 1) % len(annotations)
            meta_info['pt_idx'] = 0
        elif key == ord('c'): # copy annotations to next image
            annotations_copy = copy.deepcopy(annotations)
            save(meta_info['img_file'], annotations)
            meta_info['selected'] = 'image'
            img, annotations = select_next(img, annotations, meta_info, img_files)
            meta_info['selected'] = 'point'
            annotations = annotations_copy
        elif key == ord('q'):
            save(meta_info['img_file'], annotations)
            break
        elif key != -1:
            pass
    cv.destroyAllWindows()
    cv.waitKey(1)

if __name__ == '__main__':
    annotate_dataset('data')

    # img = np.full((500, 800, 3), 255, np.uint8)

    # img = draw_key(img, '1', (50, 50), COLOR_GRAY)
    # img = draw_key(img, 'Enter', (160, 50), COLOR_GRAY)
    # img = draw_key(img, 'Esc', (270, 50), COLOR_GRAY)

    # cv.imshow('image', img)
    # cv.waitKey()