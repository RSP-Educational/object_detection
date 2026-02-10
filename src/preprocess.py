from pathlib import Path
if __name__ == "__main__":
    import sys
    sys.path.append(str(Path(__file__).parent.parent))

import cv2 as cv
from tqdm import tqdm
from glob import glob
from src.constants import COLOR_RED, COLOR_WHITE
import shutil
import os
import numpy as np
import json

def rename_images_suffix(
        dataset_directory:str, 
        src_suffixes = ['.jpeg', '.JPEG', '.png', '.jpg', '.PNG'], 
        target_suffix = '.JPG', 
        splits=['train', 'val', 'test']
    ):
    """
    Rename image files in the dataset directory from source suffixes to target suffix.
    Args:
        dataset_directory (str): Path to the dataset directory.
        src_suffixes (list): List of source suffixes to be renamed.
        target_suffix (str): Target suffix to rename the files to.
        splits (list): List of dataset splits (subdirectories) to process.
    """
    img_files = []
    for split in splits:
        split_path = Path(dataset_directory) / split
        for suffix in src_suffixes:
            img_files.extend(glob(str(split_path / f'*{suffix}')))

    backup_dir = Path(dataset_directory) / 'backup'
    backup_dir.mkdir(parents=True, exist_ok=True)

    for img_file in tqdm(img_files, desc='Renaming images'):
        split = Path(img_file).parent.name
        fname_backup = backup_dir / Path(img_file).name
        fname_backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(img_file, fname_backup)

        fname_new = Path(img_file).with_suffix(target_suffix)
        shutil.move(img_file, fname_new)

def adjust_image_proportions(dataset_directory:str, target_prop:float=3264/4896, splits=['train', 'val', 'test']):
    def cut_img(img, target_prop, offset):
        h, w = img.shape[:2]
        prop = h / w
        if prop - target_prop > 0.:
            new_h = int(w * target_prop)
            sx = 0
            ex = w
            sy = (h - new_h) // 2 + offset
            ey = sy + new_h
            img_new_h = img[sy:ey, sx:ex].copy()
            return img_new_h
        elif prop - target_prop < -0.:
            new_w = int(h / target_prop)
            sx = (w - new_w) // 2 + offset
            ex = sx + new_w
            sy = 0
            ey = h
            img_new_w = img[sy:ey, sx:ex].copy()
            return img_new_w
        return img.copy()
    
    def render(img_org, img_new, offset, fname):
        img_org = img_org.copy()
        img_new = img_new.copy()

        cv.putText(img_org, f'{fname}, offset: {offset}', (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, COLOR_WHITE, 2)

        if img_org.shape[0] > img_new.shape[0]:
            sy = (img_org.shape[0] - img_new.shape[0]) // 2 + offset
            ey = sy + img_new.shape[0]
            img_org = cv.line(img_org, (0, sy), (img_org.shape[1], sy), COLOR_RED, 5)
            img_org = cv.line(img_org, (0, ey), (img_org.shape[1], ey), COLOR_RED, 5)
            img_org = cv.line(img_org, (0, img_org.shape[0]), (img_org.shape[1], img_org.shape[0]), COLOR_WHITE, 5)

            img_up = cv.vconcat([img_org, img_new])
        elif img_org.shape[1] > img_new.shape[1]:
            sx = (img_org.shape[1] - img_new.shape[1]) // 2 + offset
            ex = sx + img_new.shape[1]
            img_org = cv.line(img_org, (sx, 0), (sx, img_org.shape[0]), COLOR_RED, 5)
            img_org = cv.line(img_org, (ex, 0), (ex, img_org.shape[0]), COLOR_RED, 5)
            img_org = cv.line(img_org, (img_org.shape[1], 0), (img_org.shape[1], img_org.shape[0]), COLOR_WHITE, 5)

            img_up = cv.hconcat([img_org, img_new])
        else:
            img_up = cv.hconcat([img_org, img_new])

        cv.imshow('img_up', img_up)
        key = cv.waitKey()
        return key

    def save(fname, img_org, img_new):
        split = Path(fname).parent.name
        backup_dir = Path(fname).parent.parent.joinpath('backup')
        os.makedirs(backup_dir, exist_ok=True)
        fname_backup = str(backup_dir / Path(fname).name)
        fname_new = fname

        cv.imwrite(fname_backup, img_org)
        cv.imwrite(fname_new, img_new)

    def delete(fname, img_org, img_new):
        split = Path(fname).parent.name
        backup_dir = Path(fname).parent.parent.joinpath('backup', split)
        os.makedirs(backup_dir, exist_ok=True)

        fname_backup = str(backup_dir / Path(fname).name)
        fname_new = fname

        # cv.imshow('delete img_new', img_new)
        # cv.imshow('delete img_org', img_org)
        # cv.waitKey()

        cv.imwrite(fname_backup, img_org)
        os.remove(fname_new)

        print(f'Deleted: {fname_new}')

    def load_next(JPG_files, img_idx, prog:tqdm):
        while True:
            img_idx += 1
            if img_idx >= len(JPG_files):
                return None, None, img_idx

            # set counter of progress bar as int
            prog.n = int(img_idx + 1)
            prog.update((img_idx + 1) / len(JPG_files))        

            img_org = cv.imread(JPG_files[img_idx])
            h, w = img_org.shape[:2]
            if max(h, w) > 1500:
                f = 1500 / max(h, w)
                img_org = cv.resize(img_org, (0, 0), fx=f, fy=f)

            img_new = img_org.copy()

            h, w = img_new.shape[:2]
            prop = h / w
            if abs(prop - target_prop) >= 0.15:
                return img_org, img_new, img_idx

    dataset_directory = Path(dataset_directory)

    JPG_files = []
    for split in splits:
        JPG_files.extend(list(sorted(glob(str(dataset_directory / split / '*.JPG')))))
    img_idx = -1
    offset = 0

    prog = tqdm(total=len(JPG_files))
    img_org, img_new, img_idx = load_next(JPG_files, img_idx, prog)

    while True:
        # show progress based on img_idx
        prog.n = int(img_idx + 1)
        prog.update(img_idx + 1)

        if img_org is None:
            print('All done!')
            break

        h, w = img_new.shape[:2]
        if h > w:
            img_new = cut_img(img_org, target_prop, offset)

        key = render(img_org, img_new, offset, JPG_files[img_idx])

        if key == ord('1'):    # 1
            offset -= 10
            img_new = cut_img(img_org, target_prop, offset)
        elif key == ord('2'):  # 2
            offset += 10
            img_new = cut_img(img_org, target_prop, offset)
        elif key == ord('n'):
            img_org, img_new, img_idx = load_next(JPG_files, img_idx, prog)
            #offset = 0
        elif key == 13:  # enter
            save(JPG_files[img_idx], img_org, img_new)
            img_org, img_new, img_idx = load_next(JPG_files, img_idx, prog)
            #offset = 0
        elif key in [127, 40]:   # backspace or delete
            delete(JPG_files[img_idx], img_org, img_new)
            img_org, img_new, img_idx = load_next(JPG_files, img_idx, prog)
            #offset = 0
        elif key == ord('q'):
            break
        elif key != -1:
            pass

def split_data(dataset_directory:str, train_ratio:float=0.8, val_ratio:float=0.1, test_ratio:float=0.1):
    """
    Split dataset into train, validation, and test sets based on specified ratios.
    Args:
        dataset_directory (str): Path to the dataset directory.
        train_ratio (float): Ratio of training data.
        val_ratio (float): Ratio of validation data.
        test_ratio (float): Ratio of test data.
    """
    backup_dir = Path(dataset_directory) / 'backup'
    backup_dir.mkdir(parents=True, exist_ok=True)

    annotation_files = \
        list(glob(str(Path(dataset_directory) / 'data/*.json'))) +\
        list(glob(str(Path(dataset_directory) / 'train/*.json'))) +\
        list(glob(str(Path(dataset_directory) / 'val/*.json'))) +\
        list(glob(str(Path(dataset_directory) / 'test/*.json')))
    
    random_indices = np.random.choice(range(len(annotation_files)), len(annotation_files), replace=False)

    train_cnt = int(round(len(annotation_files) * train_ratio))
    val_cnt = int(round(len(annotation_files) * val_ratio))
    test_cnt = len(annotation_files) - train_cnt - val_cnt
    annotations_train = [annotation_files[i] for i in random_indices[:train_cnt]]
    annotations_val = [annotation_files[i] for i in random_indices[train_cnt:train_cnt + val_cnt]]
    annotations_test = [annotation_files[i] for i in random_indices[train_cnt + val_cnt:]]

    for ann_file in annotations_train:
        img_file = Path(ann_file).with_suffix('.JPG')

        # backup annotation and image files
        backup_ann = backup_dir / Path(ann_file).name
        backup_img = backup_dir / img_file.name
        shutil.copy2(ann_file, backup_ann)
        shutil.copy2(img_file, backup_img)

        # move annotation and image files to train directory
        dst_ann_file = Path(dataset_directory) / 'train' / Path(ann_file).name
        dst_img_file = Path(dataset_directory) / 'train' / img_file.name
        dst_ann_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(ann_file, dst_ann_file)
        shutil.move(img_file, dst_img_file)

    for ann_file in annotations_val:
        img_file = Path(ann_file).with_suffix('.JPG')

        # backup annotation and image files
        backup_ann = backup_dir / Path(ann_file).name
        backup_img = backup_dir / img_file.name
        shutil.copy2(ann_file, backup_ann)
        shutil.copy2(img_file, backup_img)

        # move annotation and image files to val directory
        dst_ann_file = Path(dataset_directory) / 'val' / Path(ann_file).name
        dst_img_file = Path(dataset_directory) / 'val' / img_file.name
        dst_ann_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(ann_file, dst_ann_file)
        shutil.move(img_file, dst_img_file)

    for ann_file in annotations_test:
        img_file = Path(ann_file).with_suffix('.JPG')

        # backup annotation and image files
        backup_ann = backup_dir / Path(ann_file).name
        backup_img = backup_dir / img_file.name
        shutil.copy2(ann_file, backup_ann)
        shutil.copy2(img_file, backup_img)

        # move annotation and image files to test directory
        dst_ann_file = Path(dataset_directory) / 'test' / Path(ann_file).name
        dst_img_file = Path(dataset_directory) / 'test' / img_file.name
        dst_ann_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(ann_file, dst_ann_file)
        shutil.move(img_file, dst_img_file)

    # json_train = dataset_directory + '/train.json'
    # json_val = dataset_directory + '/val.json'
    # json_test = dataset_directory + '/test.json'

    # with open(json_train, 'w') as f:
    #     json.dump(files_train, f, indent=4)
    # with open(json_val, 'w') as f:
    #     json.dump(files_val, f, indent=4)
    # with open(json_test, 'w') as f:
    #     json.dump(files_test, f, indent=4)

    print(f'DATA SPLIT SUMMARY')
    print(f"{'Total files:':<15} {len(annotation_files):>4}")
    print(f"{'├─ train:':<15} {len(annotations_train):>4} ({len(annotations_train)/len(annotation_files):.2%})")
    print(f"{'├─ val:':<15} {len(annotations_val):>4} ({len(annotations_val)/len(annotation_files):.2%})")
    print(f"{'└─ test:':<15} {len(annotations_test):>4} ({len(annotations_test)/len(annotation_files):.2%})")

def convert_data(dataset_directory):
    backup_directory = Path(dataset_directory) / 'backup'
    backup_directory.mkdir(parents=True, exist_ok=True)

    annotation_files = \
        list(glob(str(Path(dataset_directory) / 'train/*.json'))) +\
        list(glob(str(Path(dataset_directory) / 'val/*.json'))) +\
        list(glob(str(Path(dataset_directory) / 'test/*.json')))
    annotations = []

    for ann_file in tqdm(annotation_files, desc='Converting data'):
        split = Path(ann_file).parent.name
        backup_split_dir = backup_directory / split
        backup_split_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ann_file, backup_split_dir / Path(ann_file).name)
        with open(ann_file, 'r') as f:
            annotations = json.load(f)

        # if 'p1x' in annotations[0]:
        #     continue

        annotations_new = []

        for ann in annotations:
            ann_new = {
                'image': str(Path(ann_file).with_suffix('.JPG').name),
                'class': ann['class'],
                'p1x': ann['p1x'], 'p1y': ann['p1y'],
                'p2x': ann['p2x'], 'p2y': ann['p2y'],
                'p3x': ann['p3x'], 'p3y': ann['p3y']
            }
            annotations_new.append(ann_new)
            pass
        with open(ann_file, 'w') as f:
            json.dump(annotations_new, f, indent=4)

if __name__ == "__main__":
    dataset_directory = '/Users/schulzr/Documents/Datasets/DHSN_BottleOpener'
    #rename_images_suffix(dataset_directory)
    #adjust_image_proportions(dataset_directory)

    #convert_data(dataset_directory)

    #split_data(dataset_directory, train_ratio=0.8, val_ratio=0.2, test_ratio=0.0)