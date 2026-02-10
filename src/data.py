import json
import numpy as np
import cv2 as cv
import torch
import torchvision
from torch.utils.data import Dataset
from pathlib import Path
from huggingface_hub import snapshot_download
from glob import glob
from typing import List, Tuple
from tqdm import tqdm

REPO_ID = "SchulzR97/DHSN_BottleOpener"
LOCAL_DIR = 'data/DHSN_BottleOpener'
CLASS_LABELS = [
    'Inlay',
    'Cover',
    'Bottle Opener',
    'Cuboid'
]

def load_record(ann_file: str) -> dict:
    """Lädt eine Annotationsdatei im COCO-Format und gibt die Daten als Dictionary zurück."""
    img_file = ann_file.replace(".json", ".JPG")

    with open(ann_file, 'r') as f:
        annotations = json.load(f)
    points = np.array([pt for obj in annotations for pt in obj['points']])
    classes = np.array([obj['class'] for obj in annotations])

    img = cv.imread(img_file) / 255.0  # Normalisiere auf [0, 1]

    return img, points, classes

def load_points_classes(ann_file: str) -> Tuple[List, List]:
    """
    Lädt Annotationspunkte und Klassen aus einer JSON-Datei.

    Args:
        ann_file (str): Pfad zur Annotationsdatei.
    Returns:
        Tuple[List, List]: Liste der Punkte und Liste der Klassen."""
    
    with open(ann_file, 'r') as f:
        annotations = json.load(f)
    points = np.array([[[obj['p1x'], obj['p1y']], [obj['p2x'], obj['p2y']], [obj['p3x'], obj['p3y']]] for obj in annotations])
    points = points.reshape(-1, 2)
    classes = np.array([obj['class'] for obj in annotations])
    return points, classes

def download_dataset(force_download:bool = False) -> str:
    """
    Lädt das Dataset von Hugging Face herunter und gibt den lokalen Pfad zurück.
    Args:
        force_download (bool): Wenn True, wird das Dataset erneut heruntergeladen,
                               auch wenn es bereits lokal vorhanden ist.
    Returns:
        str: Lokaler Pfad zum heruntergeladenen Dataset.
    """
    local_dir = snapshot_download(repo_id=REPO_ID, repo_type="dataset", local_dir=LOCAL_DIR, force_download=force_download)
    return local_dir

def load_annotations(split:str) -> str:
    """
    Lädt die Annotationsdateien für den angegebenen Split.
    Args:
        split (str): Der Datensatz-Split, z.B. "train", "val", "test".
    Returns:
        Liste der Annotationsdateien.
    """
    local_dir = Path(LOCAL_DIR) / split
    annotation_files = list(glob(str(local_dir / "*.json")))

    annotations = []
    for ann_file in tqdm(annotation_files, desc=f"Loading annotations", leave=False):
        points, classes = load_points_classes(ann_file)
        annotations.append({
            "id": Path(ann_file).stem,
            "img_file": str(ann_file.replace(".json", ".JPG")),
            "ann_file": str(ann_file),
            "points": points,
            "classes": classes
        })

    return annotations

def publish_dataset(dataset_directory, splits = ['train', 'val', 'test']):
    """
    Publish the annotated dataset located in the specified directory to HuggingFace.
    Args:
        dataset_directory (str or Path): Path to the dataset directory.
    """
    def _login():
        from huggingface_hub import login
        import getpass

        token_file = Path(dataset_directory) / '.hf_token'
        if token_file.exists():
            with open(token_file, 'rb') as f:
                token = f.read().strip()
            try:
                login(token.decode())
                return
            except Exception as e:
                print(f"Stored HuggingFace token is invalid: {e}")
        token = getpass.getpass("Enter your HuggingFace token: ")
        login(token)
        with open(token_file, 'wb') as f:
            f.write(token.encode())

    from huggingface_hub import list_repo_files, create_commit, CommitOperationAdd

    _login()

    # upload annotations and images
    repo_files = list_repo_files(REPO_ID, repo_type='dataset')
    repo_annotations = [f"{split}/{Path(file).name}" for split in splits for file in repo_files if file.startswith(f"{split}/") and file.endswith('.json')]

    #local_data_dir = Path(dataset_directory) / 'data'
    local_annotations = [f"{split}/{Path(file).name}" for split in splits for file in glob(f"{dataset_directory}/{split}/*.json")]

    N_COMMIT = 20
    new_annotations = local_annotations# [f for f in local_annotations if f not in repo_annotations]
    total_cnt = len(new_annotations) + len(repo_annotations)
    operations = []
    for i, ann_id in enumerate(new_annotations):
        local_annotation_path = f"{dataset_directory}/{ann_id}"
        local_image_path = local_annotation_path.replace('.json', '.JPG')

        image_id = ann_id.replace('.json', '.JPG')

        operations.append(CommitOperationAdd(path_in_repo=ann_id, path_or_fileobj=local_annotation_path))
        #operations.append(CommitOperationAdd(path_in_repo=image_id, path_or_fileobj=local_image_path))
        if (i + 1) % N_COMMIT == 0 or (i + 1) == len(new_annotations):
            create_commit(
                repo_id=REPO_ID,
                repo_type='dataset',
                operations=operations,
                commit_message=f"Add {len(operations)} annotation files."
            )
            operations = []
            print(f"Committed {len(repo_annotations) + i + 1}/{total_cnt} ({(len(repo_annotations) + i+1)/total_cnt*100:.2f}%) annotation files.")
    pass

class ObjectDataset(Dataset):
    CLASS_LABELS = CLASS_LABELS

    def __init__(
            self,
            split:str,
            image_size:Tuple[int, int]=(1024, 1024),
            return_filename:bool = False
        ):
        super().__init__()

        assert split in ["train", "val", "test"], f"Split '{split}' is invalid. Expected 'train', 'val' or 'test'."
       
        self.split = split
        self.annotations = load_annotations(split=split)
        self.NUM_CLASSES = max([max(ann['classes'] + 1) for ann in self.annotations]) + 1 # inner +1 because index-based; outer +1 because background class = 0
        self.NUM_KEYPOINTS = self.annotations[0]['points'].shape[0]
        self.return_filename = return_filename

        self.image_size = image_size
        self.normalize = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )

    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, index):
        try:
            import cv2 as cv
            import src.augmentation as aug

            ann = self.annotations[index]
            img = cv.imread(ann["img_file"]) / 255.
            if img is None:
                raise ValueError(f"Could not read image: {ann['img_file']}")
            img = cv.resize(img, self.image_size)

            keypoints = ann["points"]

            if self.split == "train":
                img, keypoints = aug.augment_perspective(img, keypoints, distortion=0.05, p=0.3)
                img, keypoints = aug.augment_rotate(img, keypoints, angle_shift=10, scale_factor=0.05, p=0.3)
                img = aug.augment_image_numpy(img)

            boxes = []
            N_obj = keypoints.shape[0] // 3
            keypoints_formatted = np.zeros((N_obj, 3, 3))
            
            
            # Konvertiere relative Koordinaten in absolute Pixelkoordinaten
            img_height, img_width = self.image_size[1], self.image_size[0]
            
            for i, kpts in enumerate(keypoints.copy().reshape(N_obj, 3, 2)):
                # Skaliere Keypoints auf absolute Pixelkoordinaten
                kpts_abs = kpts.copy()
                kpts_abs[:, 0] *= img_width   # x-Koordinaten
                kpts_abs[:, 1] *= img_height  # y-Koordinaten
                
                # Berechne Bounding Box aus den Keypoints (in Pixelkoordinaten)
                # WICHTIG: Margin proportional zur Bildgröße für bessere Generalisierung
                # und um sicherzustellen, dass Keypoints nach Augmentation nicht abgeschnitten werden
                margin_ratio = 0.15  # 15% der Keypoint-Spanne als Margin
                kpt_width = kpts_abs[:, 0].max() - kpts_abs[:, 0].min()
                kpt_height = kpts_abs[:, 1].max() - kpts_abs[:, 1].min()
                margin_x = max(30, kpt_width * margin_ratio)  # Minimum 30 Pixel
                margin_y = max(30, kpt_height * margin_ratio)
                
                xmin, xmax = kpts_abs[:, 0].min() - margin_x, kpts_abs[:, 0].max() + margin_x
                ymin, ymax = kpts_abs[:, 1].min() - margin_y, kpts_abs[:, 1].max() + margin_y
                
                # Clip auf Bildgrenzen
                xmin, ymin = max(0, xmin), max(0, ymin)
                xmax, ymax = min(img_width, xmax), min(img_height, ymax)
                
                boxes.append([xmin, ymin, xmax, ymax])
                
                # Formatiere Keypoints für Keypoint R-CNN: [K, 3] wobei 3 = (x, y, visibility)
                # visibility: 0 = nicht annotiert, 1 = annotiert aber nicht sichtbar, 2 = annotiert und sichtbar
                kpts_with_visibility = np.concatenate([kpts_abs, np.ones((3, 1)) * 2], axis=1)  # alle Keypoints sind sichtbar
                keypoints_formatted[i] = kpts_with_visibility

            img = torch.tensor(img).permute(2, 0, 1).float()
            img = self.normalize(img)

            target = {
                "boxes": torch.tensor(boxes).float(),
                "keypoints": torch.tensor(keypoints_formatted).float(),  # Shape: [N, 3, 3] für N Objekte mit je 3 Keypoints
                "labels": torch.tensor(ann["classes"], dtype=torch.int64) + 1  # +1, because background class = 0
            }

            if self.return_filename:
                return img, target, ann["img_file"]
            return img, target
        except Exception as e:
            print(f"Exception: {e}")

    def collate_fn(batch):
        return tuple(zip(*batch))
    
    def get_weighted_sampler(self):
        """Erstellt Sampler basierend auf der seltensten Klasse pro Bild"""
        from collections import Counter
        
        # Zähle Objekte pro Klasse
        class_counts = Counter()
        for ann in self.annotations:
            for label in ann['classes']:
                class_counts[label] += 1
        
        print(f"\nClass distribution: {dict(class_counts)}")
        
        # Berechne Gewichte (inverse frequency)
        total = sum(class_counts.values())
        class_weights = {}
        for label, count in class_counts.items():
            class_weights[label] = total / (len(class_counts) * count)
        
        print(f"Class weights: {class_weights}")
        
        # Jedes Bild bekommt das HÖCHSTE Gewicht seiner Klassen (meistens sowieso nur eine Klasse pro Bild)
        sample_weights = []
        for ann in self.annotations:
            labels = ann['classes']
            # Nutze die seltenste Klasse im Bild
            weights = [class_weights[label] for label in labels]
            sample_weights.append(max(weights))
        
        return torch.utils.data.WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )

if __name__ == "__main__":
    publish_dataset('/Users/schulzr/Documents/Datasets/DHSN_BottleOpener')