import json
import numpy as np
import cv2 as cv
from pathlib import Path
from huggingface_hub import snapshot_download
from glob import glob
from typing import List, Tuple


REPO_ID = "SchulzR97/DHSN_BottleOpener"
LOCAL_DIR = 'data/DHSN_BottleOpener'

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
    for ann_file in annotation_files:
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

if __name__ == "__main__":
    publish_dataset('/Users/schulzr/Documents/Datasets/DHSN_BottleOpener')