import numpy as np
import cv2 as cv
import random

# ----------------- Bild-Augmentationen -----------------
def _aug_brithness(image: np.ndarray, factor: float) -> np.ndarray:
    """Ändert die Helligkeit des Bildes."""
    return np.clip(image * factor, 0, 1)

def _aug_contrast(image: np.ndarray, factor: float) -> np.ndarray:
    """Ändert den Kontrast des Bildes."""
    mean = image.mean(axis=(0, 1), keepdims=True)
    return np.clip((image - mean) * factor + mean, 0, 1)

def _aug_saturation(image: np.ndarray, factor: float) -> np.ndarray:
    """Ändert die Sättigung des Bildes."""
    gray = cv.cvtColor((image * 255).astype(np.uint8), cv.COLOR_BGR2GRAY)
    gray = gray.astype(np.float32) / 255.0
    gray = np.stack([gray, gray, gray], axis=-1)
    return np.clip(gray + factor * (image - gray), 0, 1)

def _aug_hue(image: np.ndarray, shift: float) -> np.ndarray:
    """Ändert den Farbton des Bildes."""
    img_uint8 = (image * 255).astype(np.uint8)
    hsv = cv.cvtColor(img_uint8, cv.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 0] = (hsv[:, :, 0] + shift * 180) % 180  # Hue ist 0-180 in OpenCV
    hsv = np.clip(hsv, 0, 255).astype(np.uint8)
    return cv.cvtColor(hsv, cv.COLOR_HSV2BGR).astype(np.float32) / 255.0

def _aug_sharpness(image: np.ndarray, factor: float) -> np.ndarray:
    """Ändert die Schärfe des Bildes."""
    w, h = image.shape[1], image.shape[0]
    ksize = int(min(w, h) / 60) | 1  # ungerade Zahl
    min_ksize = 0
    max_ksize = int(min(w, h) / 40) | 1
    ksize = min_ksize + int((max_ksize - min_ksize) * factor)
    blurred = cv.blur(image, (ksize, ksize))
    blurred = np.clip(blurred, 0, 1)
    return blurred

def augment_image_numpy(image: np.ndarray) -> np.ndarray:
    """
    Augmentiert ein Bild (HWC, float32, [0,1])

    Args:
        image (np.ndarray): Eingabebild im Format HWC mit Werten im Bereich [0, 1].
    Returns:
        np.ndarray: Augmentiertes Bild im gleichen Format wie das Eingabebild.
    """
    image = image.copy()  # Nicht das Original verändern
    
    # --- Helligkeit ---
    if random.random() < 0.5:
        factor = 0.7 + 0.6 * random.random()  # [0.7, 1.3]
        image = _aug_brithness(image, factor)
    
    # --- Kontrast ---
    if random.random() < 0.5:
        factor = 0.6 + 0.8 * random.random()  # [0.6, 1.4]
        image = _aug_contrast(image, factor)
    
    # --- Sättigung ---
    if random.random() < 0.5:
        factor = 0.4 + 1.2 * random.random()  # [0.4, 1.6]
        image = _aug_saturation(image, factor)
    
    # --- Hue (Farbton) ---
    if random.random() < 0.5:
        shift = -0.4 + 0.8 * random.random()  # [-0.4, 0.4]
        image = _aug_hue(image, shift)
    
    # --- Schärfe ---
    if random.random() < 0.5:
        factor = 0.2 + 0.4 * random.random()  # [0.2, 0.6]
        image = _aug_sharpness(image, factor)
    
    return image

# ----------------- Perspektivische Verzerrung -----------------
def augment_perspective(image:np.ndarray, keypoints:np.ndarray, distortion=0.1, p:float=0.5):
    def _random_perspective_matrix(w, h, distortion=0.1):
        """
        w, h: Bildbreite, -höhe
        distortion: max. relative Verschiebung (z.B. 0.1 = 10%)
        """

        # Originale Bild-Ecken
        src = np.float32([
            [0, 0],
            [w - 1, 0],
            [w - 1, h - 1],
            [0, h - 1],
        ])

        def jitter(pt):
            return [
                pt[0] + random.uniform(-distortion, distortion) * w,
                pt[1] + random.uniform(-distortion, distortion) * h,
            ]

        dst = np.float32([jitter(p) for p in src])

        H = cv.getPerspectiveTransform(src, dst)
        return H
    
    def _augment_perspective(H, image:np.ndarray, points:np.ndarray, p:float=0.5):     
        def _warp_image(image, H):
            h, w = image.shape[:2]
            warped = cv.warpPerspective(
                image,
                H,
                (w, h),
                flags=cv.INTER_LINEAR,
                borderMode=cv.BORDER_REFLECT_101
            )
            return warped
        
        def _warp_points(points_norm, H):
            """
            points: (N, 2) normierte Punkte [0,1]
            Rückgabe: (N, 2) normierte Punkte
            """
            pts = points_norm.copy()

            # → homogene Koordinaten
            ones = np.ones((pts.shape[0], 1))
            pts_h = np.hstack([pts, ones])  # (N, 3)

            # Transformation
            pts_warped = (H @ pts_h.T).T

            # Homogen zurück zu 2D
            pts_warped = pts_warped[:, :2] / pts_warped[:, 2:3]

            return pts_warped

        if random.random() > p:
            return image, points

        h, w = image.shape[:2]

        image_warped = _warp_image(image, H)
        # points[:, 0] *= w
        # points[:, 1] *= h
        points_copy = points.copy()
        points_copy[:, 0] *= w
        points_copy[:, 1] *= h
        points_warped = _warp_points(points_copy, H)
        points_warped[:, 0] /= w
        points_warped[:, 1] /= h

        return image_warped, points_warped

    h, w = image.shape[:2]
    H = _random_perspective_matrix(w, h, distortion)

    image_warped, keypoints_warped = _augment_perspective(H, image, keypoints, p=p)

    return image_warped, keypoints_warped

# ----------------- Rotation und Skalierung -----------------
def _augment_rotate(img, keypoints, angle, scale):
    def rotate_keypoints_relative(
        keypoints_rel: np.ndarray,  # shape (N, 2), values in [0,1]
        angle_deg: float,
        scale:float,
        w:int,
        h:int
    ):
        angle = np.deg2rad(angle_deg)

        keypoints_abs = keypoints_rel.copy()
        keypoints_abs[:, 0] *= w
        keypoints_abs[:, 1] *= h

        keypoints_abs[:, 0] -= w / 2
        keypoints_abs[:, 1] -= h / 2

        keypoints_rotated = keypoints_abs.copy()
        keypoints_rotated[:, 0] = keypoints_abs[:, 0] * np.cos(angle) + keypoints_abs[:, 1] * np.sin(angle)
        keypoints_rotated[:, 1] = -keypoints_abs[:, 0] * np.sin(angle) + keypoints_abs[:, 1] * np.cos(angle)

        keypoints_rotated[:, 0] *= scale
        keypoints_rotated[:, 1] *= scale

        keypoints_rotated[:, 0] += w / 2
        keypoints_rotated[:, 1] += h / 2

        keypoints_rotated[:, 0] /= w
        keypoints_rotated[:, 1] /= h

        return keypoints_rotated

    def rotate_image(img: np.ndarray, angle_deg: float, scale: float):
        h, w = img.shape[:2]
        cx, cy = w / 2, h / 2

        M = cv.getRotationMatrix2D((cx, cy), angle_deg, scale)
        rotated = cv.warpAffine(
            img, M, (w, h),
            flags=cv.INTER_LINEAR,
            borderMode=cv.BORDER_CONSTANT,
            borderValue=(.85, .85, .85)
        )
        return rotated
    
    h, w = img.shape[:2]
    
    img_rotated = rotate_image(img.copy(), angle, scale)
    keypoints_abs = keypoints.copy()
    keypoints_rotated = rotate_keypoints_relative(keypoints_abs.copy(), angle, scale, w, h)

    return img_rotated, keypoints_rotated

def augment_rotate(img, keypoints, angle_shift, scale_factor, p:float = 0.5):
    if random.random() > p:
        return img, keypoints
    
    angle = -angle_shift + 2 * angle_shift * random.random()
    scale = 1.0 - scale_factor + 2 * scale_factor * random.random()

    img_rotated, keypoints_rotated = _augment_rotate(img, keypoints, angle, scale)
    return img_rotated, keypoints_rotated