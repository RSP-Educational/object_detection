import numpy as np
import cv2 as cv
import random
from datasets import load_dataset

BACKGROUNDS = []
if len(BACKGROUNDS) == 0:
    ds_background = load_dataset("SchulzR97/backgrounds")

    BACKGROUNDS = []
    for record in ds_background["train"]:
        bg_image = np.array(record["image"])
        #bg_image = cv.cvtColor(bg_image, cv.COLOR_RGB2BGR)  # PIL (RGB) → OpenCV (BGR)
        BACKGROUNDS.append(bg_image)

# ----------------- Bild-Augmentationen -----------------
def _aug_brithness(image: np.ndarray, factor: float) -> np.ndarray:
    """
    Changes the brightness of the image by multiplying pixel values with a factor.
    Args:
    image: np.ndarray
        Input image in HWC format with values in [0, 1].
    factor: float
        Brightness factor. Values > 1 increase brightness, values < 1 decrease brightness.
    Returns:
        np.ndarray: Brightness-adjusted image in the same format as the input image.
    """
    return np.clip(image * factor, 0, 1)

def _aug_contrast(image: np.ndarray, factor: float) -> np.ndarray:
    """
    Changes the contrast of the image by scaling pixel values around the mean.
    Args:
        image: np.ndarray
            Input image in HWC format with values in [0, 1].
        factor: float
            Contrast factor. Values > 1 increase contrast, values < 1 decrease contrast.
    Returns:
        np.ndarray: Contrast-adjusted image in the same format as the input image.
    """
    mean = image.mean(axis=(0, 1), keepdims=True)
    return np.clip((image - mean) * factor + mean, 0, 1)

def _aug_saturation(image: np.ndarray, factor: float) -> np.ndarray:
    """
    Changes the saturation of the image.
    Args:
        image: np.ndarray
            Input image in HWC format with values in [0, 1].
        factor: float
            Saturation factor. Values > 1 increase saturation, values < 1 decrease saturation.
    Returns:
        np.ndarray: Saturation-adjusted image in the same format as the input image.
    """
    gray = cv.cvtColor((image * 255).astype(np.uint8), cv.COLOR_BGR2GRAY)
    gray = gray.astype(np.float32) / 255.0
    gray = np.stack([gray, gray, gray], axis=-1)
    return np.clip(gray + factor * (image - gray), 0, 1)

def _aug_hue(image: np.ndarray, shift: float) -> np.ndarray:
    """
    Changes the hue of the image by shifting the hue channel in HSV color space.
    Args:
        image: np.ndarray
            Input image in HWC format with values in [0, 1].
        shift: float
            Hue shift factor. Values in [-1, 1], where 1 corresponds to a full 180° shift in hue.
    Returns:
        np.ndarray: Hue-adjusted image in the same format as the input image.
    """
    img_uint8 = (image * 255).astype(np.uint8)
    hsv = cv.cvtColor(img_uint8, cv.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 0] = (hsv[:, :, 0] + shift * 180) % 180  # Hue ist 0-180 in OpenCV
    hsv = np.clip(hsv, 0, 255).astype(np.uint8)
    return cv.cvtColor(hsv, cv.COLOR_HSV2BGR).astype(np.float32) / 255.0

def _aug_sharpness(image: np.ndarray, factor: float) -> np.ndarray:
    """
    Changes the sharpness of the image by blending it with a blurred version of itself.
    Args:
        image: np.ndarray
            Input image in HWC format with values in [0, 1].
        factor: float
            Sharpness factor. Values > 1 increase sharpness, values < 1 decrease sharpness.
    Returns:
        np.ndarray: Sharpness-adjusted image in the same format as the input image.
    """
    w, h = image.shape[1], image.shape[0]
    ksize = int(min(w, h) / 60) | 1  # ungerade Zahl
    min_ksize = 0
    max_ksize = int(min(w, h) / 40) | 1
    ksize = min_ksize + int((max_ksize - min_ksize) * factor)
    blurred = cv.blur(image, (ksize, ksize))
    blurred = np.clip(blurred, 0, 1)
    return blurred

def augment_image_numpy(
        image: np.ndarray,
        p_brightness:float = 0.5,
        p_contrast:float = 0.5,
        p_saturation:float = 0.5,
        p_hue:float = 0.5,
        p_sharpness:float = 0.5,
        f_brightness:float = 0.3,
        f_contrast:float = 0.4,
        f_saturation:float = 0.6,
        f_hue:float = 0.4,
        f_sharpness:float = 0.3
    ) -> np.ndarray:
    """
    Augment an image (HWC, float32, [0,1])

    Args:
        image: np.ndarray
            Input image with format HWC and values in the range [0, 1].
        p_brightness, p_contrast, p_saturation, p_hue, p_sharpness: float
            Probabilities for applying the respective augmentation.
        f_brightness, f_contrast, f_saturation, f_hue, f_sharpness: float
            Factors controlling the strength of the respective augmentation.
    Returns:
        np.ndarray: Augmented image in the same format as the input image.
    """
    image = image.copy()  # Do not modify the original
    
    # --- Brightness ---
    if random.random() < p_brightness:
        factor = 1.0 - f_brightness + 2 * f_brightness * random.random()
        image = _aug_brithness(image, factor)
    
    # --- Contrast ---
    if random.random() < p_contrast:
        factor = 1.0 - f_contrast + 2 * f_contrast * random.random()
        image = _aug_contrast(image, factor)
    
    # --- Saturation ---
    if random.random() < p_saturation:
        factor = 1.0 - f_saturation + 2 * f_saturation * random.random()
        image = _aug_saturation(image, factor)
    
    # --- Hue ---
    if random.random() < p_hue:
        shift = -f_hue + 2 * f_hue * random.random()
        image = _aug_hue(image, shift)
    
    # --- Sharpness ---
    if random.random() < p_sharpness:
        factor = 0.05 + f_sharpness * random.random()
        image = _aug_sharpness(image, factor)
    
    return image

# ----------------- Perspective Distortion -----------------
def augment_perspective(image:np.ndarray, keypoints:np.ndarray, distortion=0.1, p:float=0.5):
    """
    Applies a random perspective transformation to the image and keypoints.
    Args:
        image: np.ndarray
            Input image in HWC format with values in [0, 1].
        keypoints: np.ndarray
            Keypoints associated with the image, shape (N, 2), values in [0, 1].
        distortion: float
            Maximum distortion factor (0 to 1).
        p: float
            Probability of applying the perspective transformation.
    Returns:
        Tuple[np.ndarray, np.ndarray]: Transformed image and keypoints.
    """
    def _random_perspective_matrix(w, h, distortion=0.1):
        """
        Generates a random perspective transformation matrix.
        Args:
            w: int
                Width of the image
            h: int
                Height of the image
            distortion: float
                Maximum distortion factor (0 to 1)
        Returns:
            np.ndarray: 3x3 perspective transformation matrix
        """

        # Original image corners
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

# ----------------- Rotation and Scaling -----------------
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

def replace_background(image: np.ndarray):
    """
    Replace background in specified hue range with random images from the specified directory.
    Args:
    image: np.ndarray
        Input image in HWC format with values in [0, 1].
    Returns:
        np.ndarray: Image with replaced background in the same format as the input image.
    """


    # --- Hintergrundmaske erstellen ---
    hsv = cv.cvtColor((image * 255).astype(np.uint8), cv.COLOR_BGR2HSV)
    # Graubereiche haben niedrige Sättigung (Saturation)
    lower_hue = np.array([0, 0, 0])  # Hue: beliebig, Saturation: niedrig, Value: beliebig
    upper_hue = np.array([180, 35, 255])  # Grauwerte haben Sättigung < 50
    mask = cv.inRange(hsv, lower_hue, upper_hue)
    mask_inv = cv.bitwise_not(mask)

    # --- Zufälliges Hintergrundbild laden ---
    bg_image = random.choice(BACKGROUNDS)
    bg_image = cv.resize(bg_image, (image.shape[1], image.shape[0]))
    bg_image = bg_image.astype(np.float32) / 255.0

    # --- Hintergrund ersetzen ---
    fg = cv.bitwise_and(image, image, mask=mask_inv)
    bg = cv.bitwise_and(bg_image, bg_image, mask=mask)
    combined = fg + bg

    return combined

if __name__ == "__main__":
    import visualization as vis
    import data as data

    while True:
        sample_name = "images/samples/P1030157.json"

        img, points, _ = data.load_record(sample_name)

        img_bg = replace_background(img.copy())

        vis.plot_images_with_points(
            [img, img_bg],
            target_points_list=[points]*2,
            titles=[
                "Original",
                "Hintergrund ersetzt"
            ]
        )