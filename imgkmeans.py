import numpy as np
import cv2
import matplotlib.pyplot as plt


def image_kmeans_mask(image_path, k, color_space='LAB', normalize_input=True,
                      num_attempts=3, max_iterations=100, threshold=0.0001):
    """
    Perform K-means clustering on the specified image and return the mask.

    Parameters:
        image_path: Path to the image file
        k: Number of clusters
        color_space: Color space, optional 'RGB' or 'LAB'
        normalize_input: Whether to normalize the input
        num_attempts: Number of algorithm attempts
        max_iterations: Maximum number of iterations per attempt
        threshold: Convergence threshold

    Returns:
        mask: Clustering mask, with the same size as the input image
        original_image: Original image (in RGB format)
        segmented_image: Segmented result image


    """
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Unable to read the image file: {image_path}")


    if not (isinstance(k, int) and k > 0):
        raise ValueError("The clustering number k must be a positive integer.")


    if color_space not in ['RGB', 'LAB']:
        raise ValueError("The color space must be 'RGB' or 'LAB'")


    if color_space == 'RGB':
        processed_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:  # LAB
        processed_img = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)


    original_dtype = processed_img.dtype


    processed_img = processed_img.astype(np.float32)


    height, width, channels = processed_img.shape
    X = processed_img.reshape(-1, channels)


    if X.shape[0] < k:
        raise ValueError("聚类数量k大于像素数量")


    avg_chn = np.zeros(channels, dtype=np.float32)
    std_dev_chn = np.ones(channels, dtype=np.float32)

    if normalize_input:
        avg_chn = np.mean(X, axis=0)
        std_dev_chn = np.std(X, axis=0)


        zero_loc = std_dev_chn == 0
        std_dev_chn[zero_loc] = 1


        X_normalized = (X - avg_chn) / std_dev_chn
    else:
        X_normalized = X.copy()


    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, max_iterations, threshold)


    _, labels_flat, norm_centers = cv2.kmeans(
        X_normalized, k, None, criteria, num_attempts, cv2.KMEANS_RANDOM_CENTERS
    )


    mask = labels_flat.reshape(height, width)


    centers = norm_centers * std_dev_chn + avg_chn
    centers = centers.astype(original_dtype)


    if k <= np.iinfo(np.uint8).max:
        mask = mask.astype(np.uint8)
    elif k <= np.iinfo(np.uint16).max:
        mask = mask.astype(np.uint16)
    elif k <= np.iinfo(np.uint32).max:
        mask = mask.astype(np.uint32)
    else:
        mask = mask.astype(np.float64)


    original_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    segmented_image = np.zeros_like(original_image)

    if color_space == 'LAB':

        centers_bgr = []
        for center in centers:
            lab_pixel = np.uint8(center).reshape(1, 1, 3)
            bgr_pixel = cv2.cvtColor(lab_pixel, cv2.COLOR_LAB2BGR)
            centers_bgr.append(bgr_pixel[0, 0])
        centers = np.array(centers_bgr, dtype=np.uint8)


        for i in range(k):
            mask_i = (mask == i)
            segmented_image[mask_i] = cv2.cvtColor(np.array([[centers[i]]]), cv2.COLOR_BGR2RGB)[0, 0]
    else:

        for i in range(k):
            mask_i = (mask == i)
            segmented_image[mask_i] = centers[i].astype(np.uint8)

    return mask, original_image, segmented_image


def show_single_class_image(original_image, mask, target_class):
    """
    Generate and return an RGB image with "only the target class displayed and other classes in black"

    Parameters:
        original_image: The original RGB image (obtained from the image_kmeans_mask function)
        mask: The clustering mask (obtained from the image_kmeans_mask function)
        target_class: The serial number of the target class (should be within the range of 0 to k-1, where k is the number of clusters)

    Returns:
        single_class_image: An RGB image with only the target class visible and others in black
    """

    valid_classes = np.unique(mask)
    if target_class not in valid_classes:
        raise ValueError(f"Target category{target_class}doesn't exist. The legal categories are{valid_classes}")


    target_mask = (mask == target_class)

    single_class_image = original_image.copy()
    single_class_image[~target_mask] = (0, 0, 0)

    return single_class_image


if __name__ == "__main__":

    image_path = "img/DJI_0200_1.jpg"  # Replace with your image path
    k = 3
    color_space = 'LAB'


    mask, original_img, segmented_img = image_kmeans_mask(image_path, k, color_space)

    single_class_img1 = show_single_class_image(original_img, mask, 0)
    single_class_img2 = show_single_class_image(original_img, mask, 1)
    single_class_img3 = show_single_class_image(original_img, mask, 2)


    plt.figure(figsize=(18, 10))

    plt.subplot(221)
    plt.imshow(original_img)
    plt.title("Class1")
    plt.axis("off")


    plt.subplot(222)
    plt.imshow(single_class_img1)
    plt.title("Class1")
    plt.axis("off")

    plt.subplot(223)
    plt.imshow(single_class_img2)
    plt.title("Class1")
    plt.axis("off")

    plt.subplot(224)
    plt.imshow(single_class_img3)
    plt.title("Class1")
    plt.axis("off")


    plt.tight_layout()
    plt.show()
