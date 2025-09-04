import numpy as np
import cv2
import matplotlib.pyplot as plt


def image_kmeans_mask(image_path, k, color_space='LAB', normalize_input=True,
                      num_attempts=3, max_iterations=100, threshold=0.0001):
    """
    对指定图像执行K-means聚类并返回掩码

    参数:
        image_path: 图像文件路径
        k: 聚类数量
        color_space: 颜色空间，可选'RGB'或'LAB'
        normalize_input: 是否对输入进行归一化
        num_attempts: 算法尝试次数
        max_iterations: 每次尝试的最大迭代次数
        threshold: 收敛阈值

    返回:
        mask: 聚类掩码，与输入图像同尺寸
        original_image: 原始图像(RGB格式)
        segmented_image: 分割结果图像
    """
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"无法读取图像文件: {image_path}")

    # 验证聚类数量
    if not (isinstance(k, int) and k > 0):
        raise ValueError("聚类数量k必须是正整数")

    # 验证颜色空间
    if color_space not in ['RGB', 'LAB']:
        raise ValueError("颜色空间必须是'RGB'或'LAB'")

    # 转换到指定的颜色空间
    if color_space == 'RGB':
        processed_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:  # LAB
        processed_img = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # 保存原始数据类型
    original_dtype = processed_img.dtype

    # 转换为float32
    processed_img = processed_img.astype(np.float32)

    # 获取图像尺寸并重塑为像素点集 (n_samples, channels)
    height, width, channels = processed_img.shape
    X = processed_img.reshape(-1, channels)

    # 检查聚类数量是否小于像素数量
    if X.shape[0] < k:
        raise ValueError("聚类数量k大于像素数量")

    # 数据归一化
    avg_chn = np.zeros(channels, dtype=np.float32)
    std_dev_chn = np.ones(channels, dtype=np.float32)

    if normalize_input:
        avg_chn = np.mean(X, axis=0)
        std_dev_chn = np.std(X, axis=0)

        # 处理标准差为0的情况
        zero_loc = std_dev_chn == 0
        std_dev_chn[zero_loc] = 1

        # 标准化
        X_normalized = (X - avg_chn) / std_dev_chn
    else:
        X_normalized = X.copy()

    # 定义K-means的终止条件
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, max_iterations, threshold)

    # 执行K-means聚类
    _, labels_flat, norm_centers = cv2.kmeans(
        X_normalized, k, None, criteria, num_attempts, cv2.KMEANS_RANDOM_CENTERS
    )

    # 将标签重塑为图像尺寸（掩码）
    mask = labels_flat.reshape(height, width)

    # 对聚类中心进行反归一化
    centers = norm_centers * std_dev_chn + avg_chn
    centers = centers.astype(original_dtype)

    # 为标签选择最节省内存的数据类型
    if k <= np.iinfo(np.uint8).max:
        mask = mask.astype(np.uint8)
    elif k <= np.iinfo(np.uint16).max:
        mask = mask.astype(np.uint16)
    elif k <= np.iinfo(np.uint32).max:
        mask = mask.astype(np.uint32)
    else:
        mask = mask.astype(np.float64)

    # 创建分割结果图像
    original_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    segmented_image = np.zeros_like(original_image)

    if color_space == 'LAB':
        # 转换聚类中心回BGR
        centers_bgr = []
        for center in centers:
            lab_pixel = np.uint8(center).reshape(1, 1, 3)
            bgr_pixel = cv2.cvtColor(lab_pixel, cv2.COLOR_LAB2BGR)
            centers_bgr.append(bgr_pixel[0, 0])
        centers = np.array(centers_bgr, dtype=np.uint8)

        # 生成分割图像
        for i in range(k):
            mask_i = (mask == i)
            segmented_image[mask_i] = cv2.cvtColor(np.array([[centers[i]]]), cv2.COLOR_BGR2RGB)[0, 0]
    else:
        # RGB空间直接生成分割图像
        for i in range(k):
            mask_i = (mask == i)
            segmented_image[mask_i] = centers[i].astype(np.uint8)

    return mask, original_image, segmented_image


def show_single_class_image(original_image, mask, target_class):
    """
    生成并返回“仅显示目标类别、其他类别为黑色”的RGB图像

    参数:
        original_image: 原始RGB图像（从image_kmeans_mask函数获取）
        mask: 聚类掩码（从image_kmeans_mask函数获取）
        target_class: 目标类别序号（需在0~k-1范围内，k为聚类数）

    返回:
        single_class_image: 仅目标类别可见、其他为黑色的RGB图像
    """
    # 验证目标类别是否合法
    valid_classes = np.unique(mask)
    if target_class not in valid_classes:
        raise ValueError(f"目标类别{target_class}不存在，合法类别为{valid_classes}")

    # 生成掩码：目标类别为True，其他为False
    target_mask = (mask == target_class)
    # 复制原始图像，非目标类别设为黑色（RGB=(0,0,0)）
    single_class_image = original_image.copy()
    single_class_image[~target_mask] = (0, 0, 0)  # ~表示取反（非目标类别）

    return single_class_image


if __name__ == "__main__":
    # 示例用法：直接调用函数获取掩码
    image_path = "img/DJI_0200_1.jpg"  # 替换为你的图像路径
    k = 3  # 聚类数量
    color_space = 'LAB'  # 可选 'LAB' 或 'RGB'

    # 调用函数获取掩码、原始图像和分割结果
    mask, original_img, segmented_img = image_kmeans_mask(image_path, k, color_space)

    single_class_img1 = show_single_class_image(original_img, mask, 0)
    single_class_img2 = show_single_class_image(original_img, mask, 1)
    single_class_img3 = show_single_class_image(original_img, mask, 2)

    # 显示结果
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
