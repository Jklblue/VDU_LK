import json
import os
import os.path as osp
import numpy as np
from labelme import utils

if __name__ == '__main__':
    pngs_path = "datasets/SegmentationClass"
    classes = ["_background_", "rapeseed"]

    count = os.listdir("D:/PhD/SegAnything/mask/")
    for i in range(0, len(count)):
        path = os.path.join("D:/PhD/SegAnything/mask", count[i])

        if os.path.isfile(path) and path.endswith('json'):
            data = json.load(open(path))

            # 创建标签名称到数值的映射
            label_name_to_value = {'_background_': 0}
            for shape in data['segmentation']:
                label_name = shape['label']
                if label_name not in label_name_to_value:
                    label_value = len(label_name_to_value)
                    label_name_to_value[label_name] = label_value

            # 确保标签值是连续的
            label_values, label_names = [], []
            for ln, lv in sorted(label_name_to_value.items(), key=lambda x: x[1]):
                label_values.append(lv)
                label_names.append(ln)
            assert label_values == list(range(len(label_values)))

            # 创建标签图像
            img_shape = (data['imageHeight'], data['imageWidth'])  # 假设json包含图像的高度和宽度信息
            lbl = utils.shapes_to_label(img_shape, data['shapes'], label_name_to_value)

            # 创建新的标签图
            new = np.zeros([np.shape(lbl)[0], np.shape(lbl)[1]])
            for name in label_names:
                index_json = label_names.index(name)
                index_all = classes.index(name)
                new = new + index_all * (np.array(lbl) == index_json)

            # 保存标签图
            utils.lblsave(osp.join(pngs_path, count[i].split(".")[0] + '.png'), new)
            print('Saved ' + count[i].split(".")[0] + '.png')
