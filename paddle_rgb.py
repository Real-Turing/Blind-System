import argparse

import cv2
import numpy as np
import os
import paddlex.utils.logging as logging
import time


import paddlex as pdx

class paddle_rgb:
    def __init__(self, parent=None):
        # self.model =pdx.load_model('.\P0003-T0003_export_model\inference_model\inference_model')
        # self.predictor = pdx.deploy.Predictor('./P0009_model/inference_model', use_gpu=True)
        self.predictor = pdx.deploy.Predictor('./P07_T012_model/inference_model', use_gpu=True)

        # result = predictor.predict(img_file='542_color.png')
        print("Model loaded.")


    def parse_args(self):
        parser = argparse.ArgumentParser(
            description='HumanSeg inference for video')
        parser.add_argument(
            '--model_dir',
            dest='model_dir',
            help='Model path for inference',
            type=str)
        parser.add_argument(
            '--video_path',
            dest='video_path',
            help='Video path for inference, camera will be used if the path not existing',
            type=str,
            default=None)
        parser.add_argument(
            '--save_dir',
            dest='save_dir',
            help='The directory for saving the inference results',
            type=str,
            default='./output')
        parser.add_argument(
            "--image_shape",
            dest="image_shape",
            help="The image shape for net inputs.",
            nargs=2,
            default=[192, 192],
            type=int)

        return parser.parse_args()

    def seg_infer_from_img(self,img,trans):

        im = img.astype('float32')
        result=self.predictor.predict(im)#得到盲道与人行道的
        result_vis=self.visualize(im, result, weight=trans)
        # image_name = str(int(time.time() * 1000)) + '.png'
        # cv2.imwrite(image_name, result_vis)
        # print("predict.")
        return result_vis

    def visualize(self,image,result,
                           weight=0.6,
                           save_dir=None,
                           color=None):
        """
           Convert segment result to color image, and save added image.
           Args:
               image: the path of origin image
               result: the predict result of image
               weight: the image weight of visual image, and the result weight is (1 - weight)
               save_dir: the directory for saving visual image
               color: the list of a BGR-mode color for each label.
           """
        label_map = result['label_map'].astype("uint8")
        color_map = self.get_color_map_list(256)
        if color is not None:
            for i in range(len(color) // 3):
                color_map[i] = color[i * 3:(i + 1) * 3]
        color_map = np.array(color_map).astype("uint8")

        # Use OpenCV LUT for color mapping
        c1 = cv2.LUT(label_map, color_map[:, 0])
        c2 = cv2.LUT(label_map, color_map[:, 1])
        c3 = cv2.LUT(label_map, color_map[:, 2])
        pseudo_img = np.dstack((c1, c2, c3))

        if isinstance(image, np.ndarray):
            im = image
            image_name = str(int(time.time() * 1000)) + '.jpg'
            if image.shape[2] != 3:
                logging.info(
                    "The image is not 3-channel array, so predicted label map is shown as a pseudo color image."
                )
                weight = 0.
        else:
            image_name = os.path.split(image)[-1]
            if not self.is_pic(image):
                logging.info(
                    "The image cannot be opened by opencv, so predicted label map is shown as a pseudo color image."
                )
                image_name = image_name.split('.')[0] + '.jpg'
                weight = 0.
            else:
                im = cv2.imread(image)
        # vis_result = pseudo_img

        # if abs(weight) < 1e-5:
        #     vis_result = pseudo_img
        # else:
        vis_result = cv2.addWeighted(im, weight,
                                         pseudo_img.astype(im.dtype), 1 - weight,
                                         0)

        # if save_dir is not None:
        #     if not os.path.exists(save_dir):
        #         os.makedirs(save_dir)
        #     out_path = os.path.join(save_dir, 'visualize_{}'.format(image_name))
        #     cv2.imwrite(out_path, vis_result)
        #     logging.info('The visualized result is saved as {}'.format(out_path))
        # else:
        # image_name = str(int(time.time() * 1000)) + '.png'
        # cv2.imwrite(image_name, vis_result)
        return vis_result


    def get_color_map_list(self,num_classes):
        """ Returns the color map for visualizing the segmentation mask,
            which can support arbitrary number of classes.
        Args:
            num_classes: Number of classes
        Returns:
            The color map
        """
        color_map = num_classes * [0, 0, 0]
        for i in range(0, num_classes):
            j = 0
            lab = i
            while lab:
                color_map[i * 3] |= (((lab >> 0) & 1) << (7 - j))
                color_map[i * 3 + 1] |= (((lab >> 1) & 1) << (7 - j))
                color_map[i * 3 + 2] |= (((lab >> 2) & 1) << (7 - j))
                j += 1
                lab >>= 3
        color_map = [color_map[i:i + 3] for i in range(0, len(color_map), 3)]
        return color_map

    def is_pic(self,img_name):
        valid_suffix = ['JPEG', 'jpeg', 'JPG', 'jpg', 'BMP', 'bmp', 'PNG', 'png']
        suffix = img_name.split('.')[-1]
        if suffix not in valid_suffix:
            return False
        return True

