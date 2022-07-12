import argparse

import cv2
import numpy as np
import os
import paddlex.utils.logging as logging
import time
import obs_detection as od
import process_seged as pseg
import paddlex as pdx
import datetime as dt
import math


class Obstacle:
    position=(0,0)
    distance=10.0
    angle=0.0
    width=-1.0


class Perception:
    blind_road_find = 1 # 是否发现盲道
    blind_road_position = (0,0) # 盲道位置
    blind_road_distance = 0.5# 盲道与使用者的距离
    blind_road_angle = -30 # 盲道在使用者的左边30度角位置
    on_blind_road = 1 # 使用者是否位于盲道上
    blind_road_departure=0 # 使用者是否正在偏离盲道

    sideways_find = 1 # 是否发现人行道
    sideways_distance = 3.5 # 人行道与使用者的距离
    sideways_angle = 30 # 人行道在使用者的右边30度角位置
    sideways_move_towrds=0#人行道走向
    on_sideways = 1 # 使用者是否位于人行道上
    sideways_departure=0 # 使用者是否正在偏离人行道

    obs_list=[]
    obs_all_find = 0 # care_meter距离内是否发现障碍物
    obs_front_find = 0 # care_meter距离内行走正前方是否发现障碍物
    closest_obs_distance = 0.5# 最近的前方可能相碰障碍物与使用者的距离
    closest_obs_angle = -10 # 最近的前方可能相碰障碍物与使用者的角度

def get_angle(x,y,depth):
    d = depth[int(y), int(width / 2)]
    L = 2 * d * np.tan(theta / 2)
    l = L / width * (width / 2 - x)
    return np.arctan(l / d)
def visualize(image, result,
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
    color_map = get_color_map_list(256)
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
        if not is_pic(image):
            logging.info(
                "The image cannot be opened by opencv, so predicted label map is shown as a pseudo color image."
            )
            image_name = image_name.split('.')[0] + '.jpg'
            weight = 0.
        else:
            im = cv2.imread(image)
    # vis_result = pseudo_img

    if abs(weight) < 1e-5:
        vis_result = pseudo_img
    else:
        vis_result = cv2.addWeighted(im, weight,
                                     pseudo_img.astype(im.dtype), 1 - weight,
                                     0)
    return vis_result


def get_color_map_list(num_classes):
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


def is_pic(self, img_name):
    valid_suffix = ['JPEG', 'jpeg', 'JPG', 'jpg', 'BMP', 'bmp', 'PNG', 'png']
    suffix = img_name.split('.')[-1]
    if suffix not in valid_suffix:
        return False
    return True

def pred_batch():
    images_path="./video_forwalkdir/"
    for image_name in os.listdir(images_path):
        # 更改颜色
        if image_name.endswith('png'):
            im = cv2.imread(images_path+image_name)
            result = predictor.predict(im)
            result_vis = visualize(im, result, weight=1.0)
            cv2.imwrite(images_path+"p_" + image_name + "_color_vis.png", result_vis)


predictor = pdx.deploy.Predictor('./P0008-T0014_export_model/inference_model/inference_model', cpu_thread_num=4)
    # predictor = pdx.deploy.Predictor('../P0009_model/inference_model', use_gpu=True)
    # predictor = pdx.deploy.Predictor('../P07_T012_model/inference_model', use_gpu=True)
    # predictor = pdx.deploy.Predictor('../rotate_model_final/inference_model_new/inference_model')
datapath='./2022-03-26new/'
now_date = dt.datetime.now().strftime('%F') + "new"
now_time = dt.datetime.now().strftime('%F_%H%M%S')
path_ok = os.path.exists(now_date)
if (path_ok == False):
    os.mkdir(now_date)
video_path = os.path.join('./', now_date, now_time + '_proc.avi')
videopath, videoname = os.path.split(video_path)
four_cc = cv2.VideoWriter_fourcc(*'XVID')
video_writer = cv2.VideoWriter(video_path, four_cc, float(10), (1280, 720))
deviation=[]    #列表存储#连续三帧判断是否偏移

for i in range(173,174):
    print(i)
    im=cv2.imread(datapath+str(i)+'_color.png')
    depth=np.load(datapath+str(i)+'depth_image_real.npy')
    rect_center,acceptedRects=od.obs_detection_file(datapath+str(i)+'depth_image_real.npy')

    # 获取图像长宽
    height, width = im.shape[:2]
    theta = 69.4/180*np.pi
    #语义分割盲道
    result = predictor.predict(im)
    perception=Perception()
    result_vis = visualize(im, result, weight=0.5)
    only_seged_for_sideway = visualize(im, result, weight=0.0)
    only_seged_for_blind=visualize(im, result, weight=0.0)
    #中心点
    cv2.circle(result_vis, (int(width / 2), int(height / 2)), 5, (0, 0, 255), 8)
    #获取人行道信息
    perception_sideways=pseg.process_sideways(only_seged_for_sideway,depth,width,height)
    perception.sideways_find=perception_sideways['sideways_find']
    cv2.putText(result_vis, 'sideways_find:' + str(perception.sideways_find), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 0, 255), 2)

    if perception.sideways_find:
        perception.sideways_distance=perception_sideways['sideways_distance']
        sideways_x=int(perception_sideways['sideways_x'])
        sideways_y=int(perception_sideways['sideways_y'])

        sideways_min_l=perception_sideways['sideways_min_l']
        sideways_max_l=perception_sideways['sideways_max_l']
        sideways_min_r=perception_sideways['sideways_min_r']
        sideways_max_r=perception_sideways['sideways_max_r']

        perception.sideways_angle=get_angle(sideways_x,sideways_y,depth)
        perception.sideways_move_towrds=perception_sideways['sideways_move_towards']
        #显示人行道信息
        cv2.circle(result_vis, (sideways_x, sideways_y), 5, (0, 255, 0), 8)
        cv2.putText(result_vis, 'sideways_distance:'+str(perception.sideways_distance), (sideways_x, sideways_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(result_vis, 'sideways_angle:'+str(perception.sideways_angle), (sideways_x, sideways_y+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        # cv2.line(result_vis,sideways_min_l ,sideways_max_l, (0, 0, 255), 2)
        # cv2.line(result_vis,sideways_min_r ,sideways_max_r, (0, 0, 255), 2)

    #获取盲道信息
    perception_blind=pseg.process_blind_road(only_seged_for_blind,depth,width,height)
    perception.blind_road_find=perception_blind['blind_road_find']
    cv2.putText(result_vis, 'blind_road_find:'+str(perception.blind_road_find), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    if perception.blind_road_find:
        perception.blind_road_distance=perception_blind['blind_road_distance']
        perception.blind_road_position=blind_road_x,blind_road_y=perception_blind['blind_road_position']
        perception.blind_road_angle=get_angle(blind_road_x,blind_road_y,depth)
        cv2.circle(result_vis, (blind_road_x, blind_road_y), 5, (0, 255, 0), 8)
        cv2.putText(result_vis, 'blind_road_find:' + str(perception.blind_road_find), (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 255), 2)
        cv2.putText(result_vis, 'blind_road_distance:' + str(perception.blind_road_distance),
                    (blind_road_x - 100, blind_road_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(result_vis,"blind_road_angle:"+str(perception.blind_road_angle),(blind_road_x,blind_road_y+20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2)
        # cv2.putText(result_vis, 'blind_road_position:' + str(perception.blind_road_position),
        #             (blind_road_x - 100, blind_road_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # #偏离盲道预警
    # #左右边界
    # left_x=perception_blind['left_x'][0]
    # right_x=perception_blind['right_x'][0]
    # left_y=perception_blind['left_x'][1]
    # right_y=perception_blind['right_x'][1]
    # #图像中心点
    # middle_x=0.5*(right_x-left_x)+left_x
    # middle_x_rela=0.5*(right_x-left_x)
    #
    # #警告阈值
    # TH=0.8
    # #计算LOR
    # # LOR=(min(abs(right_x-middle_x),abs(middle_x-left_x))-TH*middle_x)/(TH*middle_x)
    # LOR=(min(abs(right_x-middle_x),abs(middle_x-left_x))-TH*middle_x_rela)/(TH*middle_x_rela)
    #
    # cv2.circle(result_vis,(left_x,left_y) , 50, (0, 255, 255), 8)
    # cv2.circle(result_vis,(right_x,right_y), 50, (0, 255, 255), 8)
    # print(middle_x)
    # print(right_x)
    # print(left_x)
    # print(LOR)
    #偏离盲道预警
    #左右边界
    if perception.blind_road_find:
        left_x = perception_blind['left_x'][0]
        right_x = perception_blind['right_x'][0]
        left_y = perception_blind['left_x'][1]
        right_y = perception_blind['right_x'][1]
        if perception_blind['on_blind_road']:
            if perception.blind_road_find:
                cv2.circle(result_vis,(left_x,left_y) , 50, (0, 255, 255), 8)
                cv2.circle(result_vis,(right_x,right_y), 50, (0, 255, 255), 8)
                dev_left=0.5*width-left_x
                dev_right=right_x-0.5*width
                if 0.5*width<left_x:   #图像中心不在盲道范围内
                    cv2.putText(result_vis,"left deviation",(10,200),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),2)
                if 0.5 * width > right_x:
                    cv2.putText(result_vis, "right deviation", (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
                if(0.5 * width<=right_x)and(0.5*width>=left_x):
                    if abs(dev_left)>abs(dev_right):    #取距离边界较近的
                        deviation.append(1)     #距离右边界更近的
                    else:
                        deviation.append(0)     #距离左边界更近
                if len(deviation)>=3:        #连续三帧都偏离
                    if deviation.count(0)==3:
                        cv2.putText(result_vis, "left deviation", (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
                        deviation.pop(0)
                    if deviation.count(0)==0:
                        cv2.putText(result_vis, "right deviation", (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
                        deviation.pop(0)
                    if (deviation.count(0)<3)and(deviation.count(0)>0):
                        cv2.putText(result_vis, "not deviation", (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
                        deviation.pop(0)
        else:  #不在盲道上在底部绘制一个方向小箭头
            # arrow_end
            arraw_slope=math.atan((height-10-blind_road_y)/float(blind_road_x-0.5*width))
            arrow_end_x=int(0.5*width+math.cos(arraw_slope))
            arrow_end_y=int(height-10-100*math.sin(arraw_slope))
            cv2.arrowedLine(result_vis, (int(0.5*width), int(height-10)), (arrow_end_x, arrow_end_y),
                            (255, 1, 1), 3, 0, 0, 0.2)

    #显示盲道信息



    #障碍物检测
    for i in range(len(rect_center)):
        obs=Obstacle()
        (x,y),rect=rect_center[i]
        obs.position=(int(x),int(y))
        obs.distance=depth[int(y),int(x)]
        obs.width=acceptedRects[i][2]
        d=depth[int(y),int(width/2)]
        L=2*d*np.tan(theta/2)
        l=L/width*(width/2-x)
        obs.angle=np.arctan(l/d)
        #画出障碍物,并标注distance与angle
        cv2.circle(result_vis,(int(x),int(y)),5,(0,0,255),-1)
        #画出矩形框
        cv2.rectangle(result_vis, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (121, 11, 189), 2)

        # #设置输出2位小数
        obs.distance=round(obs.distance,2)
        obs.angle=round(obs.angle,2)
        cv2.putText(result_vis,str(obs.distance)+'m',(int(x),int(y)),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),1)
        cv2.putText(result_vis,str(obs.angle)+"radian",(int(x),int(y)+20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),1)
        #显示宽度
        cv2.putText(result_vis,"width:"+str(0.5)+'m',(int(x),int(y)+40),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),1)
        perception.obs_list.append(obs)
    video_writer.write(result_vis)

    # cv2.imshow('result_vis',result_vis)
    # cv2.waitKey(0)
video_writer.release()











