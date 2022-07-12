
import os
import time
import datetime as dt
import numpy as np
import threading as th
import ctypes
import inspect
import cv2
import io

FULL_KERNEL_3 = np.ones((3, 3), np.uint8)
FULL_KERNEL_5 = np.ones((5, 5), np.uint8)
FULL_KERNEL_7 = np.ones((7, 7), np.uint8)
FULL_KERNEL_9 = np.ones((9, 9), np.uint8)
FULL_KERNEL_31 = np.ones((31, 31), np.uint8)

# 3x3 cross kernel
CROSS_KERNEL_3 = np.asarray(
    [
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0],
    ], dtype=np.uint8)

# 5x5 cross kernel
CROSS_KERNEL_5 = np.asarray(
    [
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [1, 1, 1, 1, 1],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
    ], dtype=np.uint8)

# 5x5 diamond kernel
DIAMOND_KERNEL_5 = np.array(
    [
        [0, 0, 1, 0, 0],
        [0, 1, 1, 1, 0],
        [1, 1, 1, 1, 1],
        [0, 1, 1, 1, 0],
        [0, 0, 1, 0, 0],
    ], dtype=np.uint8)

# 7x7 cross kernel
CROSS_KERNEL_7 = np.asarray(
    [
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
    ], dtype=np.uint8)

# 7x7 diamond kernel
DIAMOND_KERNEL_7 = np.asarray(
    [
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 1, 1, 1, 0, 0],
        [0, 1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1, 1],
        [0, 1, 1, 1, 1, 1, 0],
        [0, 0, 1, 1, 1, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
    ], dtype=np.uint8)

TMP_PREFIX = '/'


def depthComplete(depth_map):  # 深度补全
    # filename = os.path.join(TMP_PREFIX, filename)
    # depth_map = parse_distance(filename) # np array of depths
    ## step 1
    valid_pixels = (depth_map > 0.1)
    depth_map[valid_pixels] = 100 - depth_map[valid_pixels]

    ## step 2
    depth_map = cv2.dilate(depth_map, DIAMOND_KERNEL_5)

    ## step 3
    depth_map = cv2.morphologyEx(depth_map, cv2.MORPH_CLOSE, FULL_KERNEL_5)

    ## step 4
    empty_pixels = (depth_map < 0.1)
    dilated = cv2.dilate(depth_map, FULL_KERNEL_7)
    depth_map[empty_pixels] = dilated[empty_pixels]

    # step 5
    top_row_pixels = np.argmax(depth_map > 0.1, axis=0)
    top_pixel_values = depth_map[top_row_pixels, range(depth_map.shape[1])]

    for pixel_col_idx in range(depth_map.shape[1]):
        depth_map[0:top_row_pixels[pixel_col_idx], pixel_col_idx] = \
            top_pixel_values[pixel_col_idx]

    # Large Fill
    empty_pixels = depth_map < 0.1
    dilated = cv2.dilate(depth_map, FULL_KERNEL_31)
    depth_map[empty_pixels] = dilated[empty_pixels]

    image_color_old = cv2.applyColorMap(
        np.uint8(depth_map / np.amax(depth_map) * 255),
        cv2.COLORMAP_RAINBOW)

    ## step 7
    valid_pixels = (depth_map > 0.1)
    depth_map[valid_pixels] = 100 - depth_map[valid_pixels]

    depth_map[depth_map > 10] = 10

    image_color = cv2.applyColorMap(
        np.uint8(depth_map / np.amax(depth_map) * 255),
        cv2.COLORMAP_RAINBOW)
    # cv2.imwrite(filename + ".png", image_color)
    # retval, img_str = cv2.imencode('.png', image_color)

    # img = BytesIO(img_str)
    # img.seek(0)
    return image_color

def depthComplete_file(filename):  # 深度补全
    # filename = os.path.join(TMP_PREFIX, filename)
    # depth_map = parse_distance(filename) # np array of depths
    depth_map = np.load(filename)  #
    ## step 1
    valid_pixels = (depth_map > 0.1)
    depth_map[valid_pixels] = 100 - depth_map[valid_pixels]

    ## step 2
    depth_map = cv2.dilate(depth_map, DIAMOND_KERNEL_5)

    ## step 3
    depth_map = cv2.morphologyEx(depth_map, cv2.MORPH_CLOSE, FULL_KERNEL_5)

    ## step 4
    empty_pixels = (depth_map < 0.1)
    dilated = cv2.dilate(depth_map, FULL_KERNEL_7)
    depth_map[empty_pixels] = dilated[empty_pixels]

    # step 5
    top_row_pixels = np.argmax(depth_map > 0.1, axis=0)
    top_pixel_values = depth_map[top_row_pixels, range(depth_map.shape[1])]

    for pixel_col_idx in range(depth_map.shape[1]):
        depth_map[0:top_row_pixels[pixel_col_idx], pixel_col_idx] = \
            top_pixel_values[pixel_col_idx]

    # Large Fill
    empty_pixels = depth_map < 0.1
    dilated = cv2.dilate(depth_map, FULL_KERNEL_31)
    depth_map[empty_pixels] = dilated[empty_pixels]

    image_color_old = cv2.applyColorMap(
        np.uint8(depth_map / np.amax(depth_map) * 255),
        cv2.COLORMAP_RAINBOW)

    ## step 7
    valid_pixels = (depth_map > 0.1)
    depth_map[valid_pixels] = 100 - depth_map[valid_pixels]

    depth_map[depth_map > 10] = 10

    image_color = cv2.applyColorMap(
        np.uint8(depth_map / np.amax(depth_map) * 255),
        cv2.COLORMAP_RAINBOW)
    cv2.imwrite(filename + ".png", image_color)
    retval, img_str = cv2.imencode('.png', image_color)

    # img = BytesIO(img_str)
    # img.seek(0)
    return image_color

def for_draw_file(filename):
    img1 = depthComplete_file(filename)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    # cv2.imshow("深度补全", img1)
    # cv2.waitKey()
    return img1

def for_draw(img1):
    img1 = depthComplete(img1)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    # cv2.imshow("深度补全", img1)
    # cv2.waitKey()
    return img1
# 处理颜色图像
def process_color_img(color_img):
    gray = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 3)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    # cv2.imshow('thresh', thresh)
    # cv2.waitKey()
    blurred = cv2.GaussianBlur(thresh, (3, 3), 0)
    dst = cv2.Canny(thresh, 90, 125)
    # cv2.imshow("img_result", dst)
    # cv2.waitKey()

    return thresh

def merge_box(obstacle):
    rects = obstacle
    rects.sort(key=lambda r: r[0])

    # Array of accepted rects
    acceptedRects = []

    # Merge threshold for x coordinate distance
    xThr = 200
    rectsUsed = np.array([False] * len(rects))
    # Iterate all initial bounding rects
    for supIdx, supVal in enumerate(rects):
        if rectsUsed[supIdx] == False:

            # Initialize current rect
            currxMin = supVal[0]
            currxMax = supVal[0] + supVal[2]
            curryMin = supVal[1]
            curryMax = supVal[1] + supVal[3]

            # This bounding rect is used
            rectsUsed[supIdx] = True

            # Iterate all initial bounding rects
            # starting from the next
            for subIdx, subVal in enumerate(rects[(supIdx + 1):], start=(supIdx + 1)):

                # Initialize merge candidate
                candxMin = subVal[0]
                candxMax = subVal[0] + subVal[2]
                candyMin = subVal[1]
                candyMax = subVal[1] + subVal[3]

                # Check if x distance between current rect
                # and merge candidate is small enough
                if (candxMin <= currxMax + xThr):

                    # Reset coordinates of current rect
                    currxMax = candxMax
                    curryMin = min(curryMin, candyMin)
                    curryMax = max(curryMax, candyMax)

                    # Merge candidate (bounding rect) is used
                    rectsUsed[subIdx] = True
                else:
                    break

            # No more merge candidates possible, accept current rect
            acceptedRects.append([currxMin, curryMin, currxMax - currxMin, curryMax - curryMin])
    return acceptedRects

def process_depth_img(depth_img, depthimg):
    blurred = cv2.GaussianBlur(depth_img, (3, 3), 0)
    dst = cv2.Canny(depth_img, 90, 125)
    # cv2.imshow("img_result", dst)
    cv2.waitKey()
    max_depth=3
    min_depth=0.1
    # 3.连通域分析
    contours, hierarchy = cv2.findContours(dst,
                                           cv2.RETR_LIST,
                                           cv2.CHAIN_APPROX_SIMPLE)
    img = cv2.drawContours(depth_img, contours, -1, (0, 255, 0), 3)

    bw_height = 20
    bw_width = 20
    obstacle = []

    for i in contours:
        bR = bx, by, bw, bh = cv2.boundingRect(i)

        rec = cv2.minAreaRect(i)
        box = cv2.boxPoints(rec)  # 获取最小外接矩形的4个顶点坐标(ps: cv2.boxPoints(rect) for OpenCV 3.x)
        box = np.int0(box)
        b_center_x = int((box[0][0] + box[2][0]) / 2)
        b_center_y = int((box[0][1] + box[2][1]) / 2)
        # box_center=((box[0][0]+box[2][0])/2,(box[0][1]+box[2][1])/2)
        if (bw > bw_height or bh > bw_width ):

            # 左边黄圈,右边蓝圈  #0与1有时是左右相邻,有时是上下相邻
            if (depthimg[b_center_y][b_center_x] <=min_depth or depthimg[b_center_y][b_center_x] >= max_depth):  # 距离过远或没有检测数据
                # cv2.circle(img, (b_center_x, b_center_y), 10, (0, 0, 255), 2)
                continue
            if (b_center_y < 200):  # 如果处于图像上方,可能是天空识别错误
                continue
            img = cv2.drawContours(img, [box], 0, (255, 0, 0), )
            obstacle.append(bR)  # i是轮廓,box是矩形框

    acceptedRects = merge_box(obstacle)
    obstacle=[]
    for rect in acceptedRects:
        img = cv2.rectangle(img, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (121, 11, 189), 2)
        #将矩形质心加入
        obstacle.append(( (rect[0] + rect[2]/2, rect[1] + rect[3]/2),rect))
    # cv2.imshow("img_result", img)
    # cv2.waitKey()
    return img, obstacle,acceptedRects

def obs_detection_file(filename):
    depthimge = np.load(filename)
    image1 = for_draw_file(filename)
    img1, obstacle1,acceptedRects = process_depth_img(image1, depthimge)  # process depth image
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    return (obstacle1,acceptedRects)
    # cv2.imshow("障碍物检测", img1)

def obs_detection(depth_img):
    image1 = for_draw(depth_img)
    img1, obstacle1,acceptedRects = process_depth_img(image1, depth_img)  # process depth image
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    return (obstacle1,acceptedRects)