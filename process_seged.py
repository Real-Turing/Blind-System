
import random

import numpy as np
import cv2
import math
import scipy.misc
import PIL.Image
import statistics
import timeit
import glob
from sklearn import linear_model, datasets


# get a line from a point and unit vectors
def lineCalc(vx, vy, x0, y0):
    scale = 10
    x1 = x0 + scale * vx
    y1 = y0 + scale * vy
    m = (y1 - y0) / (x1 - x0)
    b = y1 - m * x1
    return m, b


# the angle at the vanishing point
def angle(pt1, pt2):
    x1, y1 = pt1
    x2, y2 = pt2
    inner_product = x1 * x2 + y1 * y2
    len1 = math.hypot(x1, y1)
    len2 = math.hypot(x2, y2)
    print(len1)
    print(len2)
    a = math.acos(inner_product / (len1 * len2))
    return a * 180 / math.pi


# vanishing point - cramer's rule
def lineIntersect(m1, b1, m2, b2):
    # a1*x+b1*y=c1
    # a2*x+b2*y=c2
    # convert to cramer's system
    a_1 = -m1
    b_1 = 1
    c_1 = b1

    a_2 = -m2
    b_2 = 1
    c_2 = b2

    d = a_1 * b_2 - a_2 * b_1  # determinant
    dx = c_1 * b_2 - c_2 * b_1
    dy = a_1 * c_2 - a_2 * c_1

    intersectionX = dx / d
    intersectionY = dy / d
    return intersectionX, intersectionY


def process_sideways(im,depth,W,H):
    return_res={}
    start = timeit.timeit()  # start timer

    x = W
    y = H
    ratio = H / W
    # W = 800
    # H = int(W * ratio)
    radius = 250  # px
    thresh = 170
    # bw_width = 170
    bw_width = 150
    bw_height = 150

    bxLeft = []
    byLeft = []
    bxbyLeftArray = []
    bxbyRightArray = []
    bxRight = []
    byRight = []
    boundedLeft = []
    boundedRight = []

    # 1. filter the white color
    lower = np.array([0, 127, 0])#GBR
    upper = np.array([255, 255, 255])
    # im的RGb是小数
    mask = cv2.inRange(im, lower, upper)#返回一个二值图像

    # 2. erode the frame
    erodeSize = int(y / 30)
    erodeStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (erodeSize, 1))
    erode = cv2.erode(mask, erodeStructure, (-1, -1))

    contours, hierarchy = cv2.findContours(erode, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    im_temp=cv2.drawContours(im, contours, -1, (0, 255, 0), 3)
    if_get_contours_point = False
    if len(contours)> 0:
        for i in contours:
            bx, by, bw, bh = cv2.boundingRect(i)
            rect =cv2.minAreaRect(i)
            box = cv2.boxPoints(rect)  # 获取最小外接矩形的4个顶点坐标(ps: cv2.boxPoints(rect) for OpenCV 3.x)
            box = np.int0(box)


            if (bw > bw_height or bh > bw_width):
                cv2.drawContours(im, [box], 0, (255, 0, 0), )
                bxRight.append(box[2][0])  # right line
                byRight.append(box[2][1])  # right line
                bxRight.append(box[3][0])  # right line
                byRight.append(box[3][1])  # right line

                bxLeft.append(box[0][0])  # left line
                byLeft.append(box[0][1])  # left line
                bxLeft.append(box[1][0])  # left line
                byLeft.append(box[1][1])  # left line

                bxbyLeftArray.append(box[0])  # x,y for the left line
                bxbyLeftArray.append(box[3])  # x,y for the left line

                bxbyRightArray.append(box[2])  # x,y for the left line
                bxbyRightArray.append(box[1])  # x,y for the left line

                # cv2.circle(im, box[0], 5, (0, 250, 250), 2)  # circles -> left line
                # cv2.circle(im, box[2], 5, (250, 250, 0), 2)  # circles -> right line
                if_get_contours_point=1


    #是否发现人行道
    if if_get_contours_point==1:
        return_res["sideways_find"]=1
    else:
        return_res["sideways_find"]=0
        return return_res
    try:
        # calculate median average for each line
        medianR = np.median(bxbyRightArray, axis=0)
        medianL = np.median(bxbyLeftArray, axis=0)

        bxbyLeftArray = np.asarray(bxbyLeftArray)
        bxbyRightArray = np.asarray(bxbyRightArray)

        # 4. are the points bounded within the median circle?
        for i in bxbyLeftArray:
            if (((medianL[0] - i[0]) ** 2 + (medianL[1] - i[1]) ** 2) < radius ** 2) == True:
                boundedLeft.append(i)

        boundedLeft = np.asarray(boundedLeft)

        for i in bxbyRightArray:
            if (((medianR[0] - i[0]) ** 2 + (medianR[1] - i[1]) ** 2) < radius ** 2) == True:
                boundedRight.append(i)

        boundedRight = np.asarray(boundedRight)

        # 5. RANSAC Algorithm
        if len(boundedLeft)<2 or len(boundedRight)<2:
            return_res["sideways_find"] = 0
            return return_res
        # select the points enclosed within the circle (from the last part)
        bxLeft = np.asarray(boundedLeft[:, 0])
        byLeft = np.asarray(boundedLeft[:, 1])
        bxRight = np.asarray(boundedRight[:, 0])
        byRight = np.asarray(boundedRight[:, 1])

        # transpose x of the right and the left line
        bxLeftT = np.array([bxLeft]).transpose()
        bxRightT = np.array([bxRight]).transpose()
        bxLeftT=np.unique(bxLeftT,axis=0)
        bxRightT=np.unique(bxRightT,axis=0)
        if len(bxLeftT)<2 or len(bxRightT)<2 or len(bxLeftT)!=len(byLeft) or len(bxRightT)!=len(byRight):
            return_res["sideways_find"]=0
            return return_res
        # run ransac for LEFT
        model_ransac = linear_model.RANSACRegressor(linear_model.LinearRegression())
        ransacX = model_ransac.fit(bxLeftT, byLeft)
        inlier_maskL = model_ransac.inlier_mask_  # right mask

        # run ransac for RIGHT
        ransacY = model_ransac.fit(bxRightT, byRight)
        inlier_maskR = model_ransac.inlier_mask_  # left mask
    ########左边线
        #边界控制
        L_border_dots= boundedLeft[inlier_maskL]
        L_border_dots[:, 1]=np.where(L_border_dots[:, 1]<0,5,L_border_dots[:, 1])
        L_border_dots[:, 1]=np.where(L_border_dots[:, 1]>=H,H-5,L_border_dots[:, 1])
        L_border_dots[:, 0]=np.where(L_border_dots[:, 0]<0,5,L_border_dots[:, 0])
        L_border_dots[:, 0]=np.where(L_border_dots[:, 0]>=W,W-5,L_border_dots[:, 0])

        L_border_dots_meter=depth[L_border_dots[:,1],L_border_dots[:,0]]
        #L_border_dots添加列数据
        L_border_dots=np.hstack((L_border_dots,L_border_dots_meter.reshape(L_border_dots.shape[0],1)))
        #按照第三列排序
        L_border_dots=L_border_dots[L_border_dots[:,2].argsort()]
        #取得排序后的第一行,即最近的点
        L_border_dots_min=L_border_dots[0]
        #取得排序后的最后一行,即最远的点
        L_border_dots_max=L_border_dots[-1]
    ######右边线
        R_border_dots= boundedRight[inlier_maskR]
        # #边界控制
        R_border_dots[:, 1]=np.where(R_border_dots[:, 1]<0,5,R_border_dots[:, 1])
        R_border_dots[:, 1]=np.where(R_border_dots[:, 1]>=H,H-5,R_border_dots[:, 1])
        R_border_dots[:, 0]=np.where(R_border_dots[:, 0]<0,5,R_border_dots[:, 0])
        R_border_dots[:, 0]=np.where(R_border_dots[:, 0]>=W,W-5,R_border_dots[:, 0])

        R_border_dots_meter=depth[R_border_dots[:,1],R_border_dots[:,0]]
        #R_border_dots添加列数据
        R_border_dots=np.hstack((R_border_dots,R_border_dots_meter.reshape(R_border_dots.shape[0],1)))
        #按照第三列排序
        R_border_dots=R_border_dots[R_border_dots[:,2].argsort()]
        #取得排序后的第一行,即最近的点
        R_border_dots_min=R_border_dots[0]
        #取得排序后的最后一行,即最远的点
        R_border_dots_max=R_border_dots[-1]


        return_res["sideways_distance"]=L_border_dots_min[2]
        return_res["sideways_position"]=(int(L_border_dots_min[0]+R_border_dots_min[0])/2,int(L_border_dots_min[1]+R_border_dots_min[1])/2)
        return_res["sideways_x"] =int(L_border_dots_min[0]+R_border_dots_min[0])/2
        return_res["sideways_y"] =int(L_border_dots_min[1]+R_border_dots_min[1])/2
        # return_res["sideways_min_lx"] =L_border_dots_min[0]
        # return_res["sideways_min_ly"] =L_border_dots_min[1]
        # return_res["sideways_min_rx"] =R_border_dots_min[0]
        # return_res["sideways_min_ry"] =R_border_dots_min[1]
        # return_res["sideways_max_lx"] =L_border_dots_max[0]
        # return_res["sideways_max_ly"] =L_border_dots_max[1]
        # return_res["sideways_max_rx"] =R_border_dots_max[0]
        # return_res["sideways_max_ry"] =R_border_dots_max[1]
        return_res["sideways_min_l"]=(int(L_border_dots_min[0]),int(L_border_dots_min[1]))
        return_res["sideways_min_r"]=(int(R_border_dots_min[0]),int(R_border_dots_min[1]))
        return_res["sideways_max_l"]=(int(L_border_dots_max[0]),int(L_border_dots_max[1]))
        return_res["sideways_max_r"]=(int(R_border_dots_max[0]),int(R_border_dots_max[1]))



        for i, element in enumerate(boundedRight[inlier_maskR]):
            # print(i,element[0])
            cv2.circle(im, (element[0], element[1]), 10, (250, 250, 250), 2)  # circles -> right line

        for i, element in enumerate(boundedLeft[inlier_maskL]):
            # print(i,element[0])
            cv2.circle(im, (element[0], element[1]), 10, (100, 100, 250), 2)  # circles -> Left line
        # cv2.imshow("im", im)
        # cv2.waitKey(0)
        # # 6. Calcuate the intersection point of the bounding lines
        # unit vector + a point on each line
        vx, vy, x0, y0 = cv2.fitLine(boundedLeft[inlier_maskL], cv2.DIST_L2, 0, 0.01, 0.01)
        vx_R, vy_R, x0_R, y0_R = cv2.fitLine(boundedRight[inlier_maskR], cv2.DIST_L2, 0, 0.01, 0.01)

        # get m*x+b
        m_L, b_L = lineCalc(vx, vy, x0, y0)
        m_R, b_R = lineCalc(vx_R, vy_R, x0_R, y0_R)
        return_res["sideways_L_line_m"]=m_L
        return_res["sideways_L_line_b"]=b_L
        return_res["sideways_R_line_m"]=m_R
        return_res["sideways_R_line_b"]=b_R
        return_res['sideways_move_towards']=(m_L+m_R)/2

        # 计算交点
        intersectionX, intersectionY = lineIntersect(m_R, b_R, m_L, b_L)
        return_res["intersection"]=(intersectionX,intersectionY)

        # 7. 画出边界线和交点
        m = radius*2
        # if (intersectionY < H / 2):
        if (intersectionY < H /2 and intersectionX<W/2 and intersectionY>0 and intersectionX>0):
            cv2.circle(im, (int(intersectionX), int(intersectionY)), 10, (0, 0, 255),15)  # 用法： cv2.circle(image, center_coordinates, radius, color, thickness)
            cv2.line(im,(int(x0-m*vx),int(y0-m*vy)),(int(x0+m*vx),int(y0+m*vy)),(255,0,0),3)
            cv2.line(im,(int(x0_R-m*vx_R),int(y0_R-m*vy_R)),(int(x0_R+m*vx_R),int(y0_R+m*vy_R)),(255,0,0),3)
    except:
        return_res["sideways_find"] = 0
        return return_res


    # cv2.imshow("im", im)
    # cv2.waitKey(0)


    end = timeit.timeit()  # STOP TIMER
    time_ = end - start

    print(str(time_) + " seconds")
    return return_res

def process_blind_road(im,depth,W,H):
    return_res = {}
    start = timeit.timeit()  # start timer

    x = W
    y = H
    ratio = H / W
    # W = 800
    # H = int(W * ratio)
    radius = 250  # px
    thresh = 170
    # bw_width = 170
    bw_width = 100
    bw_height = 100

    bxLeft = []
    byLeft = []
    bxbyLeftArray = []
    bxbyRightArray = []
    bxRight = []
    byRight = []
    boundedLeft = []
    boundedRight = []

    lower = np.array([1, 0, 0])#GBR
    upper = np.array([255, 255, 255])
    mask = cv2.inRange(im, lower, upper)#返回一个二值图像

    # 2. erode the frame
    erodeSize = int(y / 30)
    erodeStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (erodeSize, 1))
    erode = cv2.erode(mask, erodeStructure, (-1, -1))

    contours, hierarchy = cv2.findContours(erode, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(im, contours, -1, (0, 255, 0), 3)
    if_get_contours_point = False
    Box_list = []
    blind_road_list=[]
    if len(contours)> 0:
        for i in contours:
            bx, by, bw, bh = cv2.boundingRect(i)
            rect = cv2.minAreaRect(i)
            box = cv2.boxPoints(rect)  # 获取最小外接矩形的4个顶点坐标(ps: cv2.boxPoints(rect) for OpenCV 3.x)
            box = np.int0(box)
            # print(box)

            if (bw > bw_height or bh > bw_width):
                cv2.drawContours(im, [box], 0, (255, 0, 0), )
                # cv2.circle(im, box[0], 5, (0, 250, 250), 2)  # circles -> left line
                # cv2.circle(im, box[2], 5, (250, 250, 0), 2)  # circles -> right line
                if_get_contours_point = 1
                Box_list.append(box)
                i.reshape(-1, 2)
                blind_road_list.append(i)
        # 是否发现盲道
    if if_get_contours_point == 1:
        return_res["blind_road_find"] = 1
    else:
        return_res["blind_road_find"] = 0
        return return_res

    blind_road_list=np.array(blind_road_list)
    #给blind_road_list按照元素个数排序
    blind_road_list=blind_road_list[np.argsort(np.array(blind_road_list).shape[0])]
    blind_road=blind_road_list[0]
    blind_road=blind_road.reshape(blind_road.shape[0],2)


    #给blind_road按照y排序
    blind_road_sorted=blind_road[np.argsort(blind_road[:,1])]

    #取blind_road_sorted的最后20个点求平均
    blind_road_sorted_last20=blind_road_sorted[-20:]
    blind_road_sorted_last20_mean=np.mean(blind_road_sorted_last20,axis=0)
    if blind_road_sorted_last20_mean[1]>H-10:
        return_res["on_blind_road"]=True
    else:
        return_res["on_blind_road"]=False
    return_res["blind_road_position"] = (int(blind_road_sorted_last20_mean[0]),int(blind_road_sorted_last20_mean[1]))
    return_res["blind_road_distance"] =depth[int(blind_road_sorted_last20_mean[1]),int(blind_road_sorted_last20_mean[0])]

    #按照x排序
    blind_road_sorted_x=blind_road[np.argsort(blind_road[:,0])]
    #取左右边界
    return_res['left_x']=blind_road_sorted_x[0]
    return_res['right_x']=blind_road_sorted_x[-1]


    end = timeit.timeit()  # STOP TIMER
    time_ = end - start

    print(str(time_) + " seconds")
    return return_res

