import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox
from PyQt5.QtGui import QImage, QPixmap,QPainter
from PyQt5.QtCore import QObject, pyqtSignal,QUrl
from PyQt5.QtCore import pyqtSlot, Qt,QCoreApplication
from scipy.io import savemat
import image_predict_pyqt as img_process

import paddle_rgb
from face_mainwindow_n import Ui_MainWindow
import obs_detection as od
import os
import time
import datetime as dt
import numpy as np
import threading as th
import ctypes
import inspect

import re

# First import the library
import pyrealsense2 as rs
import cv2
import math
import paddle_rgb as paddle_r
#import yolo_detect as yolo
#from skimage import io
DELAY = 0

def makedirs(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
#点云
class AppState:

    def __init__(self, *args, **kwargs):
        self.WIN_NAME = 'RealSense'
        self.pitch, self.yaw = math.radians(-10), math.radians(-15)
        self.translation = np.array([0, 0, -1], dtype=np.float32)
        self.distance = 2
        self.prev_mouse = 0, 0
        self.mouse_btns = [False, False, False]
        self.paused = False
        self.decimate = 1
        self.scale = True
        self.color = True

    def reset(self):#重置
        self.pitch, self.yaw, self.distance = 0, 0, 2
        self.translation[:] = 0, 0, -1

    @property
    def rotation(self):#旋转
        Rx, _ = cv2.Rodrigues((self.pitch, 0, 0))
        Ry, _ = cv2.Rodrigues((0, self.yaw, 0))
        return np.dot(Ry, Rx).astype(np.float32)

    @property
    def pivot(self):#中心
        return self.translation + np.array((0, 0, self.distance), dtype=np.float32)

class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()

        # Set up the user interface from Designer.
        self.setupUi(self)

        self.dis_update.connect(self.camera_view)#这个信号对应更新gui上画面，dis_update信号传来qpixmap图像
        self.pushButton_takephotos.clicked.connect(self.pushButton_takephotos_clicked)
        # self.webEngineView.load(QUrl('file:///walking.html'))

        self.thread_camera = None
        self.takePhotos = False
        # self.paddle=paddle_rgb.paddle_rgb()
        self.state = AppState()

        # self.yo=yolo.yolo()
        #保存


    # 在对应的页面类的内部，与def定义的函数同级
    dis_update = pyqtSignal(QPixmap,QPixmap,QPixmap) #定义信号,定义参数为QPixmap类型

    def paintEvent(self, event):  # set background_img
        painter = QPainter(self)
        painter.drawRect(self.rect())
        pixmap = QPixmap('background.png')  # 换成自己的图片的相对路径
        painter.drawPixmap(self.rect(), pixmap)
#处理实时RGB图像
    # def process_RGBimages(self,rgb_image,trans):
    #     args=self.paddle.parse_args()
    #     fin_img=self.paddle.seg_infer_from_img(rgb_image,trans)#得到盲道与人行道的预测结果
    #     #对结果进行处理,得到盲道与人行道的走向
    #     res=self.get_abstract_data_from_img(fin_img)

        # return fin_img
    def get_abstract_data_from_img(self,img):
        pass

    def pushButton_takephotos_clicked(self):
        self.takePhotos = True

    # 添加一个退出的提示事件
    def closeEvent(self, event):
        """我们创建了一个消息框，上面有俩按钮：Yes和No.第一个字符串显示在消息框的标题栏，第二个字符串显示在对话框，
              第三个参数是消息框的俩按钮，最后一个参数是默认按钮，这个按钮是默认选中的。返回值在变量reply里。"""

        reply = QMessageBox.question(self, 'Message', "Are you sure to quit?",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        # 判断返回值，如果点击的是Yes按钮，我们就关闭组件和应用，否则就忽略关闭事件
        if reply == QMessageBox.Yes:
            self.stop_thread(self.thread_camera)
            event.accept()
        else:
            event.ignore()

    def open_camera(self):
        #寻找设备 #查找硬件 新建线程开启摄像头
        print("Searching Devices..")
        selected_devices = []  # Store connected device(s)
        for d in rs.context().devices:
            selected_devices.append(d)
            print(d.get_info(rs.camera_info.name))
            self.thread_camera = th.Thread(target=self.open_realsense)
            # self.thread_camera = th.Thread(target=self.record_data)
            self.thread_camera.start()
            print('Open Camera')
        if not selected_devices:
            print("No RealSense device is connected!")

        # target选择开启摄像头的函数


    def camera_view(self, deep,rgb,final):
        # 调用setPixmap函数设置显示Pixmap
        self.label_show.setPixmap(rgb)
        self.label_show_RGB.setPixmap(deep)
        self.label_show_final.setPixmap(final)
        # 调用setScaledContents将图像比例化显示在QLabel上
        self.label_show.setScaledContents(True)
        self.label_show_RGB.setScaledContents(True)
        self.label_show_final.setScaledContents(True)

    def _async_raise(self, tid, exctype):
        """raises the exception, performs cleanup if needed"""
        tid = ctypes.c_long(tid)
        if not inspect.isclass(exctype):
            exctype = type(exctype)
        res = ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, ctypes.py_object(exctype))
        if res == 0:
            raise ValueError("invalid thread id")
        elif res != 1:
            # """if it returns a number greater than one, you're in trouble,
            # and you should call it again with exc=NULL to revert the effect"""
            ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, None)
            raise SystemError("PyThreadState_SetAsyncExc failed")

    def stop_thread(self, thread):
        self._async_raise(thread.ident, SystemExit)

    def save_image(self,img_np,verts):
        #0:深度图 1:RGB图
        if (self.takePhotos == True):

            now_date = dt.datetime.now().strftime('%F')
            now_time = dt.datetime.now().strftime('%F_%H%M%S')

            path_ok = os.path.exists(now_date)
            if (path_ok == False):
                os.mkdir(now_date)

            if (os.path.isdir(now_date)):
                depth_image=cv2.cvtColor(img_np[0], cv2.COLOR_RGB2BGR)
                color_image=cv2.cvtColor(img_np[1], cv2.COLOR_RGB2BGR)

                depth_full_path = os.path.join('./', now_date, now_time + '_depth.png')
                color_full_path = os.path.join('./', now_date, now_time + '_color.png')
                cv2.imencode('.png', depth_image)[1].tofile(depth_full_path)
                cv2.imencode('.png', color_image)[1].tofile(color_full_path)

                np.savetxt('./'+now_date+"/"+now_time+"verts.txt", verts, delimiter=',')

                # print('ok')
            self.takePhotos = False
    def record_data(self):
        #存深度图,RGB图,分割后的RGB图,深度图np数组,点云np数组
        #############
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 1280, 720, rs.format.rgb8, 30)

        # Start streaming
        profile = pipeline.start(config)

        # Getting the depth sensor's depth scale (see rs-align example for explanation)
        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()
        depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
        depth_intrinsics = depth_profile.get_intrinsics()
        w, h = depth_intrinsics.width, depth_intrinsics.height
        # Processing blocks
        pc = rs.pointcloud()
        decimate = rs.decimation_filter()
        decimate.set_option(rs.option.filter_magnitude, 2 ** self.state.decimate)
        colorizer = rs.colorizer()
        clipping_distance_in_meters = 1  # 1 meter
        clipping_distance = clipping_distance_in_meters / depth_scale
        align_to = rs.stream.color
        align = rs.align(align_to)
#######################
        now_date = dt.datetime.now().strftime('%F')+"new"
        now_time = dt.datetime.now().strftime('%F_%H%M%S')
        path_ok = os.path.exists(now_date)
        if (path_ok == False):
            os.mkdir(now_date)
        video_path = os.path.join('./', now_date, now_time + '_proc.avi')
        videopath, videoname = os.path.split(video_path)
        disflow = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST)
        is_init = True
        #####################################存视频
        four_cc = cv2.VideoWriter_fourcc(*'XVID')
        video_writer = cv2.VideoWriter(video_path, four_cc, float(15), (w, h))
        times=0
        # Streaming loop
        try:
            while True:
                # Get frameset of color and depth
                frames = pipeline.wait_for_frames()
                # frames.get_depth_frame() is a 640x360 depth image

                # Align the depth frame to color frame
                aligned_frames = align.process(frames)

                # Get aligned frames
                aligned_depth_frame = aligned_frames.get_depth_frame()  # aligned_depth_frame is a 640x480 depth image
                color_frame = aligned_frames.get_color_frame()

                # Validate that both frames are valid
                if not aligned_depth_frame or not color_frame:
                    continue

                depth_image = np.asanyarray(aligned_depth_frame.get_data())
                color_image = np.asanyarray(color_frame.get_data())
                ##########实际距离
                depth_meter = depth_image.astype(float) * depth_scale
                np.save('./' + now_date + "/" + str(times) + "depth_image", depth_image)
                np.save('./' + now_date + "/" + str(times) + "depth_image_real", depth_meter)
                # Render images
                depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
                ###########################处理图像
                tempt1 = time.time()
                trans = 0.4
                processed_image=color_image
                # processed_image=img_process.pred_camera(color_image,depth_meter)
                # processed_image = self.process_RGBimages(color_image, trans)  # 处理rgb图像并显示
                tempt2 = time.time()
                ########################传递图像到界面显示
                qimage1 = QImage(depth_colormap, 1280, 720, QImage.Format_RGB888)
                pixmap1 = QPixmap.fromImage(qimage1)  # 深度图

                qimage2 = QImage(color_image, 1280, 720, QImage.Format_RGB888)
                pixmap2 = QPixmap.fromImage(qimage2)  # RGB图

                processed_image = processed_image.astype(np.uint8)
                qimage3 = QImage(processed_image, 1280, 720, QImage.Format_RGB888)
                pixmap3 = QPixmap.fromImage(qimage3)  # RGB图
                ########################保存数据

                depth_image = cv2.cvtColor(depth_colormap, cv2.COLOR_RGB2BGR)
                color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
                processed_image = cv2.cvtColor(processed_image, cv2.COLOR_RGB2BGR)
                video_writer.write(processed_image)

                depth_full_path = os.path.join('./', now_date, str(times) + '_depth.png')
                color_full_path = os.path.join('./', now_date, str(times) + '_color.png')
                processed_full_path = os.path.join('./', now_date, str(times) + '_processed.png')

                cv2.imencode('.png', depth_image)[1].tofile(depth_full_path)
                cv2.imencode('.png', color_image)[1].tofile(color_full_path)
                cv2.imencode('.png', processed_image)[1].tofile(processed_full_path)

                times+=1
                ########更新
                print(times)
                self.dis_update.emit(pixmap1, pixmap2, pixmap3)  # 发射信号，启用槽函数,更新到界面上
                time.sleep(DELAY)
        finally:
            pipeline.stop()
            video_writer.release()

    def open_realsense(self):
        print('open_realsense')
        pipeline = rs.pipeline()

        config = rs.config()
        config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 1280, 720, rs.format.rgb8, 30)

        # Start streaming
        profile = pipeline.start(config)

        # Getting the depth sensor's depth scale (see rs-align example for explanation)
        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()
        depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
        depth_intrinsics = depth_profile.get_intrinsics()
        w, h = depth_intrinsics.width, depth_intrinsics.height
        # Processing blocks
        pc = rs.pointcloud()
        decimate = rs.decimation_filter()
        decimate.set_option(rs.option.filter_magnitude, 2 ** self.state.decimate)
        colorizer = rs.colorizer()

        clipping_distance_in_meters = 1  # 1 meter
        clipping_distance = clipping_distance_in_meters / depth_scale

        align_to = rs.stream.color
        align = rs.align(align_to)

        # Streaming loop
        try:
            while True:
                frames = pipeline.wait_for_frames()
                aligned_frames = align.process(frames)
                # Get aligned frames
                aligned_depth_frame = aligned_frames.get_depth_frame()  # aligned_depth_frame is a 640x480 depth image
                color_frame = aligned_frames.get_color_frame()
                # Validate that both frames are valid
                if not aligned_depth_frame or not color_frame:
                    continue

                depth_image = np.asanyarray(aligned_depth_frame.get_data())
                color_image = np.asanyarray(color_frame.get_data())
                ##########实际距离
                depth_meter = depth_image.astype(float) * depth_scale
                # Render images
                depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
              ###########################处理图像
                # np.savetxt("depth_image.txt", depth_image, fmt='%d', delimiter=',')
                # np.savetxt("depth_image_real.txt", depth_meter, fmt='%d', delimiter=',')
                np.save( "depth_image", depth_image)
                np.save("depth_image_real", depth_meter)

                tempt1 = time.time()
                trans = 0.5
                processed_image=self.process_RGBimages(color_image,trans)#处理rgb图像并显示
                # image5 = od.obs_detection("188depth_image_real.npy")  # 障碍物检测代码获得的图片

                image5 = od.obs_detection("depth_image_real.npy")  # 障碍物检测代码获得的图片
                processed_image=image5

                # processed_image = self.Generate_convex_hull(color_image)
                tempt2 = time.time()


                # images = np.hstack((bg_removed[0:720, 320:960], depth_colormap[0:720, 320:960]))#在水平方向上平铺拼接数组
   ########################保存数据
                print(tempt2 - tempt1)
                ###RGB转BGR
                # color_image=cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
                # processed_image=cv2.cvtColor(processed_image, cv2.COLOR_RGB2BGR)
                cv2.imwrite("processed_image.png", processed_image)
                cv2.imwrite("color_image.png", color_image)
                # np.savetxt("processed_image.txt", processed_image)
                # np.savetxt("color_image.txt", color_image, fmt='%d', delimiter=',')
                # print(np.array_equal(processed_image,color_image))
                # 保存图片
                # self.save_image((depth_colormap, color_image),verts)
    ########################传递图像到界面显示
                qimage1 = QImage(depth_colormap, 1280, 720, QImage.Format_RGB888)
                pixmap1 = QPixmap.fromImage(qimage1)#深度图

                qimage2 = QImage(color_image, 1280, 720, QImage.Format_RGB888)
                pixmap2 = QPixmap.fromImage(qimage2)#RGB图

                processed_image=processed_image.astype(np.uint8)
                qimage3 = QImage(processed_image, 1280, 720, QImage.Format_RGB888)
                pixmap3 = QPixmap.fromImage(qimage3)#RGB图

                self.dis_update.emit(pixmap1,pixmap2,pixmap3)#发射信号，启用槽函数,更新到界面上
                time.sleep(DELAY)
        finally:
            pipeline.stop()
    def open_video_thread(self):
        self.thread_video = th.Thread(target=self.open_video)
        self.thread_video.start()
    def record_data_thread(self):
        self.thread_video = th.Thread(target=self.record_data)
        self.thread_video.start()
    def open_video(self):
        videoname="./test_video/VID_20211030_174637.mp4"
        cap = cv2.VideoCapture(videoname)
        videopath, videoname = os.path.split(videoname)
        videoname=videoname[:-4]
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        makedirs(videoname)
        disflow = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST)
        is_init = True
        fps = cap.get(cv2.CAP_PROP_FPS)
#####################################存视频
        save_path = "./" + videoname + "/" + videoname + "processed.avi"
        four_cc = cv2.VideoWriter_fourcc(*'XVID')
        video_writer = cv2.VideoWriter(save_path, four_cc, float(fps), (width,height))

        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                im_shape = frame.shape
                #借鉴paddlx的example
                image = frame.astype(np.uint8)
                qimage1 = QImage(image, 1280, 720, QImage.Format_RGB888)
                pixmap1 = QPixmap.fromImage(qimage1)  # 深度图

                qimage2 = QImage(image, 1280, 720, QImage.Format_RGB888)
                pixmap2 = QPixmap.fromImage(qimage2)  # RGB图
                tempt1 = time.time()
                trans=0.5
                processed_image = self.process_RGBimages(image,trans)  # 处理rgb图像并显示
                tempt2 = time.time()
                print(tempt2 - tempt1)
                processed_image = processed_image.astype(np.uint8)
                cv2.imwrite("./"+videoname+"/"+str(int(time.time() * 1000))+"processed_image.png", processed_image)
                video_writer.write(processed_image)
                qimage3 = QImage(processed_image, 1280, 720, QImage.Format_RGB888)
                pixmap3 = QPixmap.fromImage(qimage3)  # RGB图
                self.dis_update.emit(pixmap1, pixmap2, pixmap3)  # 发射信号，启用槽函数,更新到界面上
                time.sleep(DELAY)
            else:
                break
        cap.release()
        video_writer.release()



if __name__ == "__main__":
    QCoreApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    # w.open_camera()
    # w.open_video_thread()
    w.record_data_thread()
    # thread_camera = th.Thread(target=w.open_realsense)
    # thread_camera.start()
    sys.exit(app.exec_())