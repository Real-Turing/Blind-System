# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'face_mainwindow_n.ui'
#
# Created by: PyQt5 UI code generator 5.15.5
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1026, 685)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.Prompt = QtWidgets.QTextEdit(self.centralwidget)
        self.Prompt.setEnabled(True)
        self.Prompt.setGeometry(QtCore.QRect(40, 460, 261, 121))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.Prompt.sizePolicy().hasHeightForWidth())
        self.Prompt.setSizePolicy(sizePolicy)
        self.Prompt.setMaximumSize(QtCore.QSize(900, 400))
        self.Prompt.setStyleSheet("border-width: 4px;border-style: solid;border-color: rgb(255, 170, 0)")
        self.Prompt.setReadOnly(True)
        self.Prompt.setObjectName("Prompt")
        self.splitter = QtWidgets.QSplitter(self.centralwidget)
        self.splitter.setGeometry(QtCore.QRect(40, 70, 301, 381))
        self.splitter.setOrientation(QtCore.Qt.Vertical)
        self.splitter.setObjectName("splitter")
        self.label_show_RGB = QtWidgets.QLabel(self.splitter)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_show_RGB.sizePolicy().hasHeightForWidth())
        self.label_show_RGB.setSizePolicy(sizePolicy)
        self.label_show_RGB.setStyleSheet("border-width: 4px;border-style: solid;border-color: rgb(255, 170, 0)")
        self.label_show_RGB.setFrameShape(QtWidgets.QFrame.Box)
        self.label_show_RGB.setFrameShadow(QtWidgets.QFrame.Raised)
        self.label_show_RGB.setObjectName("label_show_RGB")
        self.label_show = QtWidgets.QLabel(self.splitter)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_show.sizePolicy().hasHeightForWidth())
        self.label_show.setSizePolicy(sizePolicy)
        self.label_show.setStyleSheet("border-width: 4px;border-style: solid;border-color: rgb(255, 170, 0)")
        self.label_show.setFrameShape(QtWidgets.QFrame.Box)
        self.label_show.setObjectName("label_show")
        self.label_show_final = QtWidgets.QLabel(self.centralwidget)
        self.label_show_final.setGeometry(QtCore.QRect(360, 80, 651, 431))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_show_final.sizePolicy().hasHeightForWidth())
        self.label_show_final.setSizePolicy(sizePolicy)
        self.label_show_final.setStyleSheet("border-width: 4px;border-style: solid;border-color: rgb(255, 170, 0)")
        self.label_show_final.setFrameShape(QtWidgets.QFrame.Box)
        self.label_show_final.setObjectName("label_show_final")
        self.layoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.layoutWidget.setGeometry(QtCore.QRect(300, 10, 238, 25))
        self.layoutWidget.setObjectName("layoutWidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.layoutWidget)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.pushButton_takephotos = QtWidgets.QPushButton(self.layoutWidget)
        self.pushButton_takephotos.setObjectName("pushButton_takephotos")
        self.horizontalLayout.addWidget(self.pushButton_takephotos)
        self.Prompt.raise_()
        self.label_show_final.raise_()
        self.layoutWidget.raise_()
        self.splitter.raise_()
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1026, 22))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Listen Light"))
        self.Prompt.setDocumentTitle(_translate("MainWindow", "??????????????????"))
        self.label_show_RGB.setText(_translate("MainWindow", "RGB"))
        self.label_show.setText(_translate("MainWindow", "Depth"))
        self.label_show_final.setText(_translate("MainWindow", "RGB"))
        self.pushButton_takephotos.setText(_translate("MainWindow", "??????"))
