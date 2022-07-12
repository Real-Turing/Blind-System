import os
import shutil
images_path='./2022-04-28new'
i=0
for image_name in os.listdir(images_path):
    i+=1
    if image_name.endswith('color.png') and i%25==1:
        #复制image_name到color_img文件夹
        shutil.copyfile(images_path+'/'+image_name, './color_img/'+image_name)
        
        


    