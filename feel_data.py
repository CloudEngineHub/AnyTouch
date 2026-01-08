import numpy as np
import matplotlib.pyplot as plt
import deepdish as dd
import csv
import os
# import progressbar
  
folderIn = 'corl2017_v2/'
# namefile = 'calandra_corl2017_036.h5'
target_folder = 'tactile_datasets/feel/'
if not os.path.exists(target_folder):
    os.mkdir(target_folder)

target_csv = open('tactile_datasets/feel/feel.csv', 'w')
csv_writer = csv.writer(target_csv)



for now_h5id in range(0,40):
    namefile = 'calandra_corl2017_'+str(now_h5id).zfill(3)+'.h5'
    if not os.path.exists(folderIn+namefile):
        continue
    print('Loading file: %s' % namefile)
    t = dd.io.load(folderIn+namefile)
    n_data = len(t)
    print("N data: %d" % n_data)

    for i in range(len(t)):
        print(t[i]['object_name'].decode('utf-8'))
        name = t[i]['object_name'].decode('utf-8')
        # name = np.frombuffer(name, dtype='uint8')
        # name = np.array2string(name)
        item_folder = target_folder + name + '/'
        vision_before_folder = item_folder + 'vision_before/'
        vision_during_folder = item_folder + 'vision_during/'
        vision_after_folder = item_folder + 'vision_after/'
        touch_before_folder = item_folder + 'touch_before/'
        touch_during_folder = item_folder + 'touch_during/'
        touch_after_folder = item_folder + 'touch_after/'
        if not os.path.exists(item_folder):
            os.mkdir(item_folder)
            os.mkdir(vision_before_folder)
            os.mkdir(vision_during_folder)
            os.mkdir(vision_after_folder)
            os.mkdir(touch_before_folder)
            os.mkdir(touch_during_folder)
            os.mkdir(touch_after_folder)

        now_id = len(os.listdir(vision_before_folder))
        plt.imsave(vision_before_folder + str(now_id) +'.png', t[i]['kinectA_rgb_before'])
        plt.imsave(vision_during_folder + str(now_id) +'.png', t[i]['kinectA_rgb_during'])
        plt.imsave(vision_after_folder + str(now_id) +'.png', t[i]['kinectA_rgb_after'])

        plt.imsave(touch_before_folder + str(now_id) +'_A.png', t[i]['gelsightA_before'])
        plt.imsave(touch_during_folder + str(now_id) +'_A.png', t[i]['gelsightA_during'])
        plt.imsave(touch_after_folder + str(now_id) +'_A.png', t[i]['gelsightA_after'])

        plt.imsave(touch_before_folder + str(now_id) +'_B.png', t[i]['gelsightB_before'])
        plt.imsave(touch_during_folder + str(now_id) +'_B.png', t[i]['gelsightB_during'])
        plt.imsave(touch_after_folder + str(now_id) +'_B.png', t[i]['gelsightB_after'])

        csv_writer.writerow([name, now_id, int(t[i]['is_gripping'])])


    

