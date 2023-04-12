import numpy as np
from skimage.measure import block_reduce
from ParamConfig import *
from PathConfig import *
from LibConfig import *
import skimage
import scipy.io
from IPython.core.debugger import set_trace
'''
def PlotRes(pd, gt, label_dsp_dim, label_dsp_blk, dh, minvalue, maxvalue, font2, font3, SavePath):
    PD = pd.reshape(label_dsp_dim[0], label_dsp_dim[1]).swapaxes(0,1) #(301,2001)
    GT = gt.reshape(label_dsp_dim[0], label_dsp_dim[1]).swapaxes(0,1)
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    im1 = ax1.imshow(GT, extent=[0, label_dsp_dim[1] * label_dsp_blk[1] * dh / 5000., \
                                 0, label_dsp_dim[0] * label_dsp_blk[0] * dh / 1000.], vmin=minvalue, vmax=maxvalue)
    divider = make_axes_locatable(ax1)
    cax1 = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im1, ax=ax1, cax=cax1).set_label('Velocity (m/s)')
    plt.tick_params(labelsize=12)
    for label in ax1.get_xticklabels() + ax1.get_yticklabels():
        label.set_fontsize(14)
    ax1.set_xlabel('Position (km)', font2)
    ax1.set_ylabel('Depth (km)', font2)
    ax1.set_title('Ground truth', font3)
    ax1.invert_yaxis()
    plt.subplots_adjust(bottom=0.15, top=0.92, left=0.08, right=0.98)
    plt.savefig(SavePath + 'GT', transparent=True)

    fig2, ax2 = plt.subplots(figsize=(6, 4))
    im2 = ax2.imshow(PD, extent=[0, label_dsp_dim[1] * label_dsp_blk[1] * dh / 5000., \
                                 0, label_dsp_dim[0] * label_dsp_blk[0] * dh / 1000.], vmin=minvalue, vmax=maxvalue)

    plt.tick_params(labelsize=12)
    for label in ax2.get_xticklabels() + ax2.get_yticklabels():
        label.set_fontsize(14)
    ax2.set_xlabel('Position (km)', font2)
    ax2.set_ylabel('Depth (km)', font2)
    ax2.set_title('Prediction', font3)
    ax2.invert_yaxis()
    plt.subplots_adjust(bottom=0.15, top=0.92, left=0.08, right=0.98)
    plt.savefig(SavePath + 'PD', transparent=True)
    # plt.show(fig1)
    # plt.show(fig2)
    plt.show()
    plt.close()


#filename_seis = './data/train_data/SimulateData/georec_train/'+'georec1.mat'
datafilename  = 'georec'
filename_seis = '../data/train_data/SimulateData/georec_train/'+datafilename+str(2)
print(filename_seis)
# Load .mat data
data1_set = scipy.io.loadmat(filename_seis)
#print(data1_set)
#print(type(data1_set)) #<class 'dict'>
#print(data1_set[str(dataname)].shape) #(301, 5, 2001)
data1_set = np.float32(data1_set[str('Rec')].reshape(2001,301,5))

num = 1
minvalue = -0.5
maxvalue = 0.5
font2 = {'family': 'Times New Roman',
    'weight': 'normal',
    'size': 17,
    }
font3 = {'family': 'Times New Roman',
    'weight': 'normal',
    'size': 21,
    }
label_dsp_dim = (301,2001)
label_dsp_blk = (1,1)
dh            = 10
results_dir = '../'
#print(data1_set[:,:,num])
#data1_set = data1_set[:,:,num].reshape(label_dsp_dim[0], label_dsp_dim[1])
#print(data1_set.shape) #(2001,301)

#PlotRes(data1_set[:,:,num],data1_set[:,:,num],label_dsp_dim,label_dsp_blk,dh,minvalue,maxvalue,font2,font3,SavePath=results_dir)


image = np.arange(2*2*3).reshape(2, 2, 3)
print(image)
print(block_reduce(image, block_size=(1, 2, 3), func=np.mean))

a=np.array([1,2,3])
b=list(a)
#print(type(b),b.shape,np.shape(b))  #'list' object has no attribute 'shape'
print(type(a),a.shape,np.shape(a))

c = np.array([[1, 2, 3, 4],[4, 5, 6, 7], [7, 8, 9, 10]])
c.shape # (3L, 4L)
#c.shape=4,-1   #
print(c.reshape((2,-1)))

d=c.reshape([2,-1])  #行变成2行 列不变
d[1:2]=100
print(c)'''

''''''
import PIL.Image as Image
import os

# 需要拼接图片的文件位置
IMAGES_PATH_1 = r'../results/SimulateResults/GT/'
IMAGES_PATH_2 = r'../results/SimulateResults/01/'
IMAGES_PATH_3 = r'../results/SimulateResults/02/'
IMAGES_PATH_4 = r'../results/SimulateResults/03/'
IMAGES_FORMAT = ['.png', '.PNG']  # 图片格式
IMAGE_XSIZE = 600  # 每张图片横向的大小
IMAGE_YSIZE = 400  #每张照片竖向的大小
IMAGE_ROW = 3  # 合并成一张图后，一共有几行
IMAGE_COLUMN = 4  # 合并成一张图后，一共有几列
IMAGE_SAVE_PATH = r'../results/SimulateResults/01/gisoracle.png'  # 图片拼接后的图片位置

# 获取图片集地址下的所有图片名称
image_names_1 = [name for name in os.listdir(IMAGES_PATH_1) for item in IMAGES_FORMAT if
               os.path.splitext(name)[1] == item]
image_names_2 = [name for name in os.listdir(IMAGES_PATH_2) for item in IMAGES_FORMAT if
               os.path.splitext(name)[1] == item]
image_names_3 = [name for name in os.listdir(IMAGES_PATH_3) for item in IMAGES_FORMAT if
               os.path.splitext(name)[1] == item]
image_names_4 = [name for name in os.listdir(IMAGES_PATH_4) for item in IMAGES_FORMAT if
               os.path.splitext(name)[1] == item]

# 简单的对于参数的设定和实际图片集的大小进行数量判断
if len(image_names_1 + image_names_2 + image_names_3 + image_names_4) != IMAGE_ROW * IMAGE_COLUMN:
    raise ValueError("合成图片的参数和要求的数量不能匹配！")

# 定义图像拼接函数
def image_compose():
    to_image = Image.new('RGB', (IMAGE_COLUMN * IMAGE_XSIZE, IMAGE_ROW * IMAGE_YSIZE))  # 创建一个新图
    # 循环遍历，把每张图片按顺序粘贴到对应位置上
    x = 0;
    for y in range(1, IMAGE_ROW + 1):
        print("processing No. " + str(y))
        from_image_1 = Image.open(IMAGES_PATH_1 + image_names_1[x]).resize(
            (IMAGE_XSIZE, IMAGE_YSIZE), Image.ANTIALIAS)
        to_image.paste(from_image_1, (0 * IMAGE_XSIZE, (y - 1) * IMAGE_YSIZE))

        from_image_2 = Image.open(IMAGES_PATH_2 + image_names_2[x]).resize(
            (IMAGE_XSIZE, IMAGE_YSIZE), Image.ANTIALIAS)
        to_image.paste(from_image_2, (1 * IMAGE_XSIZE, (y - 1) * IMAGE_YSIZE))

        from_image_3 = Image.open(IMAGES_PATH_3 + image_names_3[x]).resize(
            (IMAGE_XSIZE, IMAGE_YSIZE), Image.ANTIALIAS)
        to_image.paste(from_image_3, (2 * IMAGE_XSIZE, (y - 1) * IMAGE_YSIZE))

        from_image_4 = Image.open(IMAGES_PATH_4 + image_names_4[x]).resize(
            (IMAGE_XSIZE, IMAGE_YSIZE), Image.ANTIALIAS)
        to_image.paste(from_image_4, (3 * IMAGE_XSIZE, (y - 1) * IMAGE_YSIZE))
        x = x + 1
    return to_image.save(IMAGE_SAVE_PATH)  # 保存新图


image_compose()  # 调用函数
