import cv2
import numpy as np
import random
import copy

# 图片数据
image_Labels=[[2,1,0,0],
 [0,1,2,0],
 [3,0,1,0],
 [1,0,0,3],
 [0,0,1,1],
 [2,0,1,0],
 [0,1,0,2],
 [3,1,0,0],
 [1,0,3,0],
 [0,0,1,2],
 [0,3,1,0],
 [1,0,0,2],
 [0,3,0,0],
 [1,0,0,0],
 [3,0,0,0],
 [0,0,1,0],
 [0,0,0,4],
 [0,2,0,0],
 [0,4,0,0],
 [0,0,0,3],
 [0,0,0,1],
 [0,0,4,0],
 [0,1,0,0],
 [4,0,0,0],
 [0,0,2,0],
 [0,3,0,0],
 [1,1,1,0],
 [2,0,0,0],
 [0,0,0,2],
 [0,1,0,3],
 [0,1,1,1],
 [1,0,1,0],
 [1,1,0,0],
 [3,0,0,1],
 [0,0,0,5],
 [5,0,0,0],
 [0,1,0,1],
 [0,0,1,3],
 [0,5,0,0],
 [1,3,0,0],
 [1,2,0,0],
 [2,0,0,1],
 [0,1,1,0],
 [1,1,1,1],
 [1,0,1,1],
 [0,1,3,1],
 [0,0,5,0],
 [0,0,3,1],
 [0,0,2,1],
 [0,2,0,1],
 [1,0,2,0],
 [1,0,0,1],
 [0,3,0,1],
 [0,0,3,0]
 ]

# 图像显示
def cv_Show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 图像旋转
def rotate_Image(image_1, angle):
    # 读取图像
    image = image_1

    # 获取图像尺寸
    height, width = image_1.shape[:2]

    # 计算旋转的中心点
    center = (width / 2, height / 2)

    # 计算旋转矩阵
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    # 计算旋转后图像的新边界尺寸
    cos = np.abs(rotation_matrix[0, 0])
    sin = np.abs(rotation_matrix[0, 1])

    new_width = int((height * sin) + (width * cos))
    new_height = int((height * cos) + (width * sin))

    # 调整旋转矩阵以考虑平移
    rotation_matrix[0, 2] += (new_width / 2) - center[0]
    rotation_matrix[1, 2] += (new_height / 2) - center[1]

    # 进行仿射变换
    rotated_image = cv2.warpAffine(image, rotation_matrix, (new_width, new_height))

    return rotated_image

# 图像覆盖
def image_Cover(img_1, img_2, X_temp, Y_temp, R_Angle):
    # 读取A和B图片  A大B小, 1大2小
    img_A = copy.deepcopy(img_1)
    #img_B0 = cv2.imread("image_B.png")
    img_B = rotate_Image(img_2, -R_Angle)
    # 获取B图片的大小信息
    rows, cols, channels = img_B.shape
    X = X_temp - cols//2 - 1
    Y = Y_temp - rows//2 - 1
    # 抠出B图需要覆盖的区域
    roi = img_A[Y : Y + rows, X : X + cols]
    # 制作蒙版mask和mask_inv
    img_B_gray = cv2.cvtColor(img_B, cv2.COLOR_BGR2GRAY) # 将原图片转为灰度（单值）图片
    ret, mask = cv2.threshold(img_B_gray, 30, 255, cv2.THRESH_BINARY) # 使用阈值将非黑部分转化全白（255）
    mask_inv = cv2.bitwise_not(mask) # 将图片中的黑白反转

    # 图像叠加
    img1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
    img2_fg = cv2.bitwise_and(img_B, img_B, mask=mask)
    dst = cv2.add(img1_bg, img2_fg)
    # 将叠加后的图像放回到A图
    img_A[Y:Y + rows, X:X + cols] = dst
    return img_A

# 判断图像是否重叠
def check_rotated_rect_overlap(rect1, rect2):
    """
    检查两个旋转矩形是否重叠

    :param rect1: 第一个旋转矩形，形式为((center_x, center_y), (width, height), angle)
    :param rect2: 第二个旋转矩形，形式为((center_x, center_y), (width, height), angle)
    :return: 如果重叠返回True，否则返回False
    """
    # 检查两个矩形是否重叠
    intersection_type, _ = cv2.rotatedRectangleIntersection(rect1, rect2)
    if intersection_type == cv2.INTERSECT_FULL:
        return 2 # 全包含
    elif intersection_type == cv2.INTERSECT_PARTIAL:
        return 1 # 有重叠
    else:
        return 0 # 无重叠
    
# 抽取图片，赋予数据
def extract_Image(base_Image, image_Labels):
    # base_Image.shape = (2048*2048)
    base_Wide = base_Image.shape[1] # 宽
    base_High = base_Image.shape[0] # 高
    # 所抽取的图片数量
    image_Num = random.randint(1, 6)
    time_Num = 0
    while time_Num < 1000:
        # 防止进入死循环
        time_Num += 1
        #根据数量抽取图片，以及图片属性
        rect_Image = [] # 存储需要旋转的矩形信息
        info_Images = [] # 存储图片数据
        is_OK = 1
        for n in range(image_Num):
            # 随机抽取图片
            temp = random.randint(1, 54)
            image_N = cv2.imread(f"meta_images/{temp}.jpg")
            # 随机中心坐标
            X_Random = random.randint(181, base_Wide - 181)
            Y_Random = random.randint(181, base_High - 181)
            # 随机旋转角度
            R_Angle = random.randint(0, 359)

            rect_Temp = ((X_Random, Y_Random), (image_N.shape[1], image_N.shape[0]), R_Angle) # 存储需要旋转的矩形信息（中心坐标，大小（宽，高），旋转角度）
            rect_Image.append(rect_Temp)
            info_Image = [image_N, (X_Random, Y_Random), R_Angle, image_Labels[temp - 1]]# 存储图片数据（图片，中心坐标，旋转角度，包含各种水果数量）
            info_Images.append(info_Image)
        for i in range(len(rect_Image)):
            #j_Boundary = check_rotated_rect_overlap(rect_Base, rect_Image[i])
            #if j_Boundary != 2:
                #is_OK = 0
                #break
            for j in range(i + 1, len(rect_Image)):
                j_Overlap = check_rotated_rect_overlap(rect_Image[i], rect_Image[j])
                if j_Overlap != 0:
                    is_OK = 0
                    break
        if is_OK == 1:
            return info_Images
    return 0
def image_Generate(base_Image, image_Labels, image_Num):
    res_Images = []
    is_Lings = [] # 各个生成图像是否满足按铃
    iteration=0
    while iteration < image_Num:
        info_Images = copy.deepcopy(extract_Image(base_Image, image_Labels))
        if info_Images == 0:
            continue
        iteration += 1
        res_Image = copy.deepcopy(base_Image)
        res_Image_Lab = [0, 0, 0 ,0] # 统计各种水果数量
        for i in range(len(info_Images)):
            res_Image = copy.deepcopy(image_Cover(res_Image, info_Images[i][0], info_Images[i][1][0], info_Images[i][1][1], info_Images[i][2]))
            res_Image_Lab = copy.deepcopy(list(map(lambda x, y: x + y, res_Image_Lab, info_Images[i][3])))
        is_Ling = 0
        for fruit_Num in res_Image_Lab:
            if fruit_Num != 0 and fruit_Num % 5 == 0:
                is_Ling = 1
                break
        is_Lings.append(is_Ling)
        cv2.imwrite(f"test_Images/{iteration - 1}.jpg", res_Image)
        #res_Images.append(res_Image)
        if iteration % 100 == 0:
            print(iteration)
    #return res_Images, is_Lings
    return is_Lings

# 显示图片
base_Image = cv2.imread("Black-1024-1024.jpg")
is_Lings = image_Generate(base_Image, image_Labels, 2000)
#for i in range(len(res_Images)):
#    cv2.imwrite(f"Res_Images/{i}.jpg", res_Images[i])

with open('test_labels.txt', 'w') as file:
    for item in is_Lings:
        file.write(str(item) + ' ')
print(is_Lings)