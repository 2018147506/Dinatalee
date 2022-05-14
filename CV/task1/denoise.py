import cv2
import numpy as np

def gaussian(x, sigma):
    return 1 / (2 * np.pi * (sigma ** 2)) * np.exp(-1*(x**2)/(2*(sigma**2)))

def add_padding(img, kernel_size):
    """
    Don't want to use zero-padding
    """
    
    height, width, channel = img.shape
    a = (int) (kernel_size / 2)
    out = np.zeros((height + kernel_size - 1, width + kernel_size - 1, channel), dtype = np.float64)
    out[a: a + height, a: a+width] = img.copy().astype(np.float64)
    
    for i in range(a):
        out[a:a+height,i] = out[a:a+height,a]
        out[a:a+height,a + width + i] = out[a:a+height,a+width - 1]
        
    for i in range(a):    
        out[i] = out[a]
        out[a + height + i] = out[a + height - 1]
    
    return out

def task1_2(src_path, clean_path, dst_path):
    """
    This is main function for task 1.
    It takes 3 arguments,
    'src_path' is path for source image.
    'clean_path' is path for clean image.
    'dst_path' is path for output image, where your result image should be saved.

    You should load image in 'src_path', and then perform task 1-2,
    and then save your result image to 'dst_path'.
    """
    noisy_img = cv2.imread(src_path)
    clean_img = cv2.imread(clean_path)
    result_img = None

    # do noise removal
    median1 = apply_median_filter(noisy_img.copy(),3)
    median2 = apply_median_filter(noisy_img.copy(),5)
    median3 = apply_median_filter(noisy_img.copy(),7)
    data = [calculate_rms(clean_img,median1),calculate_rms(clean_img,median2),calculate_rms(clean_img,median3)]
    arr = np.array(data)
    if np.argmin(arr) == 0:
        median = median1
        median_rms = data[0]
        med_window = 9
    elif np.argmin(arr) == 1:
        median = median2
        median_rms = data[1]
        med_window = 25
    elif np.argmin(arr) == 2:
        median = median3
        median_rms = data[2]
        med_window = 49
    bilateral1 = apply_bilateral_filter(noisy_img.copy(), 7, 1, 75)
    bilateral2 = apply_bilateral_filter(noisy_img.copy(), 9, 1, 75)
    bilateral3 = apply_bilateral_filter(noisy_img.copy(), 7, 10, 75)
    bilateral4 = apply_bilateral_filter(noisy_img.copy(), 9, 10, 75)

    data = [calculate_rms(clean_img,bilateral1),calculate_rms(clean_img,bilateral2),calculate_rms(clean_img,bilateral3),calculate_rms(clean_img,bilateral4)]
    arr = np.array(data)
    if np.argmin(arr) == 0:
        bilateral = bilateral1
        bi_window = 9
        bilateral_rms = data[0]
    elif np.argmin(arr) == 1:
        bilateral = bilateral2
        bi_window = 25
        bilateral_rms = data[1]
    elif np.argmin(arr) == 2:
        bilateral = bilateral3
        bi_window = 49
        bilateral_rms = data[2]
    elif np.argmin(arr) == 3:
        bilateral = bilateral4
        bi_window = 81
        bilateral_rms = data[3]
    
    mine = apply_my_filter(noisy_img.copy())
    
    data = [median_rms, bilateral_rms, calculate_rms(clean_img,mine)]
    arr = np.array(data)
    print(data)
    if np.argmin(arr) == 0:
        print("median filter")
        print(med_window)
        result_img = median
    elif np.argmin(arr) == 1:
        print("bilateral filter")
        print(bi_window)
        result_img = bilateral
    elif np.argmin(arr) == 2:
        print("my filter")
        print(25)
        result_img = mine
    
    cv2.imwrite(dst_path, result_img)
    pass



def apply_median_filter(img, kernel_size):
    """
    You should implement median filter using convolution in this function.
    It takes 2 arguments,
    'img' is source image, and you should perform convolution with median filter.
    'kernel_size' is an int value, which determines kernel size of median filter.

    You should return result image.
    """
    height, width, channel = img.shape
    a = (int) (kernel_size / 2)
    out = add_padding(img, kernel_size)
    temp = out.copy()
    
    for i in range(height):
        for j in range (width):
            if (img[i, j, 0] <= 50 and img[i, j, 1] <= 50 and img[i, j, 2] <= 50) or (img[i, j, 0] >= 205 and img[i, j, 1] >= 205 and img[i, j, 2] >= 205):
                for k in range(channel):
                        out[a + i, a + j, k] = np.median(temp[a+i:a+i+kernel_size,a+j:a+j+kernel_size,k])
    
    img = out[a : a+ height, a : a + width].astype(np.int64)
    return img


def apply_bilateral_filter(img, kernel_size, sigma_s, sigma_r):
    """
    You should implement bilateral filter using convolution in this function.
    It takes at least 4 arguments,
    'img' is source image, and you should perform convolution with median filter.
    'kernel_size' is a int value, which determines kernel size of average filter.
    'sigma_s' is a int value, which is a sigma value for G_s(gaussian function for space)
    'sigma_r' is a int value, which is a sigma value for G_r(gaussian function for range)

    You should return result image.
    """
    
    height, width, channel = img.shape
    a = (int) (kernel_size / 2)
    out = add_padding(img, kernel_size)
    img = np.zeros((height, width, channel), dtype = np.float64)
    
    for i in range(a, a + height):
        for j in range (a, a + width):
            for k in range(channel):
                weight = 0
                for m in range (i - a, i + a + 1):
                    for n in range (j - a, j + a + 1):
                        gs = gaussian(np.sqrt((i-m)**2 + (j-n)**2), sigma_s)
                        gr = gaussian(out[m][n][k] - out[i][j][k], sigma_r)
                        weight += gs * gr
                        img[i - a][j - a][k] += gs * gr * out[m][n][k]
                img[i - a][j - a][k] = np.round(img[i - a][j - a][k] / weight)
                
                
    return img


def apply_my_filter(img):
    """
    You should implement additional filter using convolution.
    You can use any filters for this function, except median, bilateral filter.
    You can add more arguments for this function if you need.

    You should return result image.
    """
    height, width, channel = img.shape
    
    out = add_padding(img, 5)
    
    for i in range(2, 2 + height):
        for j in range(2, 2 + width):
            for k in range(channel):
                point1 = np.ravel(out[i - 1:i + 2, j - 2:j,k])
                point1 = np.append(point1, out[i, j,k])

                point2 = np.ravel(out[i + 1:i + 3, j - 1:j + 2,k])
                point2 = np.append(point2, out[i, j,k])

                point3 = np.ravel(out[i - 1:i + 2, j + 1:j + 3,k])
                point3 = np.append(point3, out[i, j,k])

                point4 = np.ravel(out[i - 2:i, j - 1:j + 2,k])
                point4 = np.append(point4, out[i, j,k])

                point5 = np.ravel(out[i - 2:i, j - 2:j,k])
                point5 = np.append(point5, out[i, j - 1,k])
                point5 = np.append(point5, out[i - 1, j,k])
                point5 = np.append(point5, out[i, j,k])

                point6 = np.ravel(out[i+1:i + 3, j-2:j,k])
                point6 = np.append(point6, out[i, j - 1,k])
                point6 = np.append(point6, out[i - 1, j,k])
                point6 = np.append(point6, out[i, j,k])

                point7 = np.ravel(out[i + 1:i + 3, j + 1:j + 3,k])
                point7 = np.append(point7, out[i, j + 1,k])
                point7 = np.append(point7, out[i + 1, j,k])
                point7 = np.append(point7, out[i, j,k])

                point8 = np.ravel(out[i - 2:i, j + 1:j + 3,k])
                point8 = np.append(point8, out[i - 1, j ,k])
                point8 = np.append(point8, out[i, j + 1,k])
                point8 = np.append(point8, out[i, j,k])

                mini = min(np.var(point1), np.var(point2), np.var(point3), np.var(point4), np.var(point5), np.var(point6), np.var(point7), np.var(point8))

                if mini == np.var(point1):
                    out[i, j,k] = np.round(np.mean(point1))
                elif mini == np.var(point2):
                    out[i, j,k] = np.round(np.mean(point2))
                elif mini == np.var(point3):
                    out[i, j,k] = np.round(np.mean(point3))
                elif mini == np.var(point4):
                    out[i, j,k] = np.round(np.mean(point4))
                elif mini == np.var(point5):
                    out[i, j,k] = np.round(np.mean(point5))
                elif mini == np.var(point6):
                    out[i, j,k] = np.round(np.mean(point6))
                elif mini == np.var(point7):
                    out[i, j,k] = np.round(np.mean(point7))
                elif mini == np.var(point8):
                    out[i, j,k] = np.round(np.mean(point8))
            
    img = out[2 : 2 + height, 2 : 2 + width].astype(np.int)
    return img


def calculate_rms(img1, img2):
    """
    Calculates RMS error between two images. Two images should have same sizes.
    """
    if (img1.shape[0] != img2.shape[0]) or \
            (img1.shape[1] != img2.shape[1]) or \
            (img1.shape[2] != img2.shape[2]):
        raise Exception("img1 and img2 should have same sizes.")

    diff = np.abs(img1 - img2)
    diff = np.abs(img1.astype(dtype=np.int) - img2.astype(dtype=np.int))
    return np.sqrt(np.mean(diff ** 2))
