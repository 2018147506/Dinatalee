import cv2
import matplotlib.pyplot as plt
import numpy as np

##### To-do #####

def fftshift(img):
    '''
    This function should shift the spectrum image to the center.
    You should not use any kind of built in shift function. Please implement your own.
    '''
    height, width = img.shape
    out = img.copy()
    if (height%2 == 0):
        out[:height//2] = img[height//2:]
        out[height//2:] = img[:height//2]
    else:
        out[:height//2] = img[height//2+1:]
        out[height//2:] = img[:height//2+1]
    out2 = out.copy()
    for i in range(height):
        if (width%2 == 0):
            out[i][:width//2] = out2[i][width//2:]
            out[i][width//2:] = out2[i][:width//2]
        else:
            out[i][:width//2] = out2[i][width//2+1:]
            out[i][width//2:] = out2[i][:width//2+1]
    
    img = out.copy()
    return img

def ifftshift(img):
    '''
    This function should do the reverse of what fftshift function does.
    You should not use any kind of built in shift function. Please implement your own.
    '''
    height, width = img.shape
    out = img.copy()
    if (height%2 == 0):
        out[:height//2] = img[height//2:]
        out[height//2:] = img[:height//2]
    else:
        out[:height//2+1] = img[height//2:]
        out[height//2+1:] = img[:height//2]
    out2 = out.copy()
    for i in range(height):
        if (width%2 == 0):
            out[i][:width//2] = out2[i][width//2:]
            out[i][width//2:] = out2[i][:width//2]
        else:
            out[i][:width//2+1] = out2[i][width//2:]
            out[i][width//2+1:] = out2[i][:width//2]
    img = out.copy()
    return img

def fm_spectrum(img):
    '''
    This function should get the frequency magnitude spectrum of the input image.
    Make sure that the spectrum image is shifted to the center using the implemented fftshift function.
    You may have to multiply the resultant spectrum by a certain magnitude in order to display it correctly.
    '''
    out = np.fft.fft2(img.copy())
    fshift = fftshift(out)
    img = 20*np.log(np.abs(fshift))
    return img

def low_pass_filter(img, r=30):
    '''
    This function should return an image that goes through low-pass filter.
    '''
    height, width = img.shape
    row, col = int(height/2), int(width/2)
    
    out = np.fft.fft2(img.copy())
    lpf = fftshift(out)
    
    for i in range(height):
        for j in range(width):
            if not (in_round(i,j,row,col,r)):
                lpf[i,j] = 0.000001
    
    ishiftlpf = ifftshift(lpf)
    img_out = np.fft.ifft2(ishiftlpf)
    img = np.real(img_out)
    return img

def high_pass_filter(img, r=20):
    '''
    This function should return an image that goes through high-pass filter.
    '''
    height, width = img.shape
    row, col = int(height/2), int(width/2)
    out = np.fft.fft2(img.copy())
    hpf = fftshift(out)
    
    for i in range(height):
        for j in range(width):
            if (in_round(i,j,row,col,r)):
                hpf[i,j] = 0.000001
    
    ishifthpf = ifftshift(hpf)
    img_out = np.fft.ifft2(ishifthpf)
    img = np.real(img_out)
    return img

def denoise1(img):
    '''
    Use adequate technique(s) to denoise the image.
    Hint: Use fourier transform
    '''
    height, width = img.shape
    row, col = int(height/2), int(width/2)
    
    out = np.fft.fft2(img.copy())
    fm = fftshift(out)
    zeromat(fm,174,174)
    zeromat(fm,202,202)
    zeromat(fm,202,310)
    zeromat(fm,174,338)
    zeromat(fm,338,174)
    zeromat(fm,310,202)
    zeromat(fm,310,310)
    zeromat(fm,338,338)
    ishiftfm = ifftshift(fm)
    img_out = np.fft.ifft2(ishiftfm)
    img = np.real(img_out)
    
    return img

def denoise2(img):
    '''
    Use adequate technique(s) to denoise the image.
    Hint: Use fourier transform
    '''
    height, width = img.shape
    row, col = int(height/2), int(width/2)
    
    out = np.fft.fft2(img.copy())
    fm = fftshift(out)
    
    for i in range(height):
        for j in range(width):
            if 698 < (i - row) ** 2 + (j - col) ** 2 < 760:
                zeromat(fm,i,j)
    
    ishiftfm = ifftshift(fm)
    img_out = np.fft.ifft2(ishiftfm)
    img = np.real(img_out)
    return img

def in_round (x, y, rx, ry, r):
    if (x - rx) ** 2 + (y - ry) ** 2 <= r ** 2:
        return True
    else:
        return False

def zeromat(mat,x,y):
    mat[x-3:x+4,y-3:y+4] = 0
#################

# Extra Credit
def dft2(img):
    '''
    Extra Credit. 
    Implement 2D Discrete Fourier Transform.
    Naive implementation runs in O(N^4).
    '''
    source = np.array(img)
    height, width = img.shape
    out = np.zeros((height,width),dtype = complex)
    for i in range(height):
        for j in range(width):
            sub = 0.0
            for m in range(height):
                for n in range(width):
                    e = np.exp(- 2j * np.pi ** ((i * m) / height + (j * n) / width))
                    sub += source[m,n] * e
            out[i,j] = sub
    img = out
    return img

def idft2(img):
    '''
    Extra Credit. 
    Implement 2D Inverse Discrete Fourier Transform.
    Naive implementation runs in O(N^4). 
    '''
    source = np.array(img)
    height, width = img.shape
    out = np.zeros((height,width),dtype = complex)
    for i in range(height):
        for j in range(width):
            sub = 0.0
            for m in range(height):
                for n in range(width):
                    e = np.exp(2j * np.pi ** ((i * m) / height + (j * n) / width))
                    sub += source[m,n] * e
            out[i,j] = sub / height / width
    img = out
    return img

def fft2(img):
    '''
    Extra Credit. 
    Implement 2D Fast Fourier Transform.
    Correct implementation runs in O(N^2*log(N)).
    '''
    return img

def ifft2(img):
    '''
    Extra Credit. 
    Implement 2D Inverse Fast Fourier Transform.
    Correct implementation runs in O(N^2*log(N)).
    '''
    return img



if __name__ == '__main__':
    img = cv2.imread('task2_filtering.png', cv2.IMREAD_GRAYSCALE)
    noised1 = cv2.imread('task2_noised1.png', cv2.IMREAD_GRAYSCALE)
    noised2 = cv2.imread('task2_noised2.png', cv2.IMREAD_GRAYSCALE)

    low_passed = low_pass_filter(img)
    high_passed = high_pass_filter(img)
    denoised1 = denoise1(noised1)
    denoised2 = denoise2(noised2)

    # save the filtered/denoised images
    cv2.imwrite('low_passed.png', low_passed)
    cv2.imwrite('high_passed.png', high_passed)
    cv2.imwrite('denoised1.png', denoised1)
    cv2.imwrite('denoised2.png', denoised2)

    # draw the filtered/denoised images
    def drawFigure(loc, img, label):
        plt.subplot(*loc), plt.imshow(img, cmap='gray')
        plt.title(label), plt.xticks([]), plt.yticks([])

    drawFigure((2,7,1), img, 'Original')
    drawFigure((2,7,2), low_passed, 'Low-pass')
    drawFigure((2,7,3), high_passed, 'High-pass')
    drawFigure((2,7,4), noised1, 'Noised')
    drawFigure((2,7,5), denoised1, 'Denoised')
    drawFigure((2,7,6), noised2, 'Noised')
    drawFigure((2,7,7), denoised2, 'Denoised')

    drawFigure((2,7,8), fm_spectrum(img), 'Spectrum')
    drawFigure((2,7,9), fm_spectrum(low_passed), 'Spectrum')
    drawFigure((2,7,10), fm_spectrum(high_passed), 'Spectrum')
    drawFigure((2,7,11), fm_spectrum(noised1), 'Spectrum')
    drawFigure((2,7,12), fm_spectrum(denoised1), 'Spectrum')
    drawFigure((2,7,13), fm_spectrum(noised2), 'Spectrum')
    drawFigure((2,7,14), fm_spectrum(denoised2), 'Spectrum')

    plt.show()
