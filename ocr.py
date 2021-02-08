import cv2
import numpy as np
import pytesseract
import os
from matplotlib import pyplot as plt
try:
    from PIL import Image
except:
    import Image



# get grayscale image
def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# noise removal
def remove_noise(image):
    return cv2.medianBlur(image,5)
 
#thresholding
def otsu_thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_OTSU)[1]

#dilation
def dilate(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.dilate(image, kernel, iterations = 1)
    
#erosion
def erode(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.erode(image, kernel, iterations = 1)

#opening - erosion followed by dilation
def opening(image):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

#canny edge detection
def canny(image):
    return cv2.Canny(image, 100, 200)

#skew correction
def deskew(image):
    coords = np.column_stack(np.where(image > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    return rotated

#template matching
def match_template(image, template):
    return cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)

def unsharp_mask(image, kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=0):
    """Return a sharpened version of the image, using an unsharp mask.
    amount: amount to be sharpened
    threshold: """
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape)) #scale to 0-255
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
    return sharpened

def apply_ocr(filename):
    # load the image as a PIL/Pillow image, apply OCR, and then delete the temp file
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    try_config = r'-c preserve_interword_spaces=1x1 --psm 5 --oem 3'
    custom_oem_psm_config = r'--oem 3 --psm 6'
    text = pytesseract.image_to_string(Image.open(filename), lang="eng", config=custom_oem_psm_config)
    # os.remove(filename)

    return text


if __name__ == "__main__":
    try:
        # load the images
        img1 = cv2.imread('sample01.png')
        img2 = cv2.imread('sample02.png')

        # convert images to grayscale
        img1_gray = get_grayscale(img1)
        img2_gray = get_grayscale(img2)
        
        # question1 -- implement otsu global thresholding and check output
        otsu_1 = otsu_thresholding(img1_gray)
        otsu_2 = otsu_thresholding(img2_gray)
        cv2.imwrite('otsu_img1.png', otsu_1)
        cv2.imwrite('otsu_img2.png', otsu_2)
        
        # text_1 = apply_ocr('otsu_img1.png')
        texts = ['otsu_img1.png', 'otsu_img2.png']
        text_files = ['otsu_threshold_1.txt', 'otsu_threshold_2.txt']
        for x, y in zip(texts, text_files):
            text = apply_ocr(x)
            with open(y, 'w+') as f:
                f.write(text)

        # check output of the otsu applied images
        # cv2.imshow('output_1', otsu_1)
        # cv2.imshow('output_2', otsu_2)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # question 2 -- Design your own algorithms to address the problem of Otsu global thresholding algorithm. You may explore different approaches such as adaptive thresholding, image enhancement, etc.
        # applying gaussian blur and adaptive thresholding

        blur1 = cv2.GaussianBlur(img1_gray,(5,5),0)
        thresh1 = cv2.adaptiveThreshold(blur1,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
        blur2 = cv2.GaussianBlur(img2_gray,(5,5),0)
        thresh2 = cv2.adaptiveThreshold(blur2,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)

        cv2.imwrite('output_adaptive_1.png', thresh1) # image 1
        cv2.imwrite('output_adaptive_2.png', thresh2) # image 2
        
        
        adaptive_texts = ['output_adaptive_1.png', 'output_adaptive_2.png']
        adaptive_files = ['adaptive_threshold_1.txt', 'adaptive_threshold_2.txt']
        for x, y in zip(adaptive_texts, adaptive_files):
            text = apply_ocr(x)
            with open(y, 'w+') as f:
                f.write(text)

        # ENHANCEMENTS ==============================================
        # kernel size -- image 1
        blur3 = cv2.GaussianBlur(img1_gray,(7,7),0)
        thresh3 = cv2.adaptiveThreshold(blur3,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
        # division -- image 1
        blur4 = cv2.GaussianBlur(img1_gray, (95,95), 0)
        div_1 = cv2.divide(img1_gray, blur4, scale=250)
        # EXPERIMENTS ==============================================
        morphed = opening(cv2.bitwise_not(thresh3)) # using morphology -- bitwise not to invert w-to-b
        morphed = cv2.bitwise_not(morphed)

        cv2.imwrite('enhance_adaptive_1.png', thresh3)
        cv2.imwrite('morphed_1.png', morphed)
        cv2.imwrite('division_1.png', div_1)

        enhance_text = ['enhance_adaptive_1.png', 'morphed_1.png', 'division_1.png']
        enhance_files = ['enhance_threshold_1.txt', 'morphed_1.txt', 'division_1.txt']
        for x,y in zip(enhance_text, enhance_files):
            text = apply_ocr(x)
            with open(y, 'w+') as f:
                f.write(text)

        # ENHANCEMENTS ==============================================
        # division -- image 2
        blur_high = cv2.GaussianBlur(img2_gray,(95,95),0)
        div_2 = cv2.divide(img2_gray, blur_high ,scale=253)

        cv2.imwrite('division_2.png', div_2)
        cv2.imwrite('blur_high.png', blur_high)

        text = apply_ocr('division_2.png')
        with open('division_2.txt', mode='w+') as f:
            f.write(text)

    except KeyboardInterrupt:
        exit