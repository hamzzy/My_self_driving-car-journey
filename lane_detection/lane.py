import cv2
import matplotlib.pyplot as plt
import numpy as np
def Canny(image):
    co=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    blur=cv2.GaussianBlur(co,(5,5),0)
    cann = cv2.Canny(blur,50,150)
    return  cann


def region_interest(img):
    x = int(img.shape[1])
    y = int(img.shape[0])

    shape = np.array([[int(0), int(y)], [int(x), int(y)], [int(0.55 * x), int(0.6 * y)], [int(0.45 * x), int(0.6 * y)]])
    # define a numpy array with the dimensions of img, but comprised of zeros
    #shape = np.array([(0, y),(x/1.5,y/1.5),(x, y)])
    #channel_count = img.shape[2]
    mask = np.zeros_like(img)
    # Uses 3 channels or 1 channel for color depending on input image
    if len(img.shape) > 2:
        chanel_count=img.shape[2]
        match_mask_color = (255,)* chanel_count
    else:
        match_mask_color=[255,140,23]

    # creates a polygon with the mask color
    cv2.fillPoly(mask, np.int32([shape]), match_mask_color)
    # returns the image only where the mask pixels are not zero
    masked_image = cv2.bitwise_and(img, mask)

    return masked_image
def make_coordinate(image,line_parameter):
  slope,intercept=line_parameter
  y1=image.shape[0]
  y2=int(y1*3/5)
  x1=int((y1-intercept)/slope)
  x2=int((y2-intercept)/slope)
  return np.array([x1,y1,x2,y2])

def average_slope_gradient(image,lines):
    left_lane = []
    right_lane = []
    for line in lines:
        for x1, y1, x2, y2 in line:
            parameter=np.polyfit((x1,x2),(y1,y2),1)
            slope=parameter[0]
            intercept=parameter[1]
            if slope<0:
                left_lane.append((slope,intercept))
            else:
                right_lane.append((slope,intercept))
    left_average=np.average(left_lane,axis=0)
    right_average=np.average(right_lane,axis=0)
    left_line=make_coordinate(image,left_average)
    right_line=make_coordinate(image,right_average)
    return np.array([left_line,right_line])

def Line(image,lines):
    line_img=np.zeros_like(image)
    if lines is not None:
        for  line in lines:
          x1,y1,x2,y2 =line.reshape(4)
          cv2.line(line_img,(x1,y1),(x2,y2),color=[255, 100, 20],thickness=10)
    return line_img

def Hough_Line(image):
    hough=cv2.HoughLinesP(image,2,np.pi/60,1,lines=np.array([]),minLineLength=15,maxLineGap=5)
    return hough




def process(image):
    lane = np.copy(image)
    ca = Canny(lane)
    ma = region_interest(ca)

    hough_img = Hough_Line(ma)

    avg=average_slope_gradient(image,hough_img)
    l=Line(lane,avg)
    ims = cv2.addWeighted(image, 0.8, l, 1, 0.)
    return ims





image = cv2.imread("la.jpg")
video=cv2.VideoCapture("lan3.mp4")


while True:
    ret,frame=video.read()
    cv2.imshow("",process(image))
    if cv2.waitKey(1)==13:
        break

video.release()
cv2.destroyAllWindows()
