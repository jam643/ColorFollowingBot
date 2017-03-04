# -*- coding: utf-8 -*-
#!/usr/bin/env python
"""
Created on Sat Nov 12 15:52:06 2016

@author: Jesse
"""

import numpy as np
import cv2
import time
import sys

try:
    import PCA9685 as servo
    existServo = True
except:
    existServo = False

def main(stream = True):
    """Main Function"""
    # define global
    global hsv
    # create video capture object using webcam
    cap = cv2.VideoCapture(0)

    if existServo:
        servoController = servoClass()

    # main loop
    while(True):
        global width, height
        # Capture webcam image
        ret, frame = cap.read()

        # determine if frame contains an image
        if not ret:
            pass
        else:
            # resize the frame
            frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)

            height, width, channels = frame.shape
            # convert bgr image to hsv
            hsv =  cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # update and draws tracking circle for each color being tracked
            for idx, colorObject in enumerate(colorTracker.instances):
                # update position of tracking circle
                colorObject.update(hsv)
                if existServo:
                    servoController.update(colorObject)

                # redraw tracking circle
                if stream:
                    colorObject.draw(frame)
                    cv2.imshow('frame2',colorObject.mask)

            if not colorTracker.instances or stream:
                cv2.imshow('frame',frame)
                
            # attach callback function upon mouse click
            cv2.setMouseCallback('frame',getHsv)

        # quit if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

class colorTracker(object):
    """Object that tracks selected color and draws bounding circle"""

    # minimum pixel size to be tracked
    minRad = 4
    instances = []

    def __init__(self,hsvLimits):
        """initial function call upon object creation

        Parameters
        ----------
        hsvLimits
            color range to be tracked
        """
        # initialize variables
        self.center = None
        self.radius = None
        self.mask = None
        self.hsvLimits = hsvLimits
        colorTracker.instances.append(self)

    def update(self,hsv):
        """Update location of bounding circle

        Parameters
        ----------
        hsv
            current image in hsv colorspace
        """
        # initialize center and radius of bounding circle
        self.center = (np.nan,np.nan)
        self.radius = np.nan

        # create masked image based on hsv limits
        if self.hsvLimits[0][0] > self.hsvLimits[1][0]:
            # creates two masks due to cylindrical nature of hsv
            mask1 = cv2.inRange(hsv, self.hsvLimits[0], np.insert(self.hsvLimits[1][1:3],0,180))
            mask2 = cv2.inRange(hsv, np.insert(self.hsvLimits[0][1:3],0,0), self.hsvLimits[1])
            # combine masks
            self.mask = mask1 | mask2
        else:
            # create mask
            self.mask = cv2.inRange(hsv, self.hsvLimits[0], self.hsvLimits[1])

        # find contours in the mask and initialize the current
        # (x, y) center of the ball
        cnts = cv2.findContours(self.mask.copy(), cv2.RETR_EXTERNAL,
    		cv2.CHAIN_APPROX_SIMPLE)[-2]
        # only proceed if at least one contour was found
        if len(cnts) > 0:
            # find the largest contour in the mask, then use
            # it to compute the minimum enclosing circle and
            # centroid
            maxCnt = max(cnts, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(maxCnt)

            ## get area info
            # M = cv2.moments(maxCnt)

            # only recognize if larger than min size
            if radius > self.minRad:
                # set center and radius of object
                # self.center = [int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])]
                self.center = [int(x),int(y)]
                self.radius = radius

    def draw(self,frame):
        """Draw bounding circle around tracked color

        Parameters
        ----------
        frame
            BGR image

        Returns
        ----------
        frame
            BGR with bounding circle drawn
        """
        if not np.isnan(self.center).any():

            # color of bounding of circle is upper hsv limit
            color = cv2.cvtColor(np.uint8([[self.hsvLimits[1]]]),
                                 cv2.COLOR_HSV2BGR)[0][0]
            # draw bounding circle
            cv2.circle(frame,
                       (int(self.center[0]),
                        int(self.center[1])),
                        int(self.radius),
                        (int(color[0]),int(color[1]),int(color[2])),
                        2)

            # color of center point is lower hsv limit
            color = cv2.cvtColor(np.uint8([[self.hsvLimits[0]]]),
                                 cv2.COLOR_HSV2BGR)[0][0]

            # draw center point
            cv2.circle(frame,
                       tuple(self.center),
                       5,
                       (int(color[0]),
                        int(color[1]),
                        int(color[2])),
                        -1)
        return frame

class servoClass(object):
    """Controls servo"""

    # max and min pulses corresponding to servo extremes
    MinPulse = 200
    MaxPulse = 700
    # servo port numbers
    SERVO_X = 14
    SERVO_Y = 15
    # proportional controller constant
    k = 0.06

    def __init__(self):
        self.pwm = servo.PWM()
        self.pwm.frequency = 60

        offset_x = 0
        offset_y = 0
        try:
            for line in open('config'):
                if line[0:8] == 'offset_x':
                    offset_x = int(line[11:-1])
                if line[0:8] == 'offset_y':
                    offset_y = int(line[11:-1])
        except:
            pass
        self.theta_X_min = self.MinPulse + offset_x
        self.theta_X_max = self.MaxPulse + offset_x
        self.theta_Y_min = self.MinPulse + offset_y
        self.theta_Y_max = self.MaxPulse + offset_y

        self.theta_X = (self.theta_X_max + self.theta_X_min)/2
        self.theta_Y = (self.theta_Y_max + self.theta_Y_min)/2

        self.omega_X = 0
        self.omega_Y = 0

        self.pwm.write(self.SERVO_X,0,self.theta_X)
        self.pwm.write(self.SERVO_Y,0,self.theta_Y)

    def update(self,colorObject):
        global width, height
        
        error_theta_X,error_theta_Y = np.subtract(colorObject.center,[width/2,height/2])
        if abs(error_theta_Y) < height/15:
            error_theta_Y = 0
        if abs(error_theta_X) < width/15:
            error_theta_X = 0
        self.omega_Y = -self.k*error_theta_Y
        self.omega_X = -self.k*error_theta_X
        if not np.isnan(self.omega_X):
            self.theta_X += self.omega_X
            self.theta_Y += self.omega_Y
            self.pwm.write(self.SERVO_X,0,int(self.theta_X))
            self.pwm.write(self.SERVO_Y,0,int(self.theta_Y))

def getHsv(event,x,y,flags,param):
    """Callback function that creates circle object that tracks selected color
    upon left click from user

    Parameters
    ----------
    event
        contains mouse events generated by user
    x
        x pixel location of mouse
    y
        y pixel location of mouse
    flags
        flags passed by OpenCV
    param
        params passed by OpenCV

    """
    global hsv
    # checks if event wasa left click and that there haven't been more than max
    # allowable circle objects to be tracked
    if event == cv2.EVENT_LBUTTONDOWN:
        # hsv color at clicked location
        color = hsv[y][x]
        # allowable +/- variation in hue, saturation, and value respectively
        spread = [15,60,60]
        # calculate lower and upper hsv limits to be tracked
        colorLower = [color[0]-spread[0],max(0,color[1]-spread[1]),max(0,color[2]-spread[2])]
        colorUpper = [color[0]+spread[0],min(255,color[1]+spread[1]),min(255,color[2]+spread[2])]
        # allow hue to wrap around
        if colorLower[0] < 0:
            colorLower[0] += 180
        if colorUpper[0] > 180:
            colorUpper[0] -= 180
        # add lower and upper hsv limits to array
        colorRange = np.asarray([colorLower, colorUpper])
        
        # create new circle object that will track the color in colorRange
        # if object already created, update tracking color
        if colorTracker.instances:
            colorTracker.instances[-1].hsvLimits = colorRange
        else:
        # create new object
            colorTracker(colorRange)

if __name__ == "__main__":
    # run program with or without video to improve efficiency
    if len(sys.argv) == 2:
        main(bool(int(sys.argv[1])))
    else:
        main() 
