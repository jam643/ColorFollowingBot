# -*- coding: utf-8 -*-
"""
Created on Sat Nov 12 15:52:06 2016

@author: Jesse
"""

import numpy as np
import cv2

def main():
    """Main Function"""
    # define global 
    global hsv, circleObjects
    # create video capture object using webcam
    cap = cv2.VideoCapture(0)
    # initialize list to contain colored objects to be tracked
    circleObjects = []

    # main loop
    while(True):
        # Capture webcam image
        ret, frame = cap.read()
        
        # determine if frame contains an image
        if not ret:
            print "NO IMAGE!"
        else:
            # convert bgr image to hsv
            hsv =  cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # update and draws tracking circle for each color being tracked
            for idx, object in enumerate(circleObjects):
                # update position of tracking circle
                object.update(hsv)
                # redraw tracking circle
                object.draw(frame)
                # draw all binary images masking tracked color
                cv2.imshow('frame'+str(idx),object.opening)
            
            cv2.imshow('frame',frame)
            # attach callback function upon mouse click
            cv2.setMouseCallback('frame',getHsv)
            
        # quit if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

class colorCircle(object):
    """Object that tracks selected color and draws bounding circle"""
    
    # minimum pixel size to be tracked
    minRad = 10
    # kernel size to reduce image noise
    kernel = np.ones((10,10),np.uint8)
    
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
        self.openeing = None
        self.hsvLimits = hsvLimits
    def update(self,hsv):
        """Update location of bounding circle
        
        Parameters
        ----------
        hsv
            current image in hsv colorspace    
        """
        # initialize center of bounding circle
        self.center = None
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
            
        # reduce noise and fill in gaps in mask
        self.opening = cv2.morphologyEx(self.mask, cv2.MORPH_OPEN, self.kernel)
        #opening = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel2)
        
        # find contours in the mask and initialize the current
        # (x, y) center of the ball
        cnts = cv2.findContours(self.opening.copy(), cv2.RETR_EXTERNAL,
    		cv2.CHAIN_APPROX_SIMPLE)[-2]
        # only proceed if at least one contour was found
        if len(cnts) > 0:
            # find the largest contour in the mask, then use
            # it to compute the minimum enclosing circle and
            # centroid
            c = max(cnts, key=cv2.contourArea)
            ((x, y), self.radius) = cv2.minEnclosingCircle(c)
            M = cv2.moments(c)
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            
            # only recognize if larger than min size
            if self.radius > self.minRad:
                self.center = center
                
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
        if self.center:
            
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
                       self.center, 
                       5,
                       (int(color[0]),
                        int(color[1]),
                        int(color[2])), 
                        -1)
        return frame

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
    global hsv, circleObjects

    # checks if event wasa left click and that there haven't been more than max
    # allowable circle objects to be tracked
    if event == cv2.EVENT_LBUTTONDOWN and len(circleObjects) < 5:
        # hsv color at clicked location
        color = hsv[y][x]
        # allowable +/- variation in hue, saturation, and value respectively
        spread = [20,50,50]
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
        object = colorCircle(colorRange)
        # add circle object to list of objects
        circleObjects.append(object)
        

if __name__ == "__main__":
    main()