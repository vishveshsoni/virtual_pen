import cv2
import numpy as np
import time

#A required callback method that goes into the trackbar function.
def nothing(x):
    pass

def capture_color():

    # Initializing the webcam feed.
    cap = cv2.VideoCapture(0)
    cap.set(3,1280)
    cap.set(4,720)

    # Create a window named trackbars.
    cv2.namedWindow("Trackbars")

    cv2.createTrackbar("L - H", "Trackbars", 0, 179, nothing)
    cv2.createTrackbar("L - S", "Trackbars", 0, 255, nothing)
    cv2.createTrackbar("L - V", "Trackbars", 0, 255, nothing)
    cv2.createTrackbar("U - H", "Trackbars", 179, 179, nothing)
    cv2.createTrackbar("U - S", "Trackbars", 255, 255, nothing)
    cv2.createTrackbar("U - V", "Trackbars", 255, 255, nothing)

    while True:
        # If the user presses ESC then exit the program
        key = cv2.waitKey(1)
        if key == 27:
            break

        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip( frame, 1 )

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        l_h = cv2.getTrackbarPos("L - H", "Trackbars")
        l_s = cv2.getTrackbarPos("L - S", "Trackbars")
        l_v = cv2.getTrackbarPos("L - V", "Trackbars")
        u_h = cv2.getTrackbarPos("U - H", "Trackbars")
        u_s = cv2.getTrackbarPos("U - S", "Trackbars")
        u_v = cv2.getTrackbarPos("U - V", "Trackbars")

        lower_range = np.array([l_h, l_s, l_v])
        upper_range = np.array([u_h, u_s, u_v])

        mask = cv2.inRange(hsv, lower_range, upper_range)

        res = cv2.bitwise_and(frame, frame, mask=mask)

        mask_3 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        stacked = np.hstack((mask_3,frame,res))

        cv2.imshow('Trackbars',cv2.resize(stacked,None,fx=0.4,fy=0.4))

        if key == ord('s'):
            
            thearray = [[l_h,l_s,l_v],[u_h, u_s, u_v]]
            print(thearray)
            
            # Also save this array as penval.npy
            np.save('penval',thearray)
            break


    cap.release()
    cv2.destroyAllWindows()

def noice():
    load_from_disk = True

    # If true then load color range from memory
    if load_from_disk:
        penval = np.load('penval.npy')

    cap = cv2.VideoCapture(0)
    cap.set(3,1280)
    cap.set(4,720)

    kernel = np.ones((5,5),np.uint8)

    while(1):
    
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = cv2.flip( frame, 1 )
    
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        if load_from_disk:
            lower_range = penval[0]
            upper_range = penval[1]

        mask = cv2.inRange(hsv, lower_range, upper_range)

        mask = cv2.erode(mask,kernel,iterations = 1)
        mask = cv2.dilate(mask,kernel,iterations = 2)

        res = cv2.bitwise_and(frame,frame, mask= mask)

        mask_3 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        stacked = np.hstack((mask_3,frame,res))

        cv2.imshow('Trackbars',cv2.resize(stacked,None,fx=0.4,fy=0.4))
        
        k = cv2.waitKey(1)
        if k == 27:
            break

    cv2.destroyAllWindows()
    cap.release()

def tracking():

    load_from_disk = True

    if load_from_disk:
        penval = np.load('penval.npy')

    cap = cv2.VideoCapture(0)
    cap.set(3,1280)
    cap.set(4,720)

    kernel = np.ones((5,5),np.uint8)

    cv2.namedWindow('image', cv2.WINDOW_NORMAL)

    # This threshold is used to filter noise, the contour area must be 
    # bigger than this to qualify as an actual contour.
    noiseth = 700

    while(1):
    
        _, frame = cap.read()
        frame = cv2.flip( frame, 1 )

        # Convert BGR to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # If you're reading from memory then load the upper and lower 
        # ranges from there
        if load_from_disk:
                lower_range = penval[0]
                upper_range = penval[1]
                
        mask = cv2.inRange(hsv, lower_range, upper_range)
        
        # Perform the morphological operations to get rid of the noise
        mask = cv2.erode(mask,kernel,iterations = 1)
        mask = cv2.dilate(mask,kernel,iterations = 2)
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_SIMPLE)
        
        # Make sure there is a contour present and also make sure its size 
        # is bigger than noise threshold.
        if contours and cv2.contourArea(max(contours, 
                                   key = cv2.contourArea)) > noiseth:
            
            # Grab the biggest contour with respect to area
            c = max(contours, key = cv2.contourArea)
            
            # Get bounding box coordinates around that contour
            x,y,w,h = cv2.boundingRect(c)
            
            # Draw that bounding box
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,25,255),2)        

        cv2.imshow('image',frame)
        
        k = cv2.waitKey(5)
        if k == 27:
            break

    cv2.destroyAllWindows()
    cap.release()

def draw():
    load_from_disk = True
    if load_from_disk:
        penval = np.load('penval.npy')

    cap = cv2.VideoCapture(0)
    cap.set(3,1280)
    cap.set(4,720)

    kernel = np.ones((5,5),np.uint8)

    # Initializing the canvas on which we will draw upon
    canvas = None

    # Initilize x1,y1 points
    x1,y1=0,0

    # Threshold for noise
    noiseth = 800
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi',fourcc, 20.0, (1280,720))

    while(1):
        _, frame = cap.read()
        frame = cv2.flip( frame, 1 )
        
        # Initialize the canvas as a black image of the same size as the frame.
        if canvas is None:
            canvas = np.zeros_like(frame)

        height, width, channels = frame.shape

        transparent_canvas = np.zeros((height, width, 4), dtype=np.uint8)


        # Convert BGR to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # If you're reading from memory then load the upper and lower ranges 
        # from there
        if load_from_disk:
                lower_range = penval[0]
                upper_range = penval[1]
                
        # Otherwise define your own custom values for upper and lower range.
        else:             
           lower_range  = np.array([26,80,147])
           upper_range = np.array([81,255,255])
        
        mask = cv2.inRange(hsv, lower_range, upper_range)
        
        # Perform morphological operations to get rid of the noise
        mask = cv2.erode(mask,kernel,iterations = 1)
        mask = cv2.dilate(mask,kernel,iterations = 2)
        
        # Find Contours
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Make sure there is a contour present and also its size is bigger than 
        # the noise threshold.
        if contours and cv2.contourArea(max(contours, 
                                     key = cv2.contourArea)) > noiseth:
                    
            c = max(contours, key = cv2.contourArea)    
            x2,y2,w,h = cv2.boundingRect(c)
            
            # cv2.rectangle(frame,(x2,y2),(x2+1,y2+1),(0,25,255),20)
            cv2.circle(frame, (x2, y2), 20,(255,255,255), -1)
            # If there were no previous points then save the detected x2,y2 
            # coordinates as x1,y1. 
            # This is true when we writing for the first time or when writing 
            # again when the pen had disappeared from view.
            if x1 == 0 and y1 == 0:
                x1,y1= x2,y2

            else:
                # Draw the line on the canvas
                k = cv2.waitKey(50)
                if k == ord('d'):
                    canvas = cv2.line(canvas, (x1,y1),(x2,y2), [255,0,0], 7)
            
            # After the line is drawn the new points become the previous points.
            x1,y1= x2,y2

        else:
            # If there were no contours detected then make x1,y1 = 0
            x1,y1 =0,0
        

        _ , mask = cv2.threshold(cv2.cvtColor (canvas, cv2.COLOR_BGR2GRAY), 20,
        255, cv2.THRESH_BINARY)
        foreground = cv2.bitwise_and(canvas, canvas, mask = mask)
        background = cv2.bitwise_and(frame, frame,
        mask = cv2.bitwise_not(mask))
        frame = cv2.add(foreground,background)

        # Merge the canvas and the frame.
        # frame = cv2.add(frame,canvas)
        
        # Optionally stack both frames and show it.
        # stacked = np.hstack((canvas,frame))
        cv2.imshow('Trackbars',frame)#cv2.resize(frame,None,fx=0.6,fy=0.6))

        out.write(frame)

        k = cv2.waitKey(1)
        if k == 27:
            break
            
        # When c is pressed clear the canvas
        if k == ord('c'):
            canvas = None

    cv2.destroyAllWindows()
    out.release()
    cap.release()


if __name__ == '__main__':
    # capture_color()
    # noice()
    # tracking()
    draw()