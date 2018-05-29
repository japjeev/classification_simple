#import the necessary packages
import cv2
import math
import numpy as np

#Blob class
class blobz(object):

    def __init__(self, contour):

        global currentContour
        global currentBoundingRect
        global currentBoundingArea
        global centerPosition
        global centerPositions
        global cx
        global cy
        global dblCurrentDiagonalSize
        global dblCurrentAspectRatio
        global intCurrentRectArea
        global blnCurrentMatchFoundOrNewBlob
        global blnStillBeingTracked
        global intNumofConsecutiveFramesWithoutAMatch
        global predictedNextPosition
        global numPositions
        global blnBlobCrossedTheLine

        self.predictedNextPosition = []
        self.centerPosition = []
        currentBoundingRect = []
        currentContour = []
        self.centerPositions = []

        self.currentContour = contour
        self.currentBoundingArea = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        self.currentBoundingRect = [x, y, w, h]
        cx = (2*x + w)/2
        cy = (2*y + h)/2
        self.centerPosition = [cx, cy]
        self.dblCurrentDiagonalSize = math.sqrt(w*w + h*h)
        self.dblCurrentAspectRatio = (w/(h*1.0))
        self.intCurrentRectArea = w*h
        self.blnStillBeingTracked = True
        self.blnCurrentMatchFoundOrNewBlob = True
        self.blnBlobCrossedTheLine = False
        self.intNumofConsecutiveFramesWithoutAMatch = 0
        self.centerPositions.append(self.centerPosition)
    # predicted next position is weighted sum of last 5 positions
    def predictNextPosition(self):
        numPositions = len(self.centerPositions)
        if (numPositions == 1):
            self.predictedNextPosition = [self.centerPositions[-1][-2], self.centerPositions[-1][-1]]
        if (numPositions == 2):
            deltaX = self.centerPositions[1][0] - self.centerPositions[0][0]
            deltaY = self.centerPositions[1][1] - self.centerPositions[0][1]
            self.predictedNextPosition = [self.centerPositions[-1][-2] + deltaX, self.centerPositions[-1][-1] + deltaY]
        if (numPositions == 3):
            SumofX = (self.centerPositions[2][0] - self.centerPositions[1][0])*2 + (self.centerPositions[1][0] - self.centerPositions[0][0])*1
            deltaX = (SumofX/3)
            SumofY = (self.centerPositions[2][1] - self.centerPositions[1][1])*2 + (self.centerPositions[1][1] - self.centerPositions[0][1])*1
            deltaY = (SumofY/3)
            self.predictedNextPosition = [self.centerPositions[-1][-2] + deltaX, self.centerPositions[-1][-1] + deltaY]
        if (numPositions == 4):
            SumofX = (self.centerPositions[3][0] - self.centerPositions[2][0])*3 + (self.centerPositions[2][0] - self.centerPositions[1][0])*2 + (self.centerPositions[1][0] - self.centerPositions[0][0])*1
            deltaX = (SumofX/6)
            SumofY = (self.centerPositions[3][1] - self.centerPositions[2][1])*3 + (self.centerPositions[2][1] - self.centerPositions[1][1])*2 + (self.centerPositions[1][1] - self.centerPositions[0][1])*1
            deltaY = (SumofY/6)
            self.predictedNextPosition = [self.centerPositions[-1][-2] + deltaX, self.centerPositions[-1][-1] + deltaY]
        if (numPositions >= 5):
            SumofX = (self.centerPositions[numPositions - 1][0] - self.centerPositions[numPositions - 2][0])*4 + (self.centerPositions[numPositions - 2][0] - self.centerPositions[numPositions - 3][0])*3 + (self.centerPositions[numPositions - 3][0] - self.centerPositions[numPositions - 4][0])*2 + (self.centerPositions[numPositions - 4][0] - self.centerPositions[numPositions - 5][0])*1
            deltaX = (SumofX/10)
            SumofY = (self.centerPositions[numPositions - 1][1] - self.centerPositions[numPositions - 2][1])*4 + (self.centerPositions[numPositions - 2][1] - self.centerPositions[numPositions - 3][1])*3 + (self.centerPositions[numPositions - 3][1] - self.centerPositions[numPositions - 4][1])*2 + (self.centerPositions[numPositions - 4][1] - self.centerPositions[numPositions - 5][1])*1
            deltaY = (SumofY/10)
            self.predictedNextPosition = [self.centerPositions[-1][-2] + deltaX, self.centerPositions[-1][-1] + deltaY]


def CheckIfBlobsCrossedTheLine(blobs, horizontalLinePosition):
    carCount = 0
    motorcycle = 0
    passenger = 0
    commercial = 0
    for existingBlob in blobs:
        if ((existingBlob.blnStillBeingTracked == True) and (len(existingBlob.centerPositions) >= 4) and (existingBlob.blnBlobCrossedTheLine == False)):
            if ((existingBlob.centerPositions[-1][-1] > horizontalLinePosition) and (existingBlob.centerPositions[-2][-1] <= horizontalLinePosition)):
                carCount += 1
                existingBlob.blnBlobCrossedTheLine = True
                x, y, w, h = existingBlob.currentBoundingRect
                if w >= 200 and h >= 200:
                    commercial += 1
                elif w >= 100 and h < 200:
                    passenger += 1
                else:
                    motorcycle += 1
                #print(commercial, passenger, motorcycle, x, y, w, h)
    return [carCount, commercial, passenger, motorcycle]

def matchCurrentFrameBlobsToExistingBlobs(blobs, currentFrameBlobs):
    for existingBlob in blobs:
        existingBlob.blnCurrentMatchFoundOrNewBlob = False
        existingBlob.predictNextPosition()
    for currentFrameBlob in currentFrameBlobs:
        intIndexOfLeastDistance = -1
        dblLeastDistance = 100000.0
        for i in range(len(blobs)):
            if (blobs[i].blnStillBeingTracked == True):
                dblDistance = distanceBetweenPoints(currentFrameBlob.centerPositions[-1],blobs[i].predictedNextPosition)
                if (dblDistance < dblLeastDistance):
                    dblLeastDistance = dblDistance
                    intIndexOfLeastDistance = i
        if (dblLeastDistance < (currentFrameBlob.dblCurrentDiagonalSize * 0.5)):
            blobs = addBlobToExistingBlobs(currentFrameBlob, blobs, intIndexOfLeastDistance)
        else:
            blobs, currentFrameBlob = addNewBlob(currentFrameBlob, blobs)
    for existingBlob in blobs:
        if (existingBlob.blnCurrentMatchFoundOrNewBlob == False):
            existingBlob.intNumofConsecutiveFramesWithoutAMatch = existingBlob.intNumofConsecutiveFramesWithoutAMatch + 1
        if (existingBlob.intNumofConsecutiveFramesWithoutAMatch >= 5):
            existingBlob.blnStillBeingTracked = False
    return blobs

def distanceBetweenPoints(pos1, pos2):
    if (pos2 == [] or pos2 == None):
        dblDistance = math.sqrt((pos1[0])**2 + (pos1[1])**2)
    else:
        dblDistance = math.sqrt((pos2[0] - pos1[0])**2 + (pos2[1] - pos1[1])**2)
    return dblDistance

def addBlobToExistingBlobs(currentFrameBlob, blobs, intIndex):
    blobs[intIndex].currentContour = currentFrameBlob.currentContour
    blobs[intIndex].currentBoundingRect = currentFrameBlob.currentBoundingRect
    blobs[intIndex].centerPositions.append(currentFrameBlob.centerPositions[-1])
    blobs[intIndex].dblCurrentDiagonalSize = currentFrameBlob.dblCurrentDiagonalSize
    blobs[intIndex].dblCurrentAspectRatio = currentFrameBlob.dblCurrentAspectRatio
    blobs[intIndex].blnStillBeingTracked = True
    blobs[intIndex].blnCurrentMatchFoundOrNewBlob = True
    return blobs

def addNewBlob(currentFrameBlob, blobs):
    currentFrameBlob.blnCurrentMatchFoundOrNewBlob = True
    blobs.append(currentFrameBlob)
    return blobs, currentFrameBlob

def drawBlobInfoOnImage(blobs, m1):
    for i in range(len(blobs)):
        if (blobs[i].blnStillBeingTracked == True):
            x, y, w, h = blobs[i].currentBoundingRect
            cx = blobs[i].centerPositions[-1][-2]
            cy = blobs[i].centerPositions[-1][-1]
            cv2.rectangle(m1, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.circle(m1, (int(cx), int(cy)), 2, (0,0,0), -1)
            text = str(int(cx)) + "," + str(int(cy))
            cv2.putText(m1, text, (int(blobs[i].centerPositions[-1][-2]), int(blobs[i].centerPositions[-1][-1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return m1

##Globals
fcount = 0
frame = None
blobs = []
blnFirstFrame = True
total_carCount = 0
total_commercial = 0
total_passenger = 0
total_motorcycle = 0
crossingLine = []

cam = cv2.VideoCapture('classification.mp4')

ret,frame = cam.read()
if ret is True:
    backSubtractor = cv2.createBackgroundSubtractorMOG2(history=250, detectShadows=True)
    run = True
else:
    run = False

# Default resolutions of the frame are obtained.The default resolutions are system dependent.
# We convert the resolutions from float to integer.
frame_width = int(cam.get(3))
frame_height = int(cam.get(4))

# Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
out = cv2.VideoWriter('output.avi',cv2.VideoWriter_fourcc('X','V','I','D'), 30, (frame_width,frame_height))

while(run):
    # Read a frame from the camera
    ret,frame = cam.read()

    # If the frame was properly read.
    if ret is True:
        fcount = fcount + 1

        # Create the basic black image
        mask = np.zeros(frame.shape, dtype = "uint8")

        # Draw a white, filled rectangle on the mask image
        cv2.rectangle(mask, (300, 300), (1200, 1080), (255, 255, 255), -1)
        # Apply the mask
        maskedImg = cv2.bitwise_and(frame, mask)

        # get the foreground
        foreGround = backSubtractor.apply(maskedImg, None, 0.001)
        # threshold
        thresh = cv2.threshold(foreGround, 128, 255, cv2.THRESH_BINARY)[1]
        #### Filtering ####
        kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        dilation = cv2.dilate(thresh, kernel2, iterations=2)

        (_, cnts, _) = cv2.findContours(dilation.copy(), cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE)
      
        hulls = []

        for c in range(len(cnts)):
            hull = cv2.convexHull(cnts[c])
            hulls.append(hull)

        curFrameblobs = []
        l = 0
        for c in range(len(hulls)):
            l = l + 1
            ec = blobz(hulls[c])
            if (ec.intCurrentRectArea > 1600 and \
                ec.centerPosition[0] >= 500 and \
                ec.centerPosition[1] >= 400 and \
                fcount >= 250):
                curFrameblobs.append(ec)

        horizontalLinePosition = 0
        horizontalLinePosition, cols, _ = frame.shape
        horizontalLinePosition = horizontalLinePosition*0.55

        if (blnFirstFrame == True):
            crossingLine.append([0, horizontalLinePosition])
            crossingLine.append([cols - 1, horizontalLinePosition])
            for fl in curFrameblobs:
                blobs.append(fl)
        else:
            blobs = matchCurrentFrameBlobsToExistingBlobs(blobs, curFrameblobs)

        #PrintBlobzState(blobs)
        m1 = drawBlobInfoOnImage(blobs, frame)

        cv2.line(m1, (int(crossingLine[0][0]), int(crossingLine[0][1])), (int(crossingLine[1][0]), int(crossingLine[1][1])), (0, 0, 255))
        carCount = []
        carCount = CheckIfBlobsCrossedTheLine(blobs, horizontalLinePosition)
        total_carCount = total_carCount + carCount[0]
        total_commercial = total_commercial + carCount[1]
        total_passenger = total_passenger + carCount[2]
        total_motorcycle = total_motorcycle + carCount[3]
        res_str = "P: " + str(total_passenger) + ", C: " + str(total_commercial) + ", M: " + str(total_motorcycle)
        cv2.putText(m1, str(fcount), (100, 190), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 2)
        cv2.putText(m1, res_str, (150, 590), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 2)

        cv2.imshow('CCTV Feed',m1)
        cv2.imshow('CCTV Feed2',dilation)
        #cv2.imwrite("./frames/frame%d.jpg" % fcount, thresh)
        # Write the frame into the file 'output.avi'
        out.write(m1)
        key = cv2.waitKey(10) & 0xFF
    else:
        break

    if key == 27:
        break

    blnFirstFrame = False

cam.release()
out.release()
cv2.destroyAllWindows()
