import numbers
from turtle import circle
import cv2
import os
import numpy as np
import math
import tkinter as tk
import matplotlib. pyplot as plt
from tkinter import filedialog
import csv

### CSV CREATION
header = ['ID','ExtM','intM','IntS','ExtS','DistI','DistE','Dist']
with open('Path/to/result/file', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)


    ### USER INTERFACE ###

    # Window creation
    root = tk.Tk()
    root.title("Open field test")
    root.geometry("600x400")

    fpsDivisionFactor = 1
    experienceName = ""

    # Title
    lblTitle = tk.Label(root,text = "Open field test\n by Rayanne Martin & Clément Leroy", font = ("Arial",16), foreground = "white", background="darkblue")
    lblTitle.place(x=0,y=0,width=600)

    # Browse
    def chooseFile():
        root.sourceFile = filedialog.askopenfilenames(parent=root, title='Choose the video')



    b_chooseFile = tk.Button(root, text = "Browse", width = 20, height = 1, command = chooseFile)
    b_chooseFile.place(x = 30,y = 70)
    b_chooseFile.width = 100

    # Mooving average
    def moving_average(a, n) :
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1:] / n
    # Continue
    def continueButton():
        global fpsDivisionFactor
        global experienceName
        fpsDivisionFactor = int(FPSentry.get())
        experienceName = ENentry.get()
        root.destroy()
    b_continue = tk.Button(root, text = "Measure", width = 20, height = 1, command = continueButton)
    b_continue.place(x = 30, y = 340)
    b_continue.width = 100

    # FPS Division
    lblFPS = tk.Label(root, text = "FPS division (integer)",font = ("Arial",8), foreground = "black")
    lblFPS.place(x = 30,y=115,width=100)
    FPSentry = tk.Entry(root, width = 3)
    FPSentry.insert(-1, '1')
    FPSentry.place(x = 155,y=115)

    # Experience name
    lblEN = tk.Label(root, text = "Experience name:",font = ("Arial",8), foreground = "black")
    lblEN.place(x = 30,y=150,width=93)
    ENentry = tk.Entry(root, width = 24)
    ENentry.insert(-1, 'Measures')
    ENentry.place(x = 30,y=175)

    fA =0
    root.mainloop()
    fileDir = root.sourceFile
    fileNumber = len(fileDir)
    rayon =[]
    centre=[]

    while fA < fileNumber :
        # Take first two frames of the video
        h =0
        cap = cv2.VideoCapture(fileDir[fA])
        while h < 25 :
            ret,frame = cap.read()
            h = h+1
        
        framePrevious = frame
        ret,frame = cap.read()
        frameCurrent = frame
        ### DRAW THE CIRCLE
        firstFrame = frame.copy()

        font                   = cv2.FONT_HERSHEY_SIMPLEX
        fontScale              = 0.5
        fontColor              = (0,0,0)
        thickness              = 1
        lineType               = 2

        firstFrame = cv2.resize(firstFrame, (960,540), interpolation = cv2.INTER_NEAREST)
        cv2.putText(firstFrame,'Click on three points to define the inner circle', (10,50), font, fontScale,fontColor,thickness,lineType)
        pointList = [None,None,None]
        pointNumber = -1

        def define_circle(p1, p2, p3):
            """
            Returns the center and radius of the circle passing the given 3 points.
            In case the 3 points form a line, returns (None, infinity).
            """
            temp = p2[0] * p2[0] + p2[1] * p2[1]
            bc = (p1[0] * p1[0] + p1[1] * p1[1] - temp) / 2
            cd = (temp - p3[0] * p3[0] - p3[1] * p3[1]) / 2
            det = (p1[0] - p2[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p2[1])
            if abs(det) < 1.0e-6:
                return (None, np.inf)
            cx = (bc*(p2[1] - p3[1]) - cd*(p1[1] - p2[1])) / det
            cy = ((p1[0] - p2[0]) * cd - (p2[0] - p3[0]) * bc) / det
            radius = np.sqrt((cx - p1[0])**2 + (cy - p1[1])**2)
            return ((cx, cy), radius)

        
        def click_event(event, x, y, flags, params):
            
            global pointNumber
            global circleCenter 
            global circleRadius
            firstFrame_copy = firstFrame.copy()

            if event == cv2.EVENT_LBUTTONDOWN:
                pointNumber += 1
                pointList[pointNumber%3]=(x,y)
                if pointNumber>=2:
                    ((cx, cy), radius) = define_circle(pointList[0], pointList[1], pointList[2])
                    print(str(cx) +"  " + str(cy))
                    circleCenter = (int(cx),int(cy))
                    print(circleCenter)
                    circleRadius = int(radius)
                    print(circleRadius)
                    
            
                    cv2.circle(firstFrame_copy, (int(cx), int(cy)), int(radius), (255,0,0), 2)
                    cv2.putText(firstFrame_copy,'Press any key to continue', (10,70), font, fontScale,fontColor,thickness,lineType)
                for index in range (min(3, pointNumber+1)):
                    cv2.circle(firstFrame_copy, pointList[index], 3, (255,255,0), -1)
                cv2.imshow('Draw the circle', firstFrame_copy)

        cv2.imshow('Draw the circle', firstFrame)
        cv2.setMouseCallback('Draw the circle', click_event)
        cv2.waitKey(0)
        rayon.append(circleRadius)
        centre.append(circleCenter)
        print(rayon)
        print(centre)
        
        cv2.destroyWindow('Draw the circle')   
        fA = fA +1

    fA =0

    ratio = 0.8





    while fA < fileNumber :


        filename = fileDir[fA]
        filename = filename[0:len(filename)-4]
        path = fileDir
        ### PLAY THE VIDEO AND MEASURE

        cap = cv2.VideoCapture(fileDir[fA])
        h=0
        while h < 25 :
            ret,frame = cap.read()
            h = h+1
        error = True
        while(error) :
                try :
                    ret ,frame = cap.read()
                    frame = cv2.resize(frame, (1920,1080), interpolation = cv2.INTER_NEAREST)
                except Exception:
                    error = True
                finally:
                    error = False
        cv2.circle(frame, (centre[fA][0]*2, centre[fA][1]*2), rayon[fA]*2+1000, (255,255,255), 2000)
        framePrevious = frame
        error = True
        while(error) :
                try :
                    ret ,frame = cap.read()
                    frame = cv2.resize(frame, (1920,1080), interpolation = cv2.INTER_NEAREST)
                except Exception:
                    error = True   
                finally:
                    error = False
        cv2.circle(frame, (centre[fA][0]*2, centre[fA][1]*2), rayon[fA]*2+1000, (255,255,255), 2000)
        frameCurrent = frame
        # Parameters
        (antX,antY) = (frame.shape[0]/2,frame.shape[1]/2) #initial position of the ant
        (antXBetter,antYBetter) = (frame.shape[0]/2,frame.shape[1]/2) #initial position of the ant
        maxPas = 15 #ant position stabilization

        frameNumberIn = 0
        frameNumberOut = 0
        frameNumber = 0

        isMoving = False
        relevantFrame = True
        frameNumberMoving = 0
        frameNumberStatic = 0
        circleColor = (0,0,255)
        centerColor = (0,255,0)
        perCentile = 128
        movementRelatifNorm = 0
        
        mX = []
        mY = []
        
        i=0

        sX= []
        sY= []
        isMovingBefore = False
        ConsStatic = [0]
        j=0

        stateListCircle = []
        stateListMovement = []
        frameNumberMoveIn = 0
        frameNumberStaticIn = 0
        frameNumberMoveOut = 0
        frameNumberStaticOut = 0

        def subImage(image,center,size):
            return image[int(max(0,center[0]-size//2)):int(min(center[0]+size//2,image.shape[0])), int(max(0,center[1]-size//2)):int(min(center[1]+size//2,image.shape[1]))]

        while True:
            error = True
            while(error) :
                try :
                    ret ,frame = cap.read()
                    frame = cv2.resize(frame, (1920,1080), interpolation = cv2.INTER_NEAREST)
                except Exception:
                    error = True
                finally:
                    error = False

            cv2.circle(frame, (centre[fA][0]*2, centre[fA][1]*2), rayon[fA]*2+1000, (255,255,255), 2000)
            frameNumber +=1 

            if (ret and frameNumber%fpsDivisionFactor==0):
                #print(frameNumber)
                frameCurrent, framePrevious = frame, frameCurrent

                # Threshold of the difference between two consecutive frames
                if frameNumber<150:
                    diff = cv2.subtract(framePrevious, frameCurrent)
                else:
                    diff = cv2.subtract(subImage(framePrevious, (antXBetter,antYBetter), 150), subImage(frameCurrent, (antXBetter,antYBetter), 150))
                grayDiff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
                _, thresh = cv2.threshold(grayDiff, 50, 255, cv2.THRESH_BINARY)

                #cv2.imshow('test', subImage(framePrevious, (antXBetter,antYBetter), 150))

                # Rough estimation of the ant's position with the threshold image
                changeCoords = np.argwhere(thresh == 255)
                
                coord = np.mean(changeCoords, 0)
                if frameNumber>=150:
                    coord[0],coord[1] = coord[0]+antXBetter-75, coord[1]+antYBetter-75

                if not (math.isnan(coord[0]) or math.isnan(coord[1])):
                    antX = antX + min(maxPas*fpsDivisionFactor,np.abs(coord[0]-antX)) * (coord[0]-antX)/max(0.0000001,np.abs(coord[0]-antX))
                    antY = antY + min(maxPas*fpsDivisionFactor,np.abs(coord[1]-antY)) * (coord[1]-antY)/max(0.0000001,np.abs(coord[1]-antY))

                # Improve the position estimation
                miniImage = cv2.cvtColor(subImage(frameCurrent, (antX,antY), 120), cv2.COLOR_BGR2GRAY)
                _, threshPart = cv2.threshold(miniImage, perCentile, 255, cv2.THRESH_BINARY )
                perCentile = round(np.percentile(miniImage, 4))
                changeCoords = np.mean(np.argwhere(threshPart == 0), 0)
                if not (math.isnan(changeCoords[0]) or math.isnan(changeCoords[1])):
                    movementRelatifNorm = np.sqrt((changeCoords[0] + antX - 60 - antXBetter)**2 + (changeCoords[1] + antY - 60 - antYBetter)**2)
                    (antXBetter,antYBetter) = (changeCoords[0] + antX - 60, changeCoords[1] + antY - 60)
                
                

                

                
                #cv2.imshow('Part', threshPart)
                if isMoving:
                    thresholdMovement = 0.8
                else:
                    thresholdMovement = 3

                if (movementRelatifNorm)<thresholdMovement*fpsDivisionFactor:
                    centerColor = (0,0,255)
                    isMovingBefore = isMoving
                    isMoving = False
                    if not isMovingBefore :
                        ConsStatic[j] = ConsStatic[j] +1
                    else :
                        ConsStatic.append(0)
                        sX.append(antXBetter)
                        sY.append(antYBetter)
                        j= j+1


                    
                    frameNumberStatic += 1
                    stateListMovement.append(2)
                    

                    
                else:
                    centerColor = (0,255,0)
                    isMoving = True
                    frameNumberMoving += 1
                    stateListMovement.append(1)
                    mX.append(antXBetter)
                    mY.append(antYBetter)



                # Check if the ant is in the circle
                if ((antY/2-centre[fA][0])**2+(antX/2-centre[fA][1])**2)<(rayon[fA]*ratio)**2:
                    frameNumberIn += 1
                    circleState = 'In'
                    circleColor = (255,0,0)
                    stateListCircle.append(1)
                    
                else:
                    frameNumberOut +=1
                    circleState = 'Out'
                    circleColor = (0,0,255)
                    stateListCircle.append(2)


                #Deal with the statistics
                if stateListCircle[-1]==1 and stateListMovement[-1]==1:
                    frameNumberMoveIn += 1
                        
                if stateListCircle[-1]==1 and stateListMovement[-1]==2:
                    frameNumberStaticIn +=1
                    
                if stateListCircle[-1]==2 and stateListMovement[-1]==1:
                    frameNumberMoveOut+=1

                if stateListCircle[-1]==2 and stateListMovement[-1]==2:
                    frameNumberStaticOut+=1
                
                # Resize the video
                frameResize = cv2.resize(frame, (int(frame.shape[1]/2),int(frame.shape[0]/2)), interpolation = cv2.INTER_NEAREST)

                #Draw the circles
                image_data = cv2.circle(frameResize, (int(antYBetter/2),int(antXBetter/2)), 40, circleColor, 3)
                cv2.circle(image_data, (int(antYBetter/2),int(antXBetter/2)), 1, centerColor, 3)

                #Write the data
                #Movement
                image_data = cv2.putText(image_data,'Is Moving: '+str(isMoving), (10,60), font, fontScale,fontColor,thickness,lineType)
                image_data = cv2.putText(image_data,'Movement: '+str(frameNumberMoving)+' ('+str(round(frameNumberMoving/frameNumber*100*fpsDivisionFactor,3))+'%)', (30,80), font, fontScale,fontColor,thickness,lineType)
                image_data = cv2.putText(image_data,'Static: '+str(frameNumber-frameNumberMoving*fpsDivisionFactor)+' ('+str(round((1-(frameNumberMoving/frameNumber*fpsDivisionFactor))*100,3))+'%)', (30,100), font, fontScale,fontColor,thickness,lineType)

                #Circle
                image_data = cv2.putText(image_data,'Circle: '+circleState, (10,140), font, fontScale,fontColor,thickness,lineType)
                image_data = cv2.putText(image_data,'In:  '+str(frameNumberIn)+' ('+str(round(frameNumberIn*fpsDivisionFactor/frameNumber*100,3))+'%)', (30,160), font, fontScale,fontColor,thickness,lineType)
                image_data = cv2.putText(image_data,'Out: '+str(frameNumberOut)+' ('+str(round(frameNumberOut*fpsDivisionFactor/frameNumber*100,3))+'%)', (30,180), font, fontScale,fontColor,thickness,lineType)
                #Movement + circle
                image_data = cv2.putText(image_data,'Located movement: ', (10,220), font, fontScale,fontColor,thickness,lineType)
                image_data = cv2.putText(image_data,'Ext Moving:  '+str(frameNumberMoveOut)+' ('+str(round(frameNumberMoveOut*fpsDivisionFactor/frameNumber*100,3))+'%)', (30,240), font, fontScale,fontColor,thickness,lineType)
                image_data = cv2.putText(image_data,'Int Moving: '+str(frameNumberMoveIn)+' ('+str(round(frameNumberMoveIn*fpsDivisionFactor/frameNumber*100,3))+'%)', (30,260), font, fontScale,fontColor,thickness,lineType)
                image_data = cv2.putText(image_data,'Int Static:  '+str(frameNumberStaticIn)+' ('+str(round(frameNumberStaticIn*fpsDivisionFactor/frameNumber*100,3))+'%)', (30,280), font, fontScale,fontColor,thickness,lineType)
                image_data = cv2.putText(image_data,'Ext Static: '+str(frameNumberStaticOut)+' ('+str(round(frameNumberStaticOut*fpsDivisionFactor/frameNumber*100,3))+'%)', (30,300), font, fontScale,fontColor,thickness,lineType)
                #Frames
                image_data = cv2.putText(image_data, 'Frame number: '+str(frameNumber), (10, 340), font, fontScale,fontColor,thickness,lineType)
                image_data = cv2.putText(image_data, 'Frame number (with FPS division): '+str(int(frameNumber/fpsDivisionFactor)), (10, 360), font, fontScale,fontColor,thickness,lineType)


                cv2.imshow('Measures', image_data)
                
                #Press 'Q' to exit the video
                k = cv2.waitKey(1)
                if k == ord('q'):
                    break

            if not ret:
                cv2.destroyWindow('Measures')
                finalResults = np.zeros([600,600,3],dtype=np.uint8)
                finalResults.fill(255)

                pourcentageMoveIn = frameNumberMoveIn*fpsDivisionFactor/frameNumber*100
                pourcentageMoveOut = frameNumberMoveOut*fpsDivisionFactor/frameNumber*100
                pourcentageStaticIn = frameNumberStaticIn*fpsDivisionFactor/frameNumber*100
                pourcentageStaticOut = frameNumberStaticOut*fpsDivisionFactor/frameNumber*100
                frameNumberMove = frameNumberMoveIn+frameNumberMoveOut
                pourcentageMove = pourcentageMoveOut+pourcentageMoveIn
                pourcentageStatic = pourcentageStaticIn+pourcentageStaticOut
                pourcentageIn = pourcentageMoveIn+pourcentageStaticIn
                pourcentageOut = pourcentageMoveOut+pourcentageStaticOut

                #Write the data
                #Movement
                finalResults = cv2.putText(finalResults,'Movement: '+str(frameNumberMoving)+' ('+str(round(frameNumberMoving/frameNumber*100*fpsDivisionFactor,3))+'%)', (30,80), font, fontScale,(0,0,0),thickness,lineType)
                finalResults = cv2.putText(finalResults,'Static: '+str(frameNumber-frameNumberMoving*fpsDivisionFactor)+' ('+str(round((1-(frameNumberMoving/frameNumber*fpsDivisionFactor))*100,3))+'%)', (30,100), font, fontScale,(0,0,0),thickness,lineType)

                #Circle
                finalResults = cv2.putText(finalResults,'In:  '+str(frameNumberIn)+' ('+str(round(frameNumberIn*fpsDivisionFactor/frameNumber*100,3))+'%)', (30,120), font, fontScale,(0,0,0),thickness,lineType)
                finalResults = cv2.putText(finalResults,'Out: '+str(frameNumberOut)+' ('+str(round(frameNumberOut*fpsDivisionFactor/frameNumber*100,3))+'%)', (30,140), font, fontScale,(0,0,0),thickness,lineType)
                #Movement + Circle
                finalResults = cv2.putText(finalResults,'Ext Moving:  '+str(frameNumberMoveOut)+' ('+str(round(frameNumberMoveOut*fpsDivisionFactor/frameNumber*100,3))+'%)', (30,170), font, fontScale,(0,0,0),thickness,lineType)
                finalResults = cv2.putText(finalResults,'Int Moving: '+str(frameNumberMoveIn)+' ('+str(round(frameNumberMoveIn*fpsDivisionFactor/frameNumber*100,3))+'%)', (30,190), font, fontScale,(0,0,0),thickness,lineType)
                finalResults = cv2.putText(finalResults,'Int Static:  '+str(frameNumberStaticIn)+' ('+str(round(frameNumberStaticIn*fpsDivisionFactor/frameNumber*100,3))+'%)', (30,210), font, fontScale,(0,0,0),thickness,lineType)
                finalResults = cv2.putText(finalResults,'Ext Static: '+str(frameNumberStaticOut)+' ('+str(round(frameNumberStaticOut*fpsDivisionFactor/frameNumber*100,3))+'%)', (30,230), font, fontScale,(0,0,0),thickness,lineType)
                
                
                
                #Frames
                finalResults = cv2.putText(finalResults, 'Frame number: '+str(frameNumber), (10, 260), font, fontScale,(0,0,0),thickness,lineType)
                finalResults = cv2.putText(finalResults, 'Frame number (with FPS division): '+str(int(frameNumber/fpsDivisionFactor)), (10, 280), font, fontScale,(0,0,0),thickness,lineType)


                #Title
                finalResults = cv2.putText(finalResults,experienceName, (10,30), font, fontScale,(0,0,0),thickness,lineType)

                #Adapt the lists of state
                adaptedStateListCircle = np.zeros(580, int)
                for k in range(20,580):
                    correspondingValue = stateListCircle[int((k-20)/560*len(stateListCircle))]
                    adaptedStateListCircle[k] = correspondingValue

                adaptedStateListMovement = np.zeros(580, int)
                for k in range(20,580):
                    correspondingValue = stateListMovement[int((k-20)/560*len(stateListMovement))]
                    adaptedStateListMovement[k] = correspondingValue
                
                #Draw diagams
                finalResults = cv2.putText(finalResults,'CIRCLE: Blue=In  Red=Out', (20,420), font, fontScale,(0,0,0),thickness,lineType)
                finalResults[430:480,np.argwhere(adaptedStateListCircle==1),:]=(255,0,0)
                finalResults[430:480,np.argwhere(adaptedStateListCircle==2),:]=(0,0,255)
                finalResults[430:480,np.argwhere(adaptedStateListCircle==-1),:]=(150,150,150)

                finalResults = cv2.putText(finalResults,'MOVEMENT: Green=Movement  Red=Static', (20,520), font, fontScale,(0,0,0),thickness,lineType)
                finalResults[530:580,np.argwhere(adaptedStateListMovement==1),:]=(0,255,0)
                finalResults[530:580,np.argwhere(adaptedStateListMovement==2),:]=(0,0,255)
                finalResults[530:580,np.argwhere(adaptedStateListMovement==-1),:]=(150,150,150)


                #cv2.imshow('Final Results', finalResults)

                cv2.imwrite(filename +' Resultats.png',finalResults)
                cv2.waitKey(0)
                break




        
        
        distmL=0
        distmLe=0
        distmLi=0

        i =0 
        fmX= moving_average(mX,6)
        fmY= moving_average(mY,6)

        while(i<len(fmX)-1):
            distmL = distmL + math.sqrt(((fmX[i]-fmX[i+1])**2)+((fmY[i]-fmY[i+1])**2))
            if(math.sqrt(((fmX[i]-centre[fA][0])**2)+((fmY[i]/4-centre[fA][1])**2))> circleRadius):
                distmLe = distmLe + math.sqrt(((fmX[i]-fmX[i+1])**2)+((fmY[i]-fmY[i+1])**2))
            else:
                distmLi = distmLi + math.sqrt(((fmX[i]-fmX[i+1])**2)+((fmY[i]-fmY[i+1])**2))
            i=i+1

        i =0

        fig, ax = plt.subplots()
        InCircle = plt.Circle((centre[fA][1]*2, centre[fA][0]*2),rayon[fA]*2*ratio,color = "black" ,fill=False,zorder=5)
        ax.add_patch(InCircle)
        while(i<len(sX)):
            if(ConsStatic[i]>50):
                antPause = plt.Circle((sX[i], sY[i]),50,color = "red" ,fill=True, zorder=10)
                ax.add_artist(antPause)
            elif ConsStatic[i] > 3 :
                antPause = plt.Circle((sX[i], sY[i]),ConsStatic[i],color = "red" ,fill=True, zorder=10)
                ax.add_artist(antPause)
            i = i+1
        distmLe = (distmLe/(2*rayon[fA]))*11.5
        distmLi = (distmLi/(2*rayon[fA]))*11.5
        distmL = (distmL/(2*rayon[fA]))*11.5
        ax.plot(fmX,fmY,zorder=0)
        ax.text(centre[fA][1]*3,centre[fA][0]*2 , "Dist Int : " + str(int(distmLi)) +"\nDist Ext : "+ str(int(distmLe)) +"\nDist : "+str(int(distmL)), fontsize=10)
        ax.set_aspect("equal")
        plt.savefig(filename +" Trajet.png")
        #plt.show()


        print("distnace parcourru en mouvement =" + str(int(distmL)))
        print("distance exterieur en mouvement =" + str(int(distmLe)))
        print("distnace interieur en mouvement =" + str(int(distmLi)))
        data = [os.path.basename(filename),pourcentageMoveOut,pourcentageMoveIn,pourcentageStaticIn,pourcentageStaticOut,distmLi,distmLe,distmL]
        writer.writerow(data)

        # cv2.destroyAllWindows()
        cap.release()
        fA = fA+1