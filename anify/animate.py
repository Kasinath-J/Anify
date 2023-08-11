import os
import cv2
import mediapipe as mp
import time
import numpy as np
from xml.dom.minidom import parse, Node
from helping_functions import *

def animate():

    mpDraw = mp.solutions.drawing_utils
    mpPose = mp.solutions.pose
    pose = mpPose.Pose() 
    mpFaceMesh = mp.solutions.face_mesh
    faceMesh = mpFaceMesh.FaceMesh()

    folder_directory=os.path.dirname(os.path.abspath(__file__))

    pTime = 0
    cap = cv2.VideoCapture(0)

    def set_id_attribute(parent, attribute_name="id"):
        if parent.nodeType == Node.ELEMENT_NODE:
            if parent.hasAttribute(attribute_name):
                parent.setIdAttribute(attribute_name)
        for child in parent.childNodes:
            set_id_attribute(child, attribute_name)

    org_image=os.path.join(folder_directory,'static/original.svg')
    final_image=os.path.join(folder_directory,'final.svg')
    document = parse(org_image)
    set_id_attribute(document)
    body = document.getElementById("body")
    left_upper = document.getElementById("left_upper")
    left_lower = document.getElementById("left_lower")
    right_upper = document.getElementById("right_upper")
    right_lower = document.getElementById("right_lower")
    left_thighs = document.getElementById("left_thighs")
    right_thighs = document.getElementById("right_thighs")
    left_thighs_trouser = document.getElementById("left_thighs_trouser")
    right_thighs_trouser = document.getElementById("right_thighs_trouser")

    total_head = document.getElementById("total_head")
    head = document.getElementById("head")

    neck = document.getElementById("neck")

    openmouth = document.getElementById("openmouth")
    closemouth = document.getElementById("closemouth")

    openlefteye = document.getElementById("openlefteye")
    openrighteye = document.getElementById("openrighteye")
    closelefteye = document.getElementById("closelefteye")
    closerighteye = document.getElementById("closerighteye")

    leftbrow = document.getElementById("leftbrow")
    rightbrow = document.getElementById("rightbrow")

    circle = document.getElementById("circle")

    screen_width = 1280
    screen_height = 720

    ##landmarks

    left_shoulder = [float(body.getAttribute("x")),float(body.getAttribute("y"))]
    body_width = float(body.getAttribute("width"))
    right_shoulder = ele_sum(left_shoulder,[body_width,0])
    body_height = float(body.getAttribute("height"))
    left_hip = ele_sum(left_shoulder,[0,body_height])
    right_hip = ele_sum(left_shoulder,[body_width,body_height])

    left_upper_angle = 30 #randomly assigning
    left_upper_length = float(left_upper.getAttribute("height"))
    left_elbow = [float(left_lower.getAttribute("x")),float(left_lower.getAttribute("y"))]
    left_lower_length = float(left_lower.getAttribute("height"))

    right_upper_angle = 300 #randomly assigning
    right_upper_length = float(right_upper.getAttribute("height"))
    right_elbow = [float(right_lower.getAttribute("x")),float(right_lower.getAttribute("y"))]
    right_lower_length = float(right_lower.getAttribute("height"))

    left_thighs_angle = 0 #randomly assigning
    left_thighs_length = float(left_thighs.getAttribute("height"))
    left_thighs_trouser_length = float(left_thighs_trouser.getAttribute("height"))
    left_thighs_trouser_width = float(left_thighs_trouser.getAttribute("width"))

    right_thighs_angle = 0 #randomly assigning
    right_thighs_length = float(right_thighs.getAttribute("height"))
    right_thighs_width = float(right_thighs.getAttribute("width"))
    right_thighs_trouser_length = float(right_thighs_trouser.getAttribute("height"))
    right_thighs_trouser_width = float(right_thighs_trouser.getAttribute("width"))

    head_xy = [float(head.getAttribute("x")),float(head.getAttribute("y"))]
    head_width = float(head.getAttribute("width"))
    head_height = float(head.getAttribute("height"))
    head_index = ele_sum(left_shoulder,[body_width/2,head_height])

    neck_height = float(neck.getAttribute("height"))
    bend=90


    ##Body Points

    #First frame

    prev_lm = None #landmark information
    prev_flm=None
    prev_dic = {} #Extra information
    prev_fdic = {}
    (cam_height,cam_width,_) = (None,None,None)

    done1,done2=0,0
    while not(done1>=1 and done2>=1):
        success,img = cap.read()
        (cam_height,cam_width,_) = img.shape # camera window height , camera window width
        imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        results = pose.process(imgRGB)
        f_results = faceMesh.process(imgRGB)

        if results.pose_landmarks:
            prev_lm = results.pose_landmarks.landmark
            prev_dic["body_width"] = distance(prev_lm[12],prev_lm[11]) ## body_width
            prev_dic["body_height"] = distance(prev_lm[24],prev_lm[11]) ## body_height
            prev_dic["left_upper_length"] = distance(prev_lm[12],prev_lm[14]) ##  left_shoulder_length
            prev_dic["right_upper_length"] = distance(prev_lm[11],prev_lm[13]) ##  right_shoulder_length
            prev_dic["left_thighs_length"] = distance(prev_lm[24],prev_lm[26]) ##  left_thighs_length
            prev_dic["right_thighs_length"] = distance(prev_lm[23],prev_lm[25]) ##  right_thighs_length


            reference = prev_lm[12]
            reference.x = (prev_lm[12].x+prev_lm[11].x)/2
            prev_fdic["neck_height"] = distance(reference,prev_lm[10]) ##  head_width
            done1+=1
        
        if f_results.multi_face_landmarks:
            faceLm = f_results.multi_face_landmarks[0]
            lm = faceLm.landmark
            prev_flm = lm
            prev_fdic["head_height"] = distance(prev_flm[10],prev_flm[152]) ##  head_length
            prev_fdic["head_width"] = distance(prev_flm[234],prev_flm[454]) ##  head_width
            
            done2+=1
        
    # First frame
    t=0
    while True:
        success,img = cap.read()
        imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        results = pose.process(imgRGB)
        cur_lm = None
        if results.pose_landmarks:
            mpDraw.draw_landmarks(img, results.pose_landmarks,mpPose.POSE_CONNECTIONS)
            cur_dic = {} #current extra information
            cur_lm = results.pose_landmarks.landmark

            if(cur_lm[12].visibility>0.3 and cur_lm[24].visibility>0.3 ): #left shoulder
                left_shoulder =  [left_shoulder[0]+(cur_lm[12].x-prev_lm[12].x)*screen_width,left_shoulder[1]+(cur_lm[12].y-prev_lm[12].y)*screen_height]
                body.setAttribute("x",str(left_shoulder[0]))
                body.setAttribute("y",str(left_shoulder[1]))


            if(cur_lm[11].visibility>0.3 and cur_lm[12].visibility>0.3): ##  body_width
                cur_dic["body_width"] = distance(cur_lm[12],cur_lm[11]) 
                if "body_width" in prev_dic:
                    body_width *= (1+ ((cur_dic["body_width"] - prev_dic["body_width"])/prev_dic["body_width"]))
                body.setAttribute("width",str(body_width))
                
                #right shoulder
                right_shoulder = ele_sum(left_shoulder,[body_width,0])
                body.setAttribute("x",str(left_shoulder[0]))
                body.setAttribute("y",str(left_shoulder[1]))


                

            if(cur_lm[24].visibility>0.3 and cur_lm[11].visibility>0.3): ##  body_height
                cur_dic["body_height"] = distance(cur_lm[24],cur_lm[11]) 
                if("body_height" in prev_dic):
                    body_height *= (1+ ((cur_dic["body_height"] - prev_dic["body_height"])/prev_dic["body_height"]))
                body.setAttribute("height",str(body_height))            
            
            if(cur_lm[12].visibility>0.3 and cur_lm[14].visibility>0.3): #left upper hand
                left_upper_angle = angle(cur_lm[24],cur_lm[12],cur_lm[14])
                left_upper.setAttribute('x',str(left_shoulder[0]))
                left_upper.setAttribute('y',str(left_shoulder[1]))
                left_upper.setAttribute("transform","rotate("+str(left_upper_angle)+','+str(left_shoulder[0])+','+str(left_shoulder[1]) +')')
                cur_dic["left_upper_length"] = distance(cur_lm[12],cur_lm[14]) ##  left_shoulder_length
                if "left_upper_length" in prev_dic:
                    left_upper_length *= (1+ ((cur_dic["left_upper_length"] - prev_dic["left_upper_length"])/prev_dic["left_upper_length"]))
                    left_upper_length = min(left_upper_length,body_height*0.8)
                left_upper.setAttribute("height",str(left_upper_length))
            else:
                left_upper.setAttribute('x',str(left_shoulder[0]))
                left_upper.setAttribute('y',str(left_shoulder[1]))
                left_upper.setAttribute("transform","rotate("+str(left_upper_angle)+','+str(left_shoulder[0])+','+str(left_shoulder[1]) +')')

            if(cur_lm[16].visibility>0.3 and cur_lm[14].visibility>0.3): #left lower hand
                m = math.tan((90 - left_upper_angle) * math.pi/180)
                left_elbow[0] = (-1*left_upper_length/np.sqrt(1+m*m)) + left_shoulder[0]
                left_elbow[1] = m * -1 * (left_elbow[0] - left_shoulder[0]) + left_shoulder[1]
                left_lower.setAttribute('x',str(left_elbow[0]))
                left_lower.setAttribute('y',str(left_elbow[1]))
                left_lower_angle = angle(cur_lm[12],cur_lm[14],cur_lm[16])
                left_lower.setAttribute("transform","rotate("+str(180 - ((360-left_lower_angle) - left_upper_angle))+','+str(left_elbow[0])+','+str(left_elbow[1]) +')')
                cur_dic["left_lower_length"] = distance(cur_lm[16],cur_lm[14]) ##  left_lower_length
                if "left_lower_length" in prev_dic:
                    left_lower_length *= (1+ ((cur_dic["left_lower_length"] - prev_dic["left_lower_length"])/prev_dic["left_lower_length"]))
                    left_lower_length = min(left_lower_length,body_height*0.7)
                left_lower.setAttribute("height",str(left_lower_length))
            else:
                m = math.tan((90 - left_upper_angle) * math.pi/180)
                left_elbow[0] = (-1*left_upper_length/np.sqrt(1+m*m)) + left_shoulder[0]
                left_elbow[1] = m * -1 * (left_elbow[0] - left_shoulder[0]) + left_shoulder[1]
                left_lower_angle = angle(cur_lm[12],cur_lm[14],cur_lm[16])
                left_lower.setAttribute('x',str(left_elbow[0]))
                left_lower.setAttribute('y',str(left_elbow[1]))
                left_lower.setAttribute("transform","rotate("+str(180 - ((360-left_lower_angle) - left_upper_angle))+','+str(left_elbow[0])+','+str(left_elbow[1]) +')')


            if(cur_lm[11].visibility>0.3 and cur_lm[13].visibility>0.3): # right upper hand
                right_upper_angle = angle(cur_lm[23],cur_lm[11],cur_lm[13])
                right_upper.setAttribute('x',str(right_shoulder[0]))
                right_upper.setAttribute('y',str(right_shoulder[1]))
                right_upper.setAttribute("transform","rotate("+str(right_upper_angle)+','+str(right_shoulder[0])+','+str(right_shoulder[1]) +')')
                cur_dic["right_upper_length"] = distance(cur_lm[11],cur_lm[13]) ##  right_shoulder_length
                if "right_upper_length" in prev_dic:
                    right_upper_length *= (1+ ((cur_dic["right_upper_length"] - prev_dic["right_upper_length"])/prev_dic["right_upper_length"]))
                    right_upper_length = min(right_upper_length,body_height*0.8)
                right_upper.setAttribute("height",str(right_upper_length))
            else:
                right_upper.setAttribute('x',str(right_shoulder[0]))
                right_upper.setAttribute('y',str(right_shoulder[1]))
                right_upper.setAttribute("transform","rotate("+str(right_upper_angle)+','+str(right_shoulder[0])+','+str(right_shoulder[1]) +')')
                
            
            if(cur_lm[15].visibility>0.3 and cur_lm[13].visibility>0.3): #right lower hand
                m = math.tan((90 + right_upper_angle) * math.pi/180)
                right_elbow[0] = (right_upper_length/np.sqrt(1+m*m)) + right_shoulder[0]
                right_elbow[1] = m * (right_elbow[0] - right_shoulder[0]) + right_shoulder[1]
                right_lower.setAttribute('x',str(right_elbow[0]))
                right_lower.setAttribute('y',str(right_elbow[1]))
                right_lower_angle = angle(cur_lm[11],cur_lm[13],cur_lm[15])
                right_lower.setAttribute("transform","rotate("+str(180 - ((360-right_lower_angle) - right_upper_angle))+','+str(right_elbow[0])+','+str(right_elbow[1]) +')')
                cur_dic["right_lower_length"] = distance(cur_lm[15],cur_lm[13]) ##  right_lower_length
                if "right_lower_length" in prev_dic:
                    right_lower_length *= (1+ ((cur_dic["right_lower_length"] - prev_dic["right_lower_length"])/prev_dic["right_lower_length"]))
                    right_lower_length = min(right_lower_length,body_height*0.7)
                right_lower.setAttribute("height",str(right_lower_length))
            else:
                m = math.tan((90 + right_upper_angle) * math.pi/180)
                right_elbow[0] = (right_upper_length/np.sqrt(1+m*m)) + right_shoulder[0]
                right_elbow[1] = m * (right_elbow[0] - right_shoulder[0]) + right_shoulder[1]
                right_lower_angle = angle(cur_lm[11],cur_lm[13],cur_lm[15])
                right_lower.setAttribute('x',str(right_elbow[0]))
                right_lower.setAttribute('y',str(right_elbow[1]))
                right_lower.setAttribute("transform","rotate("+str(180 - ((360-right_lower_angle) - right_upper_angle))+','+str(right_elbow[0])+','+str(right_elbow[1]) +')')

            if(cur_lm[24].visibility>0.3 and cur_lm[26].visibility>0.3): #left upper leg
                left_hip = ele_sum(left_shoulder,[0,body_height])
                reference_point = cur_lm[24]
                reference_point.y = 1 
                left_thighs_angle = angle(cur_lm[26],cur_lm[24],reference_point)-90
                left_thighs.setAttribute('x',str(left_hip[0]))
                left_thighs.setAttribute('y',str(left_hip[1]))
                # left_thighs.setAttribute("transform","rotate("+str(left_thighs_angle)+','+str(left_hip[0])+','+str(left_hip[1]) +')')
                left_thighs_trouser.setAttribute('x',str(left_hip[0]))
                left_thighs_trouser.setAttribute('y',str(left_hip[1]))
                # left_thighs_trouser.setAttribute('width',str(body_width/2))
                
                # left_thighs_trouser.setAttribute("transform","rotate("+str(left_thighs_angle)+','+str(left_hip[0])+','+str(left_hip[1]) +')')
                # cur_dic["left_thighs_length"] = distance(cur_lm[24],cur_lm[26]) ##  left_thighs_length
                # left_thighs_length *= (1+ ((cur_dic["left_thighs_length"] - prev_dic["left_thighs_length"])/prev_dic["left_thighs_length"]))
                # left_thighs.setAttribute("height",str(left_thighs_length))
            else:
                left_hip = ele_sum(left_shoulder,[0,body_height])
                left_thighs.setAttribute('x',str(left_hip[0]))
                left_thighs.setAttribute('y',str(left_hip[1]))
                left_thighs_trouser.setAttribute('x',str(left_hip[0]))
                left_thighs_trouser.setAttribute('y',str(left_hip[1]))
            
            
            if(cur_lm[23].visibility>0.3 and cur_lm[25].visibility>0.3): #right upper leg
                right_hip = ele_sum(left_shoulder,[body_width,body_height])
                reference_point = cur_lm[23]
                reference_point.y = 1 
                # cx,cy = int(cur_lm[23].x*cam_width),int(cur_lm[23].y*cam_height)
                # cv2.circle(img,(cx,cy),5,(255,0,0),cv2.FILLED)
                right_thighs_angle = angle(cur_lm[25],cur_lm[23],reference_point)-90
                print(right_thighs_angle)
                right_thighs.setAttribute('x',str(right_hip[0]-right_thighs_width))
                right_thighs.setAttribute('y',str(right_hip[1]))
                # right_thighs.setAttribute("transform","rotate("+str(right_thighs_angle)+','+str(right_hip[0])+','+str(right_hip[1]) +')')
                right_thighs_trouser.setAttribute('x',str(right_hip[0]-right_thighs_trouser_width))
                right_thighs_trouser.setAttribute('y',str(right_hip[1]))
                # right_thighs_trouser.setAttribute("transform","rotate("+str(left_thighs_angle)+','+str(left_hip[0])+','+str(left_hip[1]) +')')
            else:
                right_hip = ele_sum(left_shoulder,[body_width,body_height])
                right_thighs.setAttribute('x',str(right_hip[0]-right_thighs_width))
                right_thighs.setAttribute('y',str(right_hip[1]))
                right_thighs_trouser.setAttribute('x',str(right_hip[0]-right_thighs_trouser_width))
                right_thighs_trouser.setAttribute('y',str(right_hip[1]))
            

            prev_lm = cur_lm
            prev_dic = cur_dic
            

        f_results = faceMesh.process(imgRGB)
        if f_results.multi_face_landmarks:
            faceLm = f_results.multi_face_landmarks[0]
            mpDraw.draw_landmarks(img, faceLm,mpFaceMesh.FACEMESH_CONTOURS)
            cur_flm = faceLm.landmark
            cur_fdic = {}

            head_index = ele_sum(left_shoulder,[body_width/2,0])        
            head_xy[0] = left_shoulder[0]+body_width/2-head_width/2
            head_xy[1] = left_shoulder[1]-head_height-20
            
            head.setAttribute("x",str(head_xy[0]))
            head.setAttribute("y",str(head_xy[1]))
            if(cur_lm[0].visibility>0.3 and cur_lm[12].visibility>0.3 and cur_lm[11].visibility>0.3 ):
                bend = angle2([int(cur_lm[0].x*cam_width),int(cur_lm[0].y*cam_height)],[int((cur_lm[12].x+cur_lm[11].x)/2*cam_width),int(cur_lm[12].y*cam_height)],[int(cur_lm[11].x*cam_width),int(cur_lm[11].y*cam_height)])
            total_head.setAttribute("transform","rotate("+str(bend-90)+','+str(head_index[0])+','+str(head_index[1]) +')')
            cur_fdic["head_height"] = distance(prev_flm[10],prev_flm[152]) ##  head_length
            if("head_height" in prev_fdic):
                head_height *= (1+ ((cur_fdic["head_height"] - prev_fdic["head_height"])/prev_fdic["head_height"]))
                head.setAttribute("height",str(head_height))

            cur_fdic["head_width"] = distance(prev_flm[234],prev_flm[454]) ##  head_length
            if("head_width" in prev_fdic):
                head_width *= (1+ ((cur_fdic["head_width"] - prev_fdic["head_width"])/prev_fdic["head_width"]))
                head.setAttribute("width",str(head_width))

            #neck
            neck.setAttribute("x",str(head_index[0]))
            neck.setAttribute("y",str(head_index[1]-neck_height))
            
            # reference = prev_lm[12]
            # reference.x = (prev_lm[12].x+prev_lm[11].x)/2
            # cur_fdic["neck_height"] = distance(reference,prev_lm[10])
            # if "neck_height" in cur_fdic:
            #     neck_height *= (1+ ((cur_fdic["neck_height"] - prev_fdic["neck_height"])/prev_fdic["neck_height"]))
            # neck.setAttribute("height",str(neck_height))
            
            #mouth
            actual_radius = distance(cur_flm[13],cur_flm[14])
            # actual_radius = distance(cur_flm[11],cur_flm[16])
            mid = [head_xy[0]+head_width*0.5,head_xy[1]+head_height*0.8]
            if(actual_radius/cur_fdic["head_height"]<0.025):
                closemouth.setAttribute("x1",str(mid[0]-0.06*head_width))
                closemouth.setAttribute("x2",str(mid[0]+0.06*head_width))
                closemouth.setAttribute("y1",str(mid[1]))
                closemouth.setAttribute("y2",str(mid[1]))
                openmouth.setAttribute("opacity",str(0))
                closemouth.setAttribute("opacity",str(1))
            else:
                openmouth.setAttribute("cx",str(mid[0]))
                openmouth.setAttribute("cy",str(mid[1]))
                openmouth.setAttribute("opacity",str(1))
                closemouth.setAttribute("opacity",str(0))

                proportion = (actual_radius/cur_fdic["head_height"])/0.1225
                max_radius = 0.03*cur_fdic["head_height"]*screen_height
                mouthradius = max( 0.6*max_radius,proportion*max_radius)
                    #0.0625=>max ratio
                openmouth.setAttribute("r",str(mouthradius))

            #eye
            mid = [head_xy[0]+head_width*0.5,head_xy[1]+head_height*0.45]
            righteyeopendist = distance(cur_flm[386],cur_flm[374])
            lefteyeopendist = distance(cur_flm[159],cur_flm[145])
            # print(lefteyeopendist)
            #296,66
            # cx,cy = int(cur_flm[66].x*cam_width),int(cur_flm[66].y*cam_height)
            # cv2.circle(img,(cx,cy),5,(255,0,0),cv2.FILLED)

            # cx,cy = int(cur_flm[374].x*cam_width),int(cur_flm[374].y*cam_height)
            # cv2.circle(img,(cx,cy),5,(255,0,0),cv2.FILLED)
            # print(lefteyeopendist/cur_fdic["head_height"])
            if(lefteyeopendist/cur_fdic["head_height"]>0.035):
                openlefteye.setAttribute("cx",str(mid[0]-(0.12*head_width)))
                openlefteye.setAttribute("cy",str(mid[1]))
                openlefteye.setAttribute("opacity","1")
                closelefteye.setAttribute("opacity","0")
                
                proportion = (lefteyeopendist/cur_fdic["head_height"])/0.054
                max_radius = 0.04*cur_fdic["head_height"]*screen_height
                lefteyeradius = max( 0.7*max_radius,proportion*max_radius)
                openlefteye.setAttribute("r",str(lefteyeradius))

            else:
                openlefteye.setAttribute("opacity","0")
                closelefteye.setAttribute("opacity","1")
                closelefteye.setAttribute("x1",str(mid[0]-(0.1-0.06)*head_width))
                closelefteye.setAttribute("x2",str(mid[0]-(0.1+0.06)*head_width))
                closelefteye.setAttribute("y1",str(mid[1]))
                closelefteye.setAttribute("y2",str(mid[1]))

            if(righteyeopendist/cur_fdic["head_height"]>0.035):
                openrighteye.setAttribute("cx",str(mid[0]+(0.12*head_width)))
                openrighteye.setAttribute("cy",str(mid[1]))
                openrighteye.setAttribute("opacity","1")
                closerighteye.setAttribute("opacity","0")

                proportion = (righteyeopendist/cur_fdic["head_height"])/0.054
                max_radius = 0.04*cur_fdic["head_height"]*screen_height
                righteyeradius = max( 0.7*max_radius,proportion*max_radius)
                openrighteye.setAttribute("r",str(righteyeradius))
            else:
                openrighteye.setAttribute("opacity","0")
                closerighteye.setAttribute("opacity","1")
                closelefteye.setAttribute("x1",str(mid[0]+(0.08-0.06)*head_width))
                closelefteye.setAttribute("x2",str(mid[0]+(0.08-0.06)*head_width))
                closerighteye.setAttribute("y1",str(mid[1]))
                closerighteye.setAttribute("y2",str(mid[1]))

    
            leftbrow.setAttribute("x",str(head_xy[0]+0.23*head_width))
            leftbrow.setAttribute("y",str(head_xy[1]+0.05*head_height))
            leftbrow.setAttribute("height",str(0.2*head_height))
            leftbrow.setAttribute("width",str(0.3*head_width))

            rightbrow.setAttribute("x",str(head_xy[0]+0.52*head_width))
            rightbrow.setAttribute("y",str(head_xy[1]+0.05*head_height))
            rightbrow.setAttribute("height",str(0.2*head_height))
            rightbrow.setAttribute("width",str(0.3*head_width))

            prev_flm = cur_flm
            prev_fdic = cur_fdic


        document.writexml(open(final_image,'w'))

        cTime = time.time()
        fps=  1/(cTime-pTime)
        pTime = cTime
        cv2.putText(img,str(int(fps)),(70,50),cv2.FONT_HERSHEY_PLAIN,3,(255,0,0),3)
        cv2.imshow("cam",img)
        cv2.waitKey(10)


# animate()

