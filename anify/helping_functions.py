import numpy as np
import math
import re

def distance(p1,p2):
    return np.sqrt(np.square(p1.x-p2.x) + np.square(p1.y-p2.y))

def ele_sum(l1,l2):
    return [l1[i]+l2[i] for i in range(len(l1))]

def angle(a, b, c):
    """Counterclockwise angle in degrees by turning from a to c around b
        Returns a float between 0.0 and 360.0"""
    ang = math.degrees(
        math.atan2(c.y-b.y, c.x-b.x) - math.atan2(a.y-b.y, a.x-b.x))
    return ang + 360 if ang < 0 else ang

def angle2(a, b, c):
    """Counterclockwise angle in degrees by turning from a to c around b
        Returns a float between 0.0 and 360.0"""
    ang = math.degrees(
        math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0]))
    return ang + 360 if ang < 0 else ang

def find_angle(x):
    x = x[7:]
    ans = re.split(",", x, 1)
    return(float(ans[0]))

def find_coor(left_upper_angle,left_upper_length,left_shoulder):
    left_elbow = [0,0]
    m = math.tan((90 + left_upper_angle) * math.pi/180)
    left_elbow[0] = (left_upper_length/np.sqrt(1+m*m)) + left_shoulder[0]
    left_elbow[1] = -1*m * (left_elbow[0] - left_shoulder[0]) + left_shoulder[1]
    print(left_elbow)
