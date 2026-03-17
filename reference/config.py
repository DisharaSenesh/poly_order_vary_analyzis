import numpy as np

# Camera parameters
# intrinsics = [800, 800, 320, 240] # fx, fy, cx, cy
# distortion = np.array([0,0,0,0,0]) # k1, k2, p1, p2, k3
# hand_eye = np.array([0, 0, 0, 0, 0, 0]) # rx, ry, rz, tx, ty, tz

intrinsics = [1486.4123717271, 1488.6919876408, 639.5371967339, 431.7580771817]
hand_eye = np.array([1.1239692871, -3.7506525776, -87.6023477856, 
                     13.1335163602, -8.5332482353, -89.7669585601])


# new calibrate
# intrinsics = [1417.342538130495, 1433.241136217907, 606.6407755122701, 510.5798901528584]
# hand_eye = np.array([3.51, 1.14, -88.17, 42.395, 6.221, -58.046])

distortion = np.array([ 0.047240116965482692, 0.76857331815544261,
       0.0005221055479888955, 0.0041421931791450536, -2.9566144231226019 ])

# distortion = np.array([ 0.0505, 0.6903, 0.00241, 0.00056, -2.4107])

camera_index = 2

#[988.721558, -129.854706, 606.509277, -138.245071, 89.904, -138.246216]

# # robot starting point
# robot_start = [525, 0, 890, 0, 90, 0]

# # robot ending point
# robot_end = [525, 0, 890, 0, 90, 0]

# # robot moving speed
# robot_speed = 10
# # robot speed increment
# robot_speed_increment = 0.1