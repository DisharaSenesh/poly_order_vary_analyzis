import numpy as np
import cv2

class ls_triangulation:
    def __init__(self, intrinsics, hand_eye, distortion):
        self.intrinsics = intrinsics
        self.hand_eye = hand_eye
        self.distortion = distortion

    # triangulation
    def triangulate(self, dataset_rows):

        R_cam2tool, t_cam2tool = self.get_camera_to_tool_transform()
        K = self.intrinsics_matrix(self.intrinsics)

        centers = []
        directions = []

        for row in dataset_rows:

            X, Y, Z, A, B, C, u, v, time = row
            print(f"time = {time}")

            t_tool2base = np.array([X, Y, Z], dtype=np.float64)
            R_tool2base = self.kuka_rotation_matrix(A, B, C)


            # ---- Camera pose in base ----
            R_cam2base = R_tool2base @ R_cam2tool
            t_cam2base = R_tool2base @ t_cam2tool + t_tool2base

            # ---- Undistort pixel ----
            pixel = self.undistort_pixel_coordinates(u, v, K, self.distortion)
            d_cam = np.array(pixel, dtype=np.float64)

            # ---- Transform ray to base ----
            d_base = self.direction_vector(d_cam, R_cam2base)

            centers.append(t_cam2base)
            directions.append(d_base)

        # ---- Triangulate using all rays ----
        X_est = self.triangulate_least_squares(None, centers, directions)

        return X_est

    # make rotation matrix
    def kuka_rotation_matrix(self, A: float, B: float, C: float):
        z = np.deg2rad(A)
        y = np.deg2rad(B)
        x = np.deg2rad(C)

        rx =  np.array([
            [1, 0, 0],
            [0, np.cos(x), -np.sin(x)],
            [0, np.sin(x), np.cos(x)]
        ], dtype=np.float64)

        ry = np.array([
            [np.cos(y), 0, np.sin(y)],
            [0, 1, 0],
            [-np.sin(y), 0, np.cos(y)]
        ], dtype=np.float64)

        rz = np.array([
            [np.cos(z), -np.sin(z), 0],
            [np.sin(z),  np.cos(z), 0],
            [0, 0, 1]
        ], dtype=np.float64)

        return rz @ ry @ rx

    # hand eye calibration transform
    def get_camera_to_tool_transform(self):
        he = self.hand_eye
        rx, ry, rz = he[0], he[1], he[2]
        tx, ty, tz = he[3], he[4], he[5]

        rx = np.deg2rad(rx)
        ry = np.deg2rad(ry)
        rz = np.deg2rad(rz)

        # Rotation around X axis
        rx_ = np.array([
            [1, 0, 0],
            [0, np.cos(rx), -np.sin(rx)],
            [0, np.sin(rx), np.cos(rx)]
        ], dtype=np.float64)

        # Rotation around Y axis
        ry_ = np.array([
            [np.cos(ry), 0, np.sin(ry)],
            [0, 1, 0],
            [-np.sin(ry), 0, np.cos(ry)]
        ], dtype=np.float64)

        # Rotation around Z axis
        rz_ = np.array([
            [np.cos(rz), -np.sin(rz), 0],
            [np.sin(rz), np.cos(rz), 0],
            [0, 0, 1]
        ], dtype=np.float64)

        # Compute rotation matrix (Rx @ Ry @ Rz)
        R = rx_ @ ry_ @ rz_
        
        # Translation vector
        t = np.array([tx, ty, tz], dtype=np.float64)
        
        return R, t

    # intrinsics matrix
    def intrinsics_matrix(self, intrinsic_param):
        fx = intrinsic_param[0]
        fy = intrinsic_param[1]
        cx = intrinsic_param[2]
        cy = intrinsic_param[3]

        K = np.array([[fx, 0, cx],
                    [0, fy, cy],
                    [0, 0, 1]], dtype=np.float64)
        return K

    # undistort pixel coordinates
    def undistort_pixel_coordinates(self, pixel_x, pixel_y, K, dist_coeffs):
        x,y,z = None, None, None
        if dist_coeffs is not None:
            src = np.array([[[pixel_x, pixel_y]]], dtype=np.float64)
            camera_matrix = K

            dst = cv2.undistortPoints(src, camera_matrix, dist_coeffs)
            
            x = dst[0, 0, 0]
            y = dst[0, 0, 1]
            z = 1.0

        if x is None or y is None or z is None:
            print("Warning: Undistortion failed, using original pixel coordinates.")
            x = (pixel_x - K[0,2]) / K[0,0]
            y = (pixel_y - K[1,2]) / K[1,1]
            z = 1.0 
            return [x,y,z]
        else:
            return [x,y,z]

    # direction vector from camera to world
    def direction_vector(self, pixel, R_cam2base):
        x = pixel[0]
        y = pixel[1]
        z = 1.0  
        
        d_cam = np.array([x, y, z], dtype=np.float64)
        
        # Transform ray from camera to base frame
        d_base = R_cam2base @ d_cam
        
        return d_base / np.linalg.norm(d_base)

    # Skew-symmetric matrix for cross product
    def skew(self, v):
        vx, vy, vz = v
        return np.array([
            [0, -vz, vy],
            [vz, 0, -vx],
            [-vy, vx, 0]
        ], dtype=np.float64)

    # Least squares triangulation
    def triangulate_least_squares(self, _, centers, directions):
        A_blocks = []
        b_blocks = []

        for C, d in zip(centers, directions):
            Dx = self.skew(d)
            A_blocks.append(Dx)
            b_blocks.append(Dx @ C)

        A = np.vstack(A_blocks)
        b = np.hstack(b_blocks)

        X, *_ = np.linalg.lstsq(A, b, rcond=None)
        return X
        


