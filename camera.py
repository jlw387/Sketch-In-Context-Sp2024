"""
Camera class used for the sphere-tracing renders.
"""

import math
import numpy as np
from vector_math import *

class Camera:
    """
        A camera for the scene, with some basic ray-tracing calls. Assumes a left-handed coordinate system 
        with x being to the right, y being upwards, and z being depth from the camera. 
    """

    def __init__(self, width = 1024, height = 1024, fov = 30.0, pos = np.array([10,0,0]), forward = None, up = np.array([0,0,1])):
        """
            Creates a new camera instance.

            Parameters:
                width - width of the camera in pixels (default = 1024) \\
                height - height of the camera in pixels (default = 1024) \\
                fov - field of view of the camera in degrees in the smaller dimension (default = 30.0) \\
                pos - position of the camera in the 3D scene (default = np.array([10,0,0])) \\
                forward - forward direction of the camera in the 3D scene (defaults to -pos normazlied) \\
                up - true upward direction of the camera in the 3D scene (default = np.array([0,0,1])) 
        """

        # Base Parameters
        self.width = width
        self.height = height
        self.fov = fov
        self.pos = pos
        if type(forward) == type(None):
            forward = -pos
        self.forward = normalize(forward)
        self.true_up = normalize(up)

        # Computed Parameters
        self.aspect_ratio = width / height
        self.right = normalize(np.cross(forward, up))
        print("Right:", self.right)
        print("Forward:", self.forward)
        self.ortho_up = normalize(np.cross(self.right, forward))
        print("Ortho Up:", self.ortho_up)

    def update_screen_parameters(self, width : int=None, height : int=None, fov : float=None):
        """
            Changes the width, height, or FOV of the camera depending on the parameters passed. USE THIS METHOD 
            TO UPDATE THESE PARAMETERS, OTHERWISE ASPECT RATIO WILL NOT BE ACCURATE!

            Parameters:
                width - new width of the camera in pixels (default=None) \\
                height (int) - new height of the camera in pixels \\
                fov (float) - new fov of the camera in degrees
        """
        if width != None:
            self.width = width
        if height != None:
            self.height = height
        if fov != None:
            self.fov = fov

        self.aspect_ratio = self.width / self.height

    def update_scene_parameters(self, pos=None, forward=None, up=None):
        """
            Changes the position, forward direction, or true upward direction of the camera depending on the parameters passed. USE THIS METHOD 
            TO UPDATE THESE PARAMETERS, OTHERWISE COMPUTED CAMERA SCENE PARAMETERS WILL NOT BE ACCURATE!

            Parameters:
                pos - new position of the camera as a 3-element 1D numpy array (default=None) \\
                forward - new forward direction of the camera as a 3-element 1D numpy array \\
                up - new true upward direction of the camera as a 3-element 1D numpy array
        """
        if pos != None:
            self.pos = pos
        if forward != None:
            self.forward = forward
        if up != None:
            self.true_up = up

        self.right = normalize(np.cross(forward, up))
        self.ortho_up = normalize(np.cross(self.right, forward))


    def get_ray_at_coordinate(self, row : int, col : int):
        """
            Computes the parameters of a ray for the camera pixel at the specified coordinates.

            Returns:
                origin - the origin of the ray \\
                direction - the normalized direction of the ray

            Parameters:
                row - the index of the row of the pixel \\
                col - the index of the column of the pixel

            Raises:
                IndexError - raised if the specified row or column index lies outside the range of possible indices for the given camera's width and height
        """
        if row >= self.height or row < 0:
            raise IndexError("Row Index is outside allowable range: expected 0 to " + str(self.height - 1) + ", got " + str(row))
        if col >= self.width or col < 0:
            raise IndexError("Column Index is outside allowable range: expected 0 to " + str(self.width - 1) + ", got " + str(col))

        if self.width < self.height:
            w_dst = math.tan(self.fov * math.pi / 360)
            h_dst = w_dst / self.aspect_ratio
        else:
            h_dst = math.tan(self.fov * math.pi / 360)
            w_dst = h_dst * self.aspect_ratio
        
        dx = ((col + 0.5) * 2 / self.width - 1) * w_dst
        dy = (1 - (row + 0.5) * 2 / self.height) * h_dst
        
        direction = normalize(dx * self.right + dy * self.ortho_up + self.forward)

        return self.pos, direction

    def get_all_ray_directions(self):
        """
            Computes the directions of all rays for the camera pixels.

            Returns:
                directions - the normalized directions of all the rays as a 3-D NDArray
        """
        directions = np.zeros((self.height, self.width,3))

        if self.width < self.height:
            w_dst = math.tan(self.fov * math.pi / 360)
            h_dst = w_dst / self.aspect_ratio
        else:
            h_dst = math.tan(self.fov * math.pi / 360)
            w_dst = h_dst * self.aspect_ratio

        dx = 2 * w_dst/ self.width
        dy = 2 * h_dst / self.height

        start_pos = (dx/2 - w_dst) * self.right + (h_dst - dy/2) * self.ortho_up + self.forward
        for i in range(self.height):
            for j in range(self.width):
                directions[i,j,:] = start_pos + (j * dx) * self.right - (i * dy) * self.ortho_up
                directions[i,j,:] = directions[i,j,:] / np.linalg.norm(directions[i,j,:])

        return directions


        
        
