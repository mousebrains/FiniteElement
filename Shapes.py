#! /usr/bin/env python3
#
# Base class for all geometry shapes in a finite element analysis
#
# Jan-2020, Pat Welch, pat@mousebrains.com

import numpy as np
from scipy.spatial.transform import Rotation as R

class Shapes(list):
    def __init__(self, info:dict, msg:str) -> None:
        list.__init__(self)
        for key in info:
            item = info[key]
            shape = item["shape"] if "shape" in item else None
            if shape is None: raise Exception("No shape specified for {}, {}".format(key, msg))
            if shape == "cylinder":
                self.append(Cylinder(key, item, msg))
            else:
                raise Exception("Unrecognized shape, {}, in {}, {}".format(shape, key, msg))

    def dryVolume(self):
        volume = 0
        for item in self: volume += item.qDry * item.volume
        return volume

    def wetVolume(self):
        volume = 0
        for item in self: volume += (not item.qDry) * item.volume
        return volume


class Base:
    """ API specification """
    def __init__(self, name:str) -> None:
        self.name = name # Name of the shape

    def grid(self, rotation:R=None, offset:np.array=None) -> np.array:
        """ Return the grid points, possibly rotated and offset """
        position = self.position
        if rotation is not None:
            # Rotate from body to grid
            position = rotation.apply(position, inverse=True)
        if offset is not None: position = np.add(position, offset)
        return position

    def integrate(self, force:np.array, rotation:R=None) -> np.array:
        """ Return surface integral over force """
        surface = self.surface
        if rotation is not None:
            surface = rotation.apply(surface, inverse=True)
        # integral of dot product
        return \
                np.sum(surface[:,0] * force[:,0]) + \
                np.sum(surface[:,1] * force[:,1]) + \
                np.sum(surface[:,2] * force[:,2])

    @staticmethod
    def mkRotation(angles:list) -> R:
        return R.from_euler("ZYX", angles, degrees=True)
    @staticmethod
    def strRotation(r:R) -> list:
        return r.as_euler("ZYX", degrees=True)

    # Plotting methods for a shape

    @staticmethod
    def plotShow(ax=None) -> None:
        if ax is not None:
            import matplotlib.pyplot as plt
            plt.show()

    def plotInit(self, ax=None):
        import matplotlib.pyplot as plt
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
        return ax

    def plotScatter(self, ax, pos:np.array, sz=5):
        import matplotlib.pyplot as plt
        ax = self.plotInit(ax)
        ax.scatter(pos[:,0], pos[:,1], pos[:,2], s=sz)
        return ax

    def plot(self, ax=None):
        return self.plotScatter(ax, self.position)

class Cylinder(Base):
    """ Cylindrical part of a cylinder with outward pointing normals to surface """
    def __init__(self, name:str, info:dict, msg:str) -> None:
        Base.__init__(self, name + "::cylinder")
        for key in ["length", "radius", "offset", "rotation", "nRadial", "nLength"]:
            if key not in info:
                raise Exception("Field {} not in {}, {}".format(key, name, msg))

        self.__info = info

        radius = info["radius"] # Radius of the cylinder
        length = info["length"] # Length of cylinder

        self.__radius = radius

        self.qDry = info["qDry"] if "qDry" in info else False
        self.volume = np.pi * radius * radius * length # Volume of cylinder

        dTheta = (2 * np.pi) / info["nRadial"] # Step size in theta
        dZ = length / info["nLength"]  # Step size along axis

        self.__area = dZ * self.__radius * dTheta # Area of each element

        # Angle to center of each segment
        theta = np.arange(dTheta/2, 2 * np.pi, dTheta) # (0, 2pi)

        # z of center of each segment
        zMax = info["length"] / 2
        z = np.arange(-zMax + dZ/2, zMax, dZ) # (-zMax, zMax)

        (gTheta, gZ) = np.meshgrid(theta, z) # Grid of all possible theta/z pairs
        gTheta = gTheta.flatten() # Change from 2D to 1D
        gZ = gZ.flatten() # Change from 2D to 1D

        nx = np.cos(gTheta) # x component of outward pointing unit normal vector
        ny = np.sin(gTheta) # y component of outward pointing unit normal vector

        unitNorm = np.column_stack((nx, ny, np.zeros(nx.shape))) # Outward pointing unit normal
        position = np.column_stack((radius * nx, radius * ny, gZ)) # center of each element

        # Rotate the cylinder

        self.__rotation = self.mkRotation(info["rotation"])
        unitNorm = self.__rotation.apply(unitNorm)  # Rotate outward pointing normals
        position  = self.__rotation.apply(position) # Rotate positions of centers

        # Offset the cylinder, the unit norm does not change
        self.__offset = info["offset"]
        self.position = np.add(position, self.__offset) # Position of each surface element

        self.surface = self.__area * unitNorm # outward pointing surface vector


    def __repr__(self) -> str:
        msg = self.name + \
                " radius {} area {:.4g} n {} qDry {} volume {:.4g}".format(
                        self.__radius, self.__area, self.position.shape[0],
                        self.qDry, self.volume)
        msg+= "\n" + self.name + \
                " rotation {}".format(self.strRotation(self.__rotation))
        msg+= "\n" + self.name + " offset {}".format(self.__offset)
        msg+= "\n" + self.name + " area={}".format(self.integrate(self.surface / self.__area))
        return msg


if __name__ == "__main__":
    import yaml
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--glider", type=str, metavar="foo.yml", required=True,
            help="geometry configuration YAML file")
    parser.add_argument("--plot", action="store_true", help="Plot shapes")
    parser.add_argument("--pitch", type=float, metavar="degrees", default=0,
            help="Pitch angle of body")
    parser.add_argument("--heading", type=float, metavar="degrees", default=0,
            help="heading angle of body")
    parser.add_argument("--roll", type=float, metavar="degrees", default=0,
            help="Roll angle of body")
    args = parser.parse_args()

    print(args)

    with open(args.glider, "r") as fp: info = yaml.safe_load(fp)
    if "geometry" not in info: raise Exception("geometry not in " + args.glider)
    shapes = Shapes(info["geometry"], "in " + args.glider)

    ax = None
    r = Base.mkRotation([args.heading, args.pitch, args.roll])
    for shp in shapes:
        print(shp)
        grid = shp.grid(r)
        print("Grid")
        print(grid)
        if args.plot:
            ax = shp.plot(ax)
            ax = shp.plotScatter(ax, grid)

    Base.plotShow(ax)
