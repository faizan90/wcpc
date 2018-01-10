'''
Created on Jan 4, 2018

@author: Faizan-Uni
'''
import pyproj


def change_pt_crs(x, y, in_epsg, out_epsg):
    """
    Purpose:
        To return the coordinates of given points in a different coordinate system.

    Description of arguments:
        x (int or float, single or list): The horizontal position of the input point
        y (int or float, single or list): The vertical position of the input point
        Note: In case of x and y in list form, the output is also in a list form.
        in_epsg (string or int): The EPSG code of the input coordinate system
        out_epsg (string or int): The EPSG code of the output coordinate system
    """
    in_crs = pyproj.Proj("+init=EPSG:" + str(in_epsg))
    out_crs = pyproj.Proj("+init=EPSG:" + str(out_epsg))
    return pyproj.transform(in_crs, out_crs, float(x), float(y))
