import os
os.environ['PROJ_LIB'] = 'C:\\Users\\proko\\miniconda3\\pkgs\\proj-6.2.1-h9f7ef89_0\\Library\\share\\proj'
os.environ['GDAL_DATA'] = 'C:\\Users\\proko\\miniconda3\\pkgs\\proj-6.2.1-h9f7ef89_0\\Library\\share'

import matplotlib.pyplot as plt
from osgeo import gdal
import pandas as pd

tif = gdal.Open("CZE_wind-speed_100m.tif")

#some basic raster info
print(tif.RasterXSize, tif.RasterYSize)
print(tif.GetProjection())
print(tif.GetGeoTransform())

"""
GT(0) x-coordinate of the upper-left corner of the upper-left pixel.
GT(1) w-e pixel resolution / pixel width.
GT(2) row rotation (typically zero).
GT(3) y-coordinate of the upper-left corner of the upper-left pixel.
GT(4) column rotation (typically zero).
GT(5) n-s pixel resolution / pixel height (negative value for a north-up image).
"""

print(tif.RasterCount)

# print band info
band1 = tif.GetRasterBand(1)

min = band1.GetMinimum()
max = band1.GetMaximum()
if not min or not max:
    (min,max) = band1.ComputeRasterMinMax(True)
print("Min={:.3f}, Max={:.3f}".format(min,max))

print("No Data:", band1.GetNoDataValue())
print("Minimum:", band1.GetMinimum())
print("Maximum:", band1.GetMaximum())
print("Type:", band1.GetUnitType())





tifArray = tif.ReadAsArray()
imgplot2 = plt.imshow(tifArray)
plt.show()





