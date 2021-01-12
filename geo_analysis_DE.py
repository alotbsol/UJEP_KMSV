import os
os.environ['PROJ_LIB'] = 'C:\\Users\\proko\\miniconda3\\pkgs\\proj-6.2.1-h9f7ef89_0\\Library\\share\\proj'
os.environ['GDAL_DATA'] = 'C:\\Users\\proko\\miniconda3\\pkgs\\proj-6.2.1-h9f7ef89_0\\Library\\share'

import matplotlib.pyplot as plt
from osgeo import gdal
import pandas as pd

tif = gdal.Open("DEU_power-density_100m.tif")
geojson = gdal.Open("germany.geojson")

#some basic raster info
print(tif.RasterXSize, tif.RasterYSize)
print(tif.GetProjection())
print(tif.GetGeoTransform())
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
