from osgeo import gdal
import os
import pandas as pd
import xlsxwriter

os.environ['PROJ_LIB'] = 'C:\\Users\\proko\\miniconda3\\pkgs\\proj-6.2.1-h9f7ef89_0\\Library\\share\\proj'
os.environ['GDAL_DATA'] = 'C:\\Users\\proko\\miniconda3\\pkgs\\proj-6.2.1-h9f7ef89_0\\Library\\share'

"""
I used this
https://gis.stackexchange.com/questions/221292/retrieve-pixel-value-with-geographic-coordinate-as-input-with-gdal/221430
"""

dataset = gdal.Open("CZE_wind-speed_100m.tif")
band = dataset.GetRasterBand(1)

cols = dataset.RasterXSize
rows = dataset.RasterYSize

transform = dataset.GetGeoTransform()
xOrigin = transform[0]
yOrigin = transform[3]
pixelWidth = transform[1]
pixelHeight = -transform[5]

data = band.ReadAsArray(0, 0, cols, rows)


wind_data = pd.read_excel("wind_renewable_power_plants_CZ.xls", sheet_name="wind_renewable_power_plants_CZ",)

points_list = []
for i in range(0, len(wind_data.lon)):
    point = (wind_data.lon[i], wind_data.lat[i])
    points_list.append(point)

output_list = []
for point in points_list:
    col = int((point[0] - xOrigin) / pixelWidth)
    row = int((yOrigin - point[1]) / pixelHeight)

    print(row, col, data[row][col])

    output_list.append(data[row][col])

print("output list:", output_list)
wind_data['average wind speed'] = output_list

print(wind_data)

#write
writer = pd.ExcelWriter("DE_analysis_output.xlsx", engine="xlsxwriter")
wind_data.to_excel(writer, sheet_name="AllData")
writer.save()
