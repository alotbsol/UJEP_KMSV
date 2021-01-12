from osgeo import gdal
import os
import pandas as pd
import numpy as np
import xlsxwriter

import matplotlib.pyplot as plt

os.environ['PROJ_LIB'] = 'C:\\Users\\proko\\miniconda3\\pkgs\\proj-6.2.1-h9f7ef89_0\\Library\\share\\proj'
os.environ['GDAL_DATA'] = 'C:\\Users\\proko\\miniconda3\\pkgs\\proj-6.2.1-h9f7ef89_0\\Library\\share'

"""
I used this
https://gis.stackexchange.com/questions/221292/retrieve-pixel-value-with-geographic-coordinate-as-input-with-gdal/221430
"""

CZ_tif = "CZE_wind-speed_100m.tif"
CZ_wind_data = pd.read_excel("wind_renewable_power_plants_CZ.xls", sheet_name="wind_renewable_power_plants_CZ")

DE_tif = "DEU_wind-speed_100m.tif"
DE_wind_data = pd.read_excel("DE_data2.xls", sheet_name="AllData")

"""
DE_wind_data['lon'].replace('', np.nan, inplace=True)
DE_wind_data['lat'].replace('', np.nan, inplace=True)
DE_wind_data.dropna(subset=['lon'], inplace=True)
DE_wind_data.dropna(subset=['lat'], inplace=True)
"""


class Analysis:
    def __init__(self, tif_file=CZ_tif, power_plant_file=CZ_wind_data):

        self.wind_data = power_plant_file

        self.dataset = gdal.Open(tif_file)
        band = self.dataset.GetRasterBand(1)

        cols = self.dataset.RasterXSize
        rows = self.dataset.RasterYSize

        transform = self.dataset.GetGeoTransform()
        xOrigin = transform[0]
        yOrigin = transform[3]
        pixelWidth = transform[1]
        pixelHeight = -transform[5]
        data = band.ReadAsArray(0, 0, cols, rows)

        points_list = []
        for i in range(0, len(self.wind_data.lon)):
            point = (self.wind_data.lon[i], self.wind_data.lat[i])
            points_list.append(point)

        output_list = []
        self.farms_pixel_coordinates = []

        for point in points_list:
            col = int((point[0] - xOrigin) / pixelWidth)
            row = int((yOrigin - point[1]) / pixelHeight)

            self.farms_pixel_coordinates.append((row, col))
            output_list.append(data[row][col])

        print("farms_coordinates:", self.farms_pixel_coordinates)

        print("output list:", output_list)
        self.wind_data['average wind speed'] = output_list

    def map_print(self):
        tif_array = self.dataset.ReadAsArray()
        img_plot = plt.imshow(tif_array, cmap="magma")
        plt.show()

    def map_w_farms(self):
        tif_array = self.dataset.ReadAsArray()
        img_plot = plt.imshow(tif_array, cmap="magma")

        for i in self.farms_pixel_coordinates:
            plt.scatter(i[1], i[0], c="black", s=10, marker="o", alpha=1, zorder=3)

        plt.savefig("map_w_farms")
        plt.show()

    def heat_map_farms(self, resolution=100):
        pass

        cols = self.dataset.RasterXSize
        rows = self.dataset.RasterYSize

        heat_map = np.zeros((cols, rows))
        print(heat_map)
        for i in range(0, resolution):
            heat_map.append([])


        print(self.farms_pixel_coordinates)

        plt.imshow(self.farms_pixel_coordinates, cmap='hot', interpolation='nearest')
        plt.savefig("heat_map_farms")
        plt.show()

    def save(self):
        writer = pd.ExcelWriter("CZ_analysis_output.xlsx", engine="xlsxwriter")
        self.wind_data.to_excel(writer, sheet_name="AllData")
        writer.save()


if __name__ == '__main__':
    Data = Analysis()
    # Data.map_w_farms()
    Data.heat_map_farms()



