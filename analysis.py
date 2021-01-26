from osgeo import gdal
import os
import pandas as pd
import numpy as np
import seaborn as sns
import xlsxwriter
from matplotlib.pyplot import cm
import matplotlib.pyplot as plt

from reg import multi_lin_reg

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
    def __init__(self, tif_file=DE_tif, power_plant_file=DE_wind_data):
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
        x_pixel = []
        y_pixel = []

        for point in points_list:
            col = int((point[0] - xOrigin) / pixelWidth)
            row = int((yOrigin - point[1]) / pixelHeight)

            self.farms_pixel_coordinates.append((row, col))
            output_list.append(data[row][col])
            x_pixel.append(col)
            y_pixel.append(row)

        self.wind_data['average wind speed'] = output_list
        self.wind_data['x_pixel'] = x_pixel
        self.wind_data['y_pixel'] = y_pixel
        self.wind_data['decade'] = np.floor(self.wind_data['year']/10)*10

    def map_print(self):
        tif_array = self.dataset.ReadAsArray()
        img_plot = plt.imshow(tif_array, cmap="viridis")
        plt.savefig("Wind_speed_map_DE")
        plt.clf()
        plt.close()

    def heat_map_farms(self, scale_down=100, name="heatmap", lowest_value=10):
        cols = self.dataset.RasterXSize
        rows = self.dataset.RasterYSize
        heat_map_x_length = round(cols/scale_down)
        heat_map_y_length = round(rows/scale_down)

        heat_map = np.zeros((heat_map_x_length, heat_map_y_length))

        for i in self.wind_data.index:
            x = round(heat_map_x_length/cols * self.wind_data.loc[i, 'x_pixel']) - 1
            y = round(heat_map_y_length/rows * self.wind_data.loc[i, 'y_pixel']) - 1

            # add capacity
            heat_map[x][y] += self.wind_data.loc[i, 'electrical_capacity']

        sns.heatmap(heat_map, cmap='viridis', mask=(heat_map < lowest_value), square=True,
                    xticklabels=False, yticklabels=False, linewidths=0.5, robust=True)

        plt.savefig(name + "year_")
        plt.clf()
        plt.close()

    def heat_map_farms_yearly(self, scale_down=100, name="heatmap", lowest_value=10, years=[1990, 2000, 2010]):
        cols = self.dataset.RasterXSize
        rows = self.dataset.RasterYSize
        heat_map_x_length = round(cols/scale_down)
        heat_map_y_length = round(rows/scale_down)

        for ii in years:
            heat_map = np.zeros((heat_map_x_length, heat_map_y_length))
            yearly_df = self.wind_data.loc[(self.wind_data.year > ii) & (self.wind_data.year < ii+10)]

            for i in yearly_df.index:
                x = round(heat_map_x_length/cols * yearly_df.loc[i, 'x_pixel']) - 1
                y = round(heat_map_y_length/rows * yearly_df.loc[i, 'y_pixel']) - 1

                # add capacity
                heat_map[x][y] += yearly_df.loc[i, 'electrical_capacity']

            sns.heatmap(heat_map, cmap='viridis', mask=(heat_map < lowest_value), square=True,
                        xticklabels=False, yticklabels=False, linewidths=0.5, robust=True)

            plt.savefig(name + "year_" + str(ii))
            plt.clf()
            plt.close()

    def create_histogram(self, name="hist"):
        colours_list = cm.get_cmap("viridis")

        sns.histplot(data=self.wind_data, x="average wind speed", bins=20, kde=True,
                     weights="electrical_capacity", stat="probability", color=colours_list(0))

        plt.savefig(name)
        plt.clf()
        plt.close()

    def create_histogram_decade(self, years=[1980, 1990, 2000, 2010], name="hist"):
        for i in years:
            colours_list = cm.get_cmap("viridis")

            sns.histplot(data=self.wind_data.loc[self.wind_data.decade == i], x="average wind speed", bins=20, kde=True,
                         weights="electrical_capacity", stat="probability", color=colours_list((i-min(years))/(max(years)-min(years))))

            plt.savefig(name + "years" + str(i))
            plt.clf()
            plt.close()

    def create_histogram_hue(self):
        sns.histplot(data=self.wind_data, x="average wind speed", bins=20, kde=True, weights="electrical_capacity",
                     stat="probability", hue="decade", palette="viridis", common_norm=False, multiple="dodge")

        plt.savefig("hist_hue_decade")
        plt.clf()
        plt.close()

    def create_histogram_yearly_add(self, name="hist_yearly_add"):
        colours_list = cm.get_cmap("viridis")
        years = []
        sums = []

        for i in range(1980, 2019):
            years.append(i)
            new_df = (self.wind_data.loc[self.wind_data.year == i])
            sums.append(new_df["electrical_capacity"].sum())

        df = pd.DataFrame(list(zip(sums, years)), columns=["Sums", "Years"])
        sns.barplot(x="Years", y="Sums", data=df, palette="viridis")
        plt.xticks(rotation='vertical')

        plt.savefig(name)
        plt.clf()
        plt.close()

    def save(self):
        writer = pd.ExcelWriter("Analysis_output.xls", engine="xlsxwriter")
        self.wind_data.to_excel(writer, sheet_name="AllData")
        writer.save()

    def do_simple_reg(self):
        years = self.wind_data["year"].unique()
        years_sq = years ** 2

        year_count = []
        for i in range(1, len(years) +1):
            year_count.append(i)

        print(year_count)
        year_count_sq = [x**2 for x in year_count]
        print(year_count_sq)

        ref_yield = []
        for i in years:
            if i > 2012:
                ref_yield.append(0)
            else:
                ref_yield.append(1)

        average_speed = []
        for i in years:
            x = self.wind_data.loc[self.wind_data.year == i]
            x = x["average wind speed"].mean(axis=0)
            average_speed.append(x)

        df = pd.DataFrame(list(zip(years, years_sq, year_count, year_count_sq, ref_yield, average_speed)),
                          columns=["years", "years_sq", "year_count", "year_count_sq", "ref_yield", "average_speed"])

        multi_lin_reg(input_df=df, independent_vars=['year_count', 'ref_yield'], dependent_var=['average_speed'])


        # average wind speed per country, average per region
        # year, year2, referenceyield = TF, auctions = TF, FeedIn = TF


        # multi_lin_reg(input_df=df, independent_vars=['Interest_Rate', 'Unemployment_Rate'], dependent_var=['Stock_Index_Price'])

if __name__ == '__main__':
    Data = Analysis()

    """
    # Data.save()

    Data.map_print()

    for i in [1]:
        Data.heat_map_farms(name="heatmap_{0}".format(i), lowest_value=i)

        Data.heat_map_farms_yearly(name="heatmap_{0}".format(i), lowest_value=i)

    Data.create_histogram()
    Data.create_histogram_hue()
    Data.create_histogram_decade()

    Data.create_histogram_yearly_add()
    
    """

    Data.do_simple_reg()







