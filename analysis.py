from osgeo import gdal
import os
import pandas as pd
import numpy as np
import seaborn as sns
import xlsxwriter
from matplotlib.pyplot import cm
import matplotlib.pyplot as plt
import xlrd

import statsmodels.api as sm
from statsmodels.formula.api import ols

import scipy.stats as stats
from reg import multi_lin_reg
from technical_data import average_hub_heights

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

        self.regulatory_phases = {"Period1": [1979, 1990],
                                  "Period2": [1990, 1995], "Period3": [1995, 2000],
                                  "Period4": [2000, 2005], "Period5": [2005, 2010],
                                  "Period6": [2010, 2015], "Period7": [2015, 2019]}

        """
        self.regulatory_phases = {"Phase1": [1979, 1990], "Phase2": [1990, 2000],
                                  "Phase3": [2000, 2017], "Phase4": [2017, 2019]}
                                  """

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

        ax = plt.axes()
        sns.heatmap(heat_map, ax=ax, cmap='viridis', mask=(heat_map < lowest_value), square=True,
                    xticklabels=False, yticklabels=False, linewidths=0.5, robust=True,
                    cbar_kws={'label': 'MW'})

        text_props = dict(boxstyle='square', facecolor='white', edgecolor="none", alpha=0.9, pad=0.5)

        total_added_capacity = round(self.wind_data['electrical_capacity'].sum(), 1)
        years_period = self.wind_data['year'].max() - self.wind_data['year'].min()

        plt.text(0, 1,
                 "Total added capacity: " + str(total_added_capacity) + " MW"
                 + "\n"
                 "Added capacity per year: " + str(round(total_added_capacity/years_period)) + " MW",
                 fontsize=10, verticalalignment='top', bbox=text_props)

        ax.set_title(str(self.wind_data['year'].min()) + " - " + str(self.wind_data['year'].max()))

        plt.savefig(name + "year_")
        plt.clf()
        plt.close()

    def heat_map_farms_yearly(self, scale_down=200, name="heatmap", lowest_value=10,):
        cols = self.dataset.RasterXSize
        rows = self.dataset.RasterYSize
        heat_map_x_length = round(cols/scale_down)
        heat_map_y_length = round(rows/scale_down)

        for ii in self.regulatory_phases:
            heat_map = np.zeros((heat_map_x_length, heat_map_y_length))
            yearly_df = self.wind_data.loc[(self.wind_data.year >= self.regulatory_phases[ii][0]) &
                                           (self.wind_data.year < self.regulatory_phases[ii][1])]

            for iii in yearly_df.index:
                x = round(heat_map_x_length/cols * yearly_df.loc[iii, 'x_pixel']) - 1
                y = round(heat_map_y_length/rows * yearly_df.loc[iii, 'y_pixel']) - 1

                # add capacity
                heat_map[x][y] += yearly_df.loc[iii, 'electrical_capacity']

            ax = plt.axes()
            sns.heatmap(heat_map, ax=ax, cmap='viridis', mask=(heat_map < lowest_value), square=True,
                        xticklabels=False, yticklabels=False, linewidths=0.5, robust=True,
                        cbar_kws={'label': 'MW'})

            ax.set_title(str(ii) + ": " + str(self.regulatory_phases[ii][0]) + " - "
                         + str(self.regulatory_phases[ii][1]-1))

            text_props = dict(boxstyle='square', facecolor='white', edgecolor="none", alpha=0.9, pad=0.5)

            total_added_capacity = round(yearly_df['electrical_capacity'].sum(),)
            years_period = self.regulatory_phases[ii][1] - self.regulatory_phases[ii][0]

            plt.text(0, 1,
                     "Total added capacity: " + str(total_added_capacity) + " MW"
                     + "\n"
                     "Added capacity per year: " + str(round(total_added_capacity/years_period,)) + " MW",

                     fontsize=10, verticalalignment='top', bbox=text_props)

            plt.savefig(name + "year_" + str(ii))
            plt.clf()
            plt.close()

    def create_histogram(self, name="hist"):
        colours_list = cm.get_cmap("viridis")

        ax = plt.axes()
        sns.histplot(data=self.wind_data, ax=ax, x="average wind speed", bins=20, kde=True,
                     weights="electrical_capacity", stat="probability", color=colours_list(0))

        ax.set_title(str(self.wind_data['year'].min()) + " - " + str(self.wind_data['year'].max()))


        text_props = dict(boxstyle='square', facecolor='white', edgecolor="none", alpha=0.9, pad=0.5)
        total_added_capacity = round(self.wind_data['electrical_capacity'].sum(), 1)
        years_period = self.wind_data['year'].max() - self.wind_data['year'].min()

        y = ax.get_ylim()
        x = ax.get_xlim()

        avg = self.wind_data["average wind speed"].mean()
        med = self.wind_data["average wind speed"].median()
        stand_dev = self.wind_data["average wind speed"].std()
        plt.text(x[0]*1.05, y[1]*0.95,
                 "Total added capacity: " + str(total_added_capacity) + " MW"
                 + "\n"
                 "Added capacity per year: " + str(round(total_added_capacity/years_period)) + " MW"
                 + "\n"
                   "Mean: " + str(round(avg, 1))
                 + "\n"
                   "Median: " + str(round(med, 1))
                 + "\n"
                   "Standard deviation: " + str(round(stand_dev, 1)),

                 fontsize=10, verticalalignment='top', ha="left", bbox=text_props)

        plt.savefig(name)
        plt.clf()
        plt.close()

    def create_histogram_yearly(self, name="hist"):

        for i in self.regulatory_phases:
            colours_list = cm.get_cmap("viridis")

            yearly_df = self.wind_data.loc[(self.wind_data.year >= self.regulatory_phases[i][0]) &
                                           (self.wind_data.year < self.regulatory_phases[i][1])]

            ax = plt.axes()
            sns.histplot(ax=ax, data=yearly_df, x="average wind speed", bins=20, kde=True,
                         weights="electrical_capacity", stat="probability", color=colours_list(0))

            ax.set_title(str(i) + ": " + str(self.regulatory_phases[i][0]) + " - "
                         + str(self.regulatory_phases[i][1]-1))

            text_props = dict(boxstyle='square', facecolor='white', edgecolor="none", alpha=0.9, pad=0.5)
            total_added_capacity = round(yearly_df['electrical_capacity'].sum())
            years_period = self.regulatory_phases[i][1] - self.regulatory_phases[i][0]

            y = ax.get_ylim()
            x = ax.get_xlim()

            avg = yearly_df["average wind speed"].mean()
            med = yearly_df["average wind speed"].median()
            stand_dev = yearly_df["average wind speed"].std()

            plt.text(x[0] * 1.05, y[1] * 0.95,
                     "Total added capacity: " + str(total_added_capacity) + " MW"
                     + "\n"
                       "Added capacity per year: " + str(round(total_added_capacity / years_period)) + " MW"
                     + "\n"
                       "Mean: " + str(round(avg, 1))
                     + "\n"
                       "Median: " + str(round(med, 1))
                     + "\n"
                       "Standard deviation: " + str(round(stand_dev, 1)),

                     fontsize=10, verticalalignment='top', ha="left", bbox=text_props)

            plt.savefig(name + "years" + str(i))
            plt.clf()
            plt.close()

    def average_per_region(self):
        y = 0
        years = self.wind_data["year"].unique()
        states = self.wind_data["federal_state"].unique()

        df_pivot = pd.pivot_table(self.wind_data, values="average wind speed", index="federal_state", columns=["year"],
                                  aggfunc=np.mean)

        df_average = pd.pivot_table(self.wind_data, values="average wind speed", columns=["year"],
                                    aggfunc=np.mean)

        """
        average_speed = []
        for i in years:
            x = self.wind_data.loc[self.wind_data.year == i]
            x = x["average wind speed"].mean(axis=0)
            average_speed.append(x)
        """

        fig, ax = plt.subplots(figsize=(18, 9))
        sns.lineplot(x="year", y="average wind speed", hue="federal_state",
                     data=self.wind_data, estimator="mean", ci=None, marker='o', markersize=2)
        sns.lineplot(x="year", y="average wind speed", data=self.wind_data, estimator="mean", color="black",
                     linewidth=4)

        plt.savefig("Average_wind_speed{0}.png".format(str(y)))
        plt.clf()
        plt.close()

        y += 1

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

    def do_simple_reg(self, base_year=1990):
        year = self.wind_data["year"].unique()
        year_sq = year ** 2

        year_count = []
        for i in range(1, len(year) +1):
            year_count.append(i)

        year_count_sq = [x**2 for x in year_count]

        phases = {}
        for i in self.regulatory_phases:
            phases[i] = []
            for ii in year:
                if ii >= self.regulatory_phases[i][0] and ii < self.regulatory_phases[i][1]:
                    phases[i].append(1)
                else:
                    phases[i].append(0)

        ref_yield = []
        for i in year:
            if i > 2000:
                ref_yield.append(1)
            else:
                ref_yield.append(0)


        average_speed = []
        for i in year:
            x = self.wind_data.loc[self.wind_data.year == i]
            x = x["average wind speed"].mean(axis=0)
            average_speed.append(x)

        df = pd.DataFrame(list(zip(year, year_sq, year_count, year_count_sq, ref_yield, average_speed)),
                          columns=["year", "year_sq", "year_count", "year_count_sq", "ref_yield", "average_speed"])

        for i in phases:
            df[i] = phases[i]

        df['average_hub_height'] = df['year'].map(average_hub_heights)

        print(df)

        """adjusted base year"""
        df = df.loc[df.year > base_year]

        the_model = multi_lin_reg(input_df=df,
                                  independent_vars=["average_hub_height", "ref_yield",],
                                  dependent_var=['average_speed'])

        predictions = []
        predictions_t = []
        predictions_f = []

        for i in df["year"]:
            predictions_t.append(float(the_model.predict_it(independent_vars=[average_hub_heights[i], 1])))
            predictions_f.append(float(the_model.predict_it(independent_vars=[average_hub_heights[i], 0])))
            predictions.append(float(the_model.predict_it(independent_vars=[average_hub_heights[i],
                                                          df.loc[df["year"] == i, "ref_yield"].iloc[0]])))

        """
        the_model = multi_lin_reg(input_df=df,
                          independent_vars=["year_count", "year_count_sq", "ref_yield",
                                            ],
                          dependent_var=['average_speed'])

        predictions = []
        predictions_t = []
        predictions_f = []

        for i in df["year_count"]:
            predictions_t.append(float(the_model.predict_it(independent_vars=[i, i**2, 1])))
            predictions_f.append(float(the_model.predict_it(independent_vars=[i, i**2, 0])))
            predictions.append(float(the_model.predict_it(independent_vars=[i, i**2,
                                                        df.loc[df["year_count"] == i, "ref_yield"].iloc[0]])))
        
        """




        for i in [predictions, predictions_t, predictions_f, df["average_speed"]]:
            plt.plot(df["year"].unique(), i, label=["predictions", "predictions", "predictions", "average speed"])

        plt.show()

        # average wind speed per country, average per region
        # year, year2, referenceyield = TF, auctions = TF, FeedIn = TF


    def do_anova(self, low_year_limit=1995, up_year_limit=2005, base_year=2000):
        """ https://www.reneshbedre.com/blog/anova.html """

        data_before = self.wind_data.loc[(self.wind_data.year < base_year) & (self.wind_data.year >= low_year_limit)]
        data_after = self.wind_data.loc[(self.wind_data.year < up_year_limit) & (self.wind_data.year >= base_year)]

        fvalue, pvalue = stats.f_oneway(data_before["average wind speed"],
                                        data_after["average wind speed"])
        print(fvalue, pvalue)


        new_df = pd.DataFrame(data=self.wind_data.loc[(self.wind_data.year < up_year_limit) & (self.wind_data.year >= low_year_limit)]
                              [["year", "average wind speed"]])

        new_df.year[new_df.year < base_year] = low_year_limit
        new_df.year[new_df.year >= base_year] = up_year_limit-1
        print(new_df)

        new_df = new_df.reset_index()

        new_df.columns = ['index', 'treatments', 'value']
        print(new_df)

        colours_list = cm.get_cmap("viridis")
        ax = sns.boxplot(x='treatments', y='value', data=new_df, showmeans=True, color=colours_list(0.2))

        plt.savefig("Means_boxplot_anova")
        plt.clf()
        plt.close()

        model = ols('value ~ C(treatments)', new_df).fit()
        # print(model.summary())
        res = sm.stats.anova_lm(model, typ=2)
        print(res)


        """
        model = ols('value ~ C(treatments)', data=new_df).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)

        print(anova_table)
        """

    def do_anova_multi_way(self, low_year_limit=1995, up_year_limit=2005, base_year=2000):
        """ https://www.reneshbedre.com/blog/anova.html """

        data_before = self.wind_data.loc[(self.wind_data.year < base_year) & (self.wind_data.year >= low_year_limit)]
        data_after = self.wind_data.loc[(self.wind_data.year < up_year_limit) & (self.wind_data.year >= base_year)]

        fvalue, pvalue = stats.f_oneway(data_before["average wind speed"],
                                        data_after["average wind speed"])
        print(fvalue, pvalue)

        new_df = pd.DataFrame(
            data=self.wind_data.loc[(self.wind_data.year < up_year_limit) & (self.wind_data.year >= low_year_limit)]
            [["year", "average wind speed"]])

        new_df.year[new_df.year < base_year] = low_year_limit
        new_df.year[new_df.year >= base_year] = up_year_limit - 1
        print(new_df)

        new_df = new_df.reset_index()

        new_df.columns = ['index', 'treatments', 'value']
        print(new_df)

        colours_list = cm.get_cmap("viridis")
        ax = sns.boxplot(x='treatments', y='value', data=new_df, showmeans=True, color=colours_list(0.2))

        plt.savefig("Means_boxplot_anova")
        plt.clf()
        plt.close()

        model = ols('value ~ C(treatments)', new_df).fit()
        # print(model.summary())
        res = sm.stats.anova_lm(model, typ=2)
        print(res)




    def reg_by_state(self, base_year=1980):
        # df_grouped_by = self.wind_data[["average wind speed", "year", "federal_state"]]
        # df_grouped_by = df_grouped_by.groupby(["federal_state", "year"]).mean().reset_index()

        df = self.wind_data[["average wind speed", "year", "federal_state"]].copy()
        first_year = df["year"].min() - 1
        df["year_count"] = df["year"] - first_year

        ref_yield = []
        for i in df["year"]:
            if i > 2013:
                ref_yield.append(1)
            else:
                ref_yield.append(0)

        df["ref_yield"] = ref_yield
        df = df.loc[df.year > base_year]

        for i in ["Niedersachsen"]:
            df_adjusted = df.loc[df.federal_state == i].copy()

            first_year = df_adjusted["year"].min()

            df_adjusted["year_count"] = df_adjusted["year"] - first_year

            df_adjusted = df_adjusted.groupby(["year", ]).mean().reset_index()

            the_model = multi_lin_reg(input_df=df_adjusted, independent_vars=['year_count', "ref_yield"],
                                      dependent_var=['average wind speed'])

            predictions = []
            predictions_t = []
            predictions_f = []
            for ii in df_adjusted["year_count"]:
                predictions_t.append(float(the_model.predict_it(independent_vars=[ii, 1])))
                predictions_f.append(float(the_model.predict_it(independent_vars=[ii, 0])))
                predictions.append(float(the_model.predict_it(independent_vars=[ii,
                                            df_adjusted.loc[df_adjusted["year_count"] == ii, "ref_yield"].iloc[0]])))

            for ii in (predictions, predictions_t, predictions_f, df_adjusted["average wind speed"]):
                plt.plot(df_adjusted["year"], ii, label=["predictions", "predictions TRUE", "predictions False" "average speed"])


            plt.show()


if __name__ == '__main__':
    Data = Analysis()
    # Data.save()

    """
    Data.map_print()
    """

    """
    Data.heat_map_farms(name="heatmap", lowest_value=1)
    Data.heat_map_farms_yearly(name="heatmap", lowest_value=0.001)
    
    Data.create_histogram()
    Data.create_histogram_yearly()
  
    Data.average_per_region()
    """



    """
    Data.do_simple_reg()
    """


    Data.do_anova()


    # Data.reg_by_state()







