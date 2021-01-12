import pandas as pd
import xlsxwriter

wind_data = pd.read_csv("renewable_power_plants_DE.csv", low_memory=False)

print("COLUMN NAMES", wind_data.columns)
print("NUMBER OF ROWS", wind_data.count())

print(wind_data['technology'].unique())

wind_data = wind_data.loc[wind_data['technology'] == "Onshore"]
print("NUMBER OF ROWS", wind_data.count())

writer = pd.ExcelWriter("DE_data.xlsx", engine="xlsxwriter")
wind_data.to_excel(writer, sheet_name="AllData")
writer.save()




