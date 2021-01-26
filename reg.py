import pandas as pd
from sklearn import linear_model
import statsmodels.api as sm


class multi_lin_reg:
    def __init__(self, input_df, independent_vars, dependent_var):
        x = input_df[independent_vars]
        y = input_df[dependent_var]

        # with sklearn
        self.regr = linear_model.LinearRegression()
        self.regr.fit(x, y)

        print('Intercept: \n', self.regr.intercept_)
        print('Coefficients: \n', self.regr.coef_)

        # with statsmodels
        x = sm.add_constant(x)  # adding a constant

        model = sm.OLS(y, x).fit()
        predictions = model.predict(x)

        print_model = model.summary()
        print(print_model)

    def predict_it(self, independent_vars):
        # prediction with sklearn
        print('Predicted value: \n', self.regr.predict([independent_vars]))
        return self.regr.predict([independent_vars])



