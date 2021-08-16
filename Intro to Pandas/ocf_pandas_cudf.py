# -*- coding: utf-8 -*-
"""OCF - Pandas/CuDF

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1HA9eTHoE3hpJ6vtmqBu-clt2Bhp6ly-M

# Intro to Pandas

Pandas is a Python package for data analysis and exposes two new
data structures: Dataframes and Series.

- [Dataframes](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html) store tabular data consisting of rows and columns.
- [Series](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.Series.html) are similar to Python's built-in list or set data types.

In this notebook, we will explore the data structures that Pandas
provides, and learn how to interact with them.

### 1. Importing Pandas

To import an external Python library such as Pandas, use Python's
import function. To save yourself some typing later on, you can
give the library you import an alias. Here, we are importing Pandas
and giving it an alias of `pd`.
"""

import pandas as pd

"""### 2. Creating A Dataframe and Basic Exploration
We will load a CSV file as a dataframe using Panda's `read_csv`
method. This will allow us to use Pandas' dataframe functions to
explore the data in the CSV.
"""

df = pd.read_csv ('https://raw.githubusercontent.com/ai6ph/Loan-default-prediction/main/Default_Fin.csv')

df.head()

df.shape

df.dtypes

df.describe()

"""### 3. Selecting Data
To examine a specfic column of the DataFrame either at the beginning or end, we respectively see the following:
"""

df['Employed'].head()

# Get rows 1 through 3 and columns 0 through 5.
df.iloc[1:3,:5]

# Get rows with index values of 2-4 and the columns basket_amount and activity
df.loc[2:4, ["Bank Balance", "Employed"]]

"""### 5. Checking for missing value

"""

df.isnull().sum().sum()

df.isnull().sum()

df.dropna()

df.fillna(method='ffill')

df.fillna(method='bfill')

"""# CuDF

cuDF is a Python-based GPU DataFrame library for working with data including loading, joining, aggregating, and filtering data and otherwise manipulating data. 
This makes it very easy for Data Scientists, Analysts, and Engineers to integrate it into their workflow.

## Some basic exploration with CuDF in comparison with pandas

The necessary dependencies will be installed to enable us run cudf in colab
"""

#import cudf
#import pandas as pd

#Loading the data
#%time gdf = cudf.read_csv('https://raw.githubusercontent.com/ai6ph/Loan-default-prediction/main/Default_Fin.csv')
#gdf.shape

#%time df = pd.read_csv('https://raw.githubusercontent.com/ai6ph/Loan-default-prediction/main/Default_Fin.csv')
#gdf.shape == df.shape

#converting data types
#%time gdf['Employed'] = gdf['Employed'].astype('float32')
#gdf['Employed']

#%time df['Employed'] = df['Employed'].astype('float32')
#df['Employed']

#String operation
#%time gdf['Bank Balance'] = gdf['Bank Balance'].astype('string')

#%time df['Bank Balance'] = df['Bank Balance'].astype('string')
#df['Bank Balance']