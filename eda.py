import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

def drop_columns(df, columns):
    return df.drop(columns, axis=1)

def set_datetime_index(df):
    df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y %I:%M:%S %p')
    df.index = pd.DatetimeIndex(df['Date'])
    return df

def plot_heatmap(df):
    correlation = df.corr()
    sns.set(rc={'figure.figsize': (15, 8)})
    sns.heatmap(correlation, xticklabels=correlation.columns, yticklabels=correlation.columns, annot=True)
    plt.show()

def plot_crimes_per_month(df):
    plt.figure(figsize=(11, 5))
    df.resample('M').size().plot(legend=False)
    plt.title('Number of crimes per month')
    plt.xlabel('Months')
    plt.ylabel('Number of crimes')
    plt.show()

def plot_crimes_by_type(df):
    plt.figure(figsize=(8, 10))
    df.groupby(['Primary Type']).size().sort_values(ascending=True).plot(kind='barh')
    plt.title('Number of crimes by type')
    plt.ylabel('Crime Type')
    plt.xlabel('Number of crimes')
    plt.show()

def plot_crimes_by_day(df):
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    df.groupby([df.index.dayofweek]).size().plot(kind='barh')
    plt.ylabel('Days of the week')
    plt.yticks(range(7), days)
    plt.xlabel('Number of crimes')
    plt.title('Number of crimes by days')
    plt.show()

def plot_crimes_by_location(df):
    plt.figure(figsize=(10, 30))
    df.groupby(['Location Description']).size().sort_values(ascending=True).plot(kind='barh')
    plt.title('Number of crimes by Location')
    plt.ylabel('Crime Location')
    plt.xlabel('Number of crimes')
    plt.show()
