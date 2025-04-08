import pandas as pd

def get_stock_list():
    df = pd.read_csv('https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv')
    return sorted(df['Symbol'].unique())
