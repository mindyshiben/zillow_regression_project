import pandas as pd
import numpy as np
from env import get_db_url
import acquire 

def wrangle_zillow(df):

    '''
    This function takes a dataframe and applies several parameters to clean the data in useable form including renmaming
    columns, removing outliers, and changing data types. A cleaned dataframe is returned.
    '''

    df = df.dropna()
    df.rename(columns={'bedroomcnt': 'bedrooms', 'taxvaluedollarcnt': 'tax_value', 'lotsizesquarefeet': 'lot_size', 'bathroomcnt': 'bathrooms', 'calculatedfinishedsquarefeet': 'square_feet', 'yearbuilt': 'year_built'}, inplace=True)
    df = df[df.bathrooms != 0]
    df = df[df.bathrooms != 1.75]
    df= df[df.bathrooms < 7]
    df = df[df.bedrooms != 0.0]
    df = df[df.bedrooms < 8.0]
    df = df[df.tax_value < 2500000.0]
    df = df[df.year_built > 1900]
    df = df[df.lot_size > 500]
    df = df[df.square_feet > 500]
    df = df[df.square_feet < 5000]
    df = df[df.tax_value > 100000.0]
    df['year_built'] = df['year_built'].astype('int')
    df['fips'] = df['fips'].astype('int')
    df['square_feet'] = df['square_feet'].astype('int')
    df['tax_value'] = df['tax_value'].astype('int')
    df['county'] = df['fips'].replace({6037: 'los_angeles', 6059: 'orange', 6111: 'ventura'})
    df['lot_size'] = df.lot_size.astype(int)

    return df

def wrangle_locs(df):

    '''
    This function takes a dataframe and manipulates the format of latitude and longitude columns 
    to be in the correct form. The dataframe is returned with said changes.
    '''
    
    long = pd.DataFrame(df['longitude'])
    for c in long:
        long[c] = (long[c] / 1000000)
    lat = pd.DataFrame(df['latitude'])
    for c in lat:
        lat[c] = (lat[c] / 1000000)
    df.drop(columns = ['latitude', 'longitude'], inplace=True)
    df = pd.concat([df, lat, long], axis=1)

    return df


    