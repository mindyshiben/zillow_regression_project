import pandas as pd
import numpy as np
from env import get_db_url
import acquire 

def wrangle_zillow(df):
    df = df.dropna()
    df.rename(columns={'bedroomcnt': 'bedrooms', 'bathroomcnt': 'bathrooms', 'calculatedfinishedsquarefeet': 'square_feet', 'taxvaluedollarcnt': 'value', 'yearbuilt': 'year', 'taxamount': 'tax'}, inplace=True)
    df = df[df.bathrooms != 0]
    df = df[df.bathrooms != 1.75]
    df= df[df.bathrooms < 8]
    df = df[df.square_feet > 200]
    df = df[df.bedrooms != 0.0]
    df = df[df.bedrooms < 8.0]
    df = df[df.value > 200.0]
    df = df[df.value < 8000000.0]
    df = df[df.assessmentyear != 2014 & 2015]
    df.drop(columns = ['assessmentyear'], inplace=True)
    df['year'] = df['year'].astype('int')
    df['fips'] = df['fips'].astype('int')
    df['square_feet'] = df['square_feet'].astype('int')
    df['value'] = df['value'].astype('int')
    df['county'] = df['fips'].replace({6037: 'los_angeles', 6059: 'orange', 6111: 'ventura'})
    df.rename(columns={'landtaxvaluedollarcnt': 'land_value', 'lotsizesquarefeet': 'lot_size', 'square_feet': 'home_size'}, inplace=True)
    df['land_value'] = df.land_value.astype(int)
    df['lot_size'] = df.lot_size.astype(int)

    return df

def wrangle_locs(df):
    long = pd.DataFrame(df['longitude'].astype(str))
    for c in long:
        long[c] = (long[c].str[:4] + '.' + long[c].str[2:])
    lat = pd.DataFrame(df['latitude'].astype(str))
    for c in lat:
        lat[c] = (lat[c].str[:2] + '.' + lat[c].str[2:])
    df.drop(columns = ['latitude', 'longitude'], inplace=True)
    lat.latitude = lat.latitude.str.rstrip('.0') 
    df = pd.concat([df, lat, long], axis=1)

    return df

    