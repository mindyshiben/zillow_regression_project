import pandas as pd
import numpy as np
from env import get_db_url
import os

def get_zillow_data():
    filename = 'zillow.csv'
    
    if os.path.isfile(filename):
        return pd.read_csv(filename)
    
    else:
        sql = """
        SELECT parcelid, taxvaluedollarcnt, bedroomcnt, bathroomcnt, yearbuilt, fips, calculatedfinishedsquarefeet, lotsizesquarefeet, latitude, longitude
        FROM properties_2017
        JOIN predictions_2017 USING (parcelid)
        WHERE transactiondate LIKE '2017%%' and propertylandusetypeid = 261;
        """

        df = pd.read_sql(sql, get_db_url('zillow'))

        df.to_csv(filename)

        return df 

def get_zillow_locs():
    filename = 'zillowloc.csv'

    if os.path.isfile(filename):
        return pd.read_csv(filename)

    else:
        sql = """
        SELECT parcelid, latitude, longitude
        FROM properties_2017
        WHERE propertylandusetypeid = 261;
        """

        df = pd.read_sql(sql, get_db_url('zillow'))

        df.to_csv(filename)

        return df

def get_zillow_zips():
    filename = 'zips_df.csv'

    if os.path.isfile(filename):
        return pd.read_csv(filename)
    
    else:
        sql = """
        SELECT parcelid, latitude, longitude
        FROM properties_2017
        WHERE propertylandusetypeid = 261;
        """

        df = pd.read_sql(sql, get_db_url('zillow'))

        df.to_csv(filename)

        return df

def get_zillow_other():
    filename = 'zillow_other.csv'
    
    if os.path.isfile(filename):
        return pd.read_csv(filename)
    
    else:
        sql = """
        SELECT parcelid
        FROM properties_2017
        JOIN predictions_2017 USING (parcelid)
        WHERE transactiondate LIKE '2017%%'  and propertylandusetypeid = 261;
        """

        df = pd.read_sql(sql, get_db_url('zillow'))

        df.to_csv(filename)

        return df 






