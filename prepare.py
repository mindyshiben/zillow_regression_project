
import pandas as pd
import numpy as np
from env import get_db_url
import acquire
import prepare
from IPython.display import Markdown, display
from sklearn.model_selection import train_test_split

def printmd(string):
    display(Markdown(string))

    '''  
    This function takes a string (a print statement) and returns the print statement in bold text"
    '''
    
def zillow_summary(df):
    print()
    printmd("**Zillow Data (Min, Max, Average)**")
    print("--------------------------------")
    printmd("**Tax Assessed Value of Home**")
    printmd('*Maximum Tax Assessed Value: {:,}*'
     .format(df['tax_value'].max()))
    printmd('*Minimum Tax Assessed Value: {:,}*'
     .format(df['tax_value'].min()))
    printmd('*Average Tax Assessed Value: {:,}*'
     .format(round(df['tax_value'].mean())))
    print("--------------------------------")
    printmd("**Home Size in Square Feet**")
    printmd('*Maximum Home Size: {:,} square feet*'
     .format(df['square_feet'].max()))
    printmd('*Minimum Home Size: {:,} square feet*'
     .format(df['square_feet'].min()))
    printmd('*Average Home Size: {:,} square feet*'
     .format(round(df['square_feet'].mean())))
    print("--------------------------------")
    printmd("**Lot Size in Square Feet**")
    printmd('*Maximum Lot Size: {:,} square feet*'
     .format(df['lot_size'].max()))
    printmd('*Minimum Lot Size: {:,} square feet*'
     .format(df['lot_size'].min()))
    printmd('*Average Lot Size: {:,} square feet*'
     .format(round(df['lot_size'].mean())))

def split_zillow_data(df):
    
    '''
    This function takes in a dataframe and splits it into three subgroups: train, test, validate
    for proper evalution, statistical testing, and modeling. Three dataframes are returned.
    '''

    train, test = train_test_split(df, test_size=.2, random_state=123)
    train, validate = train_test_split(train, test_size=.3, random_state=123)
    
    return train, validate, test