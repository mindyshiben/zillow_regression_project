## Zillow Regression Project

### Project Goals:

- Construct a machine learning regression model that improves predicted propery tax assessed values of single family homes in Ventura, Orange, and Los Angeles counties.
- Find the key drivers of property value for single family properties.
- Forcast tax assessed home value with enhanced accuracy.
- Empower zillow executives to improve the current predictions using my recommentations.
- Thoroughly document the process and key findings.

### Hypotheses :

> #1.)  Location - Based on basic domain knowledge, I believe location plays a large role in determining tax assessed property value. While the county may be a driver of the target(tax assessed property value), I'd like to dive deeper by narrowing in on location even further. Location determines other unknown attributes (such as proximity to the ocean, school district, etc.) which could greatly impact tax assessed value. Zip codes in the original data are incorrect and do not align with properties in California. Therefore, I will attempt to use latitude and longitude coordinates to explore the notion that location may drive the target variable.

> #2.) Square Footage - I hypothesize that home size and potentially lot size are related to tax assessed property value. Logically, I would anticipate larger home having a higher tax assessed value. I'm not sure whether or not the lot size will good predictor of the target as smaller homes may be on a lot of land and vice versa, yet it's possible that lot size and the target are correlated the majority of the time. I will explore these notions through graphical and statistical anaylsis.

> #3.) Rooms Counts - I believe that as the number of bedrooms and bathroom increases, the tax assessed property value increases. However, there is not much variance in these features and an overwhelming majority of properties have the same number of bedrooms and/or bathrooms. Because of this, I hypothesize that bedroom and bathroom counts will not be the top drivers of the target. Because my basic domain knowledge tells me that room counts likely impact value, I will explore these potential relationships.

> #4.) Year Built - My intuition tells me that more modern homes may have enhanced attributes which could mean that year built may be positively correlated to the target. However, my research shows that "historic homes" only need to be 50 years old (or more) and historic homes may be valued higher. I am unsure if (and how many) homes in this dataset are classified as "historic" and whether or not such homes would be considered outliers. I hypothesis that the majority of home ages are coorelated with the target as such older homes would have lower tax assessed values. I will explore this notion and keep an eye out for outliers.

### Data Dictionary:

- parcelid: property identifier
- bedroomcnt: bedroom count of property
- bathroomcnt: bathroom count of property
- yearbuilt: property's year built
- fips: county of property
- calculatedfinishedsquarefeet: square footage of property
- lotsizesquarefeet: square footage of property's lot/land
- latitude: latitude coordinate of property
- longitude: longitude coordinate of property
- Project Planning (lay out your process through the data science pipeline):
- data acquistion, preparation, exploratory data analysis and statistical testing, modeling, and model evaluation


### Key Findings, Recommendations, and Takeaways
- My analysis indicates that the top drivers of tax assessed home values are:
     > - property size (square footage)
     > - property's year built
     > - bedroom count
     > - bathroom count

### Summary of Recommendations:

- I built and trained a Polynomial Regression model which is able to improve predict tax assessed home values by ~ $80,000 (22% improved from baseline/previous predictions).
 
- My polynomial regression model has been tailored with ideal parameters (degree=3) and has performed consistently well on trained data and unseen data.

- This model is ready for utilization and I strongly recommend Zillow to employ this model to predict tax assessed property value as soon as possible.

### Reproducing this project
Acquire and utilize credentials to access the Zillow database. Store credentials and database access in a env.py file and create a .gitignore which includes env.py to protect your credentials. Replicate zillow_final_report.ipynb, acquire.py, and wrangle.py from this repository. 

