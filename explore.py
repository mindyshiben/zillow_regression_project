{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "3867a004",
   "metadata": {},
   "outputs": [],
   "source": [
    "import warnings\n",
    "warnings.filterwarnings(\"ignore\")\n",
    "\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "\n",
    "from sklearn.model_selection import train_test_split\n",
    "from scipy.stats import pearsonr, spearmanr\n",
    "\n",
    "import env\n",
    "import acquire\n",
    "import wrangle"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "a4ebdc07",
   "metadata": {},
   "outputs": [],
   "source": [
    "from math import sqrt\n",
    "from scipy import stats"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "d44a3386",
   "metadata": {},
   "outputs": [],
   "source": [
    "df = acquire.get_zillow_data()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "ade6a0c7",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 2152863 entries, 0 to 2152862\n",
      "Data columns (total 12 columns):\n",
      " #   Column                        Dtype  \n",
      "---  ------                        -----  \n",
      " 0   bedroomcnt                    float64\n",
      " 1   bathroomcnt                   float64\n",
      " 2   calculatedfinishedsquarefeet  float64\n",
      " 3   taxvaluedollarcnt             float64\n",
      " 4   yearbuilt                     float64\n",
      " 5   taxamount                     float64\n",
      " 6   fips                          float64\n",
      " 7   assessmentyear                float64\n",
      " 8   landtaxvaluedollarcnt         float64\n",
      " 9   lotsizesquarefeet             float64\n",
      " 10  latitude                      float64\n",
      " 11  longitude                     float64\n",
      "dtypes: float64(12)\n",
      "memory usage: 197.1 MB\n"
     ]
    }
   ],
   "source": [
    "df.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "b33561cb",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.01705589254866659"
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.isna().sum().sum() / len(df)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "b010e384",
   "metadata": {},
   "outputs": [],
   "source": [
    "df = wrangle.wrangle_zillow(df)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "5fc998a7",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "Int64Index: 2118616 entries, 4 to 2152861\n",
      "Data columns (total 13 columns):\n",
      " #   Column                 Dtype  \n",
      "---  ------                 -----  \n",
      " 0   bedrooms               float64\n",
      " 1   bathrooms              float64\n",
      " 2   square_feet            int64  \n",
      " 3   value                  int64  \n",
      " 4   year                   int64  \n",
      " 5   tax                    float64\n",
      " 6   fips                   int64  \n",
      " 7   assessmentyear         float64\n",
      " 8   landtaxvaluedollarcnt  float64\n",
      " 9   lotsizesquarefeet      float64\n",
      " 10  latitude               float64\n",
      " 11  longitude              float64\n",
      " 12  county                 object \n",
      "dtypes: float64(8), int64(4), object(1)\n",
      "memory usage: 226.3+ MB\n"
     ]
    }
   ],
   "source": [
    "df.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "ed0d54dc",
   "metadata": {},
   "outputs": [],
   "source": [
    "df.rename(columns={'landtaxvaluedollarcnt': 'land_value', 'lotsizesquarefeet': 'lot_square_feet'}, inplace=True) "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "00fa1d87",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "Int64Index: 2118616 entries, 4 to 2152861\n",
      "Data columns (total 13 columns):\n",
      " #   Column           Dtype  \n",
      "---  ------           -----  \n",
      " 0   bedrooms         float64\n",
      " 1   bathrooms        float64\n",
      " 2   square_feet      int64  \n",
      " 3   value            int64  \n",
      " 4   year             int64  \n",
      " 5   tax              float64\n",
      " 6   fips             int64  \n",
      " 7   assessmentyear   float64\n",
      " 8   land_value       float64\n",
      " 9   lot_square_feet  float64\n",
      " 10  latitude         float64\n",
      " 11  longitude        float64\n",
      " 12  county           object \n",
      "dtypes: float64(8), int64(4), object(1)\n",
      "memory usage: 226.3+ MB\n"
     ]
    }
   ],
   "source": [
    "df.info()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "d01010fd",
   "metadata": {},
   "source": [
    "### Hypotheses, Questions, what I want to explore-\n",
    "\n",
    "    - bedrooms, bathrooms, and square feet- want to assess this relationship\n",
    "    - is the pecrentage of tax to value consistent based on county?\n",
    "    - is tax impacted by year built?\n",
    "    - what zipcodes fall under what fips?\n",
    "    - what is the relationship of zipcode to value?\n",
    "    - is home square feet or lot square feet more correlated to value?\n",
    "    - how much are home square footage and lot square footage correlated to eachother?\n",
    "    - how is year built related to value?"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "e0273feb",
   "metadata": {},
   "outputs": [],
   "source": [
    "train_and_validate, test = train_test_split(df, test_size=.2, random_state=123)\n",
    "train, validate = train_test_split(train_and_validate, test_size=.3, random_state=123)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "dfb50e33",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "436585.6049404441"
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "(train['value'].where(train.fips == 6111)).mean()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "f6eb57bf",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "515008.1275567997"
      ]
     },
     "execution_count": 12,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "(train['value'].where(train.fips == 6059)).mean()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "a892c562",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "416560.875220892"
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "(train['value'].where(train.fips == 6037)).mean()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "ec5eee4d",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "4987.887828602173"
      ]
     },
     "execution_count": 14,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "(train['tax'].where(train.fips == 6111)).mean()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "7cc4192e",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "5947.72110103503"
      ]
     },
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "(train['tax'].where(train.fips == 6059)).mean()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "8894df16",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "5256.943536911473"
      ]
     },
     "execution_count": 16,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "(train['tax'].where(train.fips == 6037)).mean()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "5078d06a",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<AxesSubplot:xlabel='value', ylabel='tax'>"
      ]
     },
     "execution_count": 17,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.scatterplot(x='value', y='tax', data=train, hue='county')"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "8f192c1f",
   "metadata": {},
   "source": [
    "Takeaway- average tax is different per county"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "050d27ee",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<AxesSubplot:xlabel='county', ylabel='value'>"
      ]
     },
     "execution_count": 18,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.barplot(x='county', y='value', data=train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "fabd4c2e",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<AxesSubplot:xlabel='assessmentyear', ylabel='value'>"
      ]
     },
     "execution_count": 20,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAZgAAAEGCAYAAABYV4NmAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjQuMywgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/MnkTPAAAACXBIWXMAAAsTAAALEwEAmpwYAAAXIklEQVR4nO3dfbTdVX3n8feHRBBBkIfwYECDQ3wAaxVSxHHacYoNdKzCGnGariJxSodVFvWhTxTaWqZQaqW2tNqBGSwIIi0g2jHtLMQMih1dCISHgoBAFIQAkWiQxgcoge/8cXYmJ5eTcJPcncu99/1a66zzO/u39/7tk1/Ih9/+nbNPqgpJkibadpM9AEnS9GTASJK6MGAkSV0YMJKkLgwYSVIXsyd7AM8Xe+65Z82bN2+yhyFJU8pNN9303aqaM2qfAdPMmzePZcuWTfYwJGlKSfLtje1zikyS1IUBI0nqwoCRJHVhwEiSujBgJEldGDCSpC4MGElSFwaMJKkLv2g5A5xyyimsXLmSffbZh7PPPnuyhyNphjBgZoCVK1fy0EMPTfYwJM0wTpFJkrowYCRJXRgwkqQuDBhJUhcGjCSpCwNGktSFASNJ6sKAkSR1YcBIkrowYCRJXRgwkqQuDBhJUhcGjCSpCwNGktSFASNJ6sKAkSR1YcBIkrroGjBJfiPJHUm+nuTvkrwwye5Jlia5tz3vNlT/tCTLk9yd5Mih8kOT3N72fTRJWvkOSS5v5dcnmTfUZnE7xr1JFvd8n5KkZ+sWMEnmAu8DFlTVa4FZwCLgVOCaqpoPXNNek+Sgtv9g4Cjg3CSzWnfnAScC89vjqFZ+AvBYVR0InAN8uPW1O3A68EbgMOD04SCTJPXXe4psNrBjktnAi4CHgaOBi9v+i4Fj2vbRwGVV9WRV3QcsBw5Lsi+wS1VdV1UFfHJMm3V9XQkc0a5ujgSWVtXqqnoMWMr6UJIkbQPdAqaqHgI+AjwAPAI8XlVfAPauqkdanUeAvVqTucCDQ12saGVz2/bY8g3aVNVa4HFgj030tYEkJyZZlmTZqlWrtvzNSpKepecU2W4MrjAOAF4K7JTkuE01GVFWmyjf0jbrC6rOr6oFVbVgzpw5mxiaJGlz9ZwieytwX1WtqqqngM8C/xb4Tpv2oj0/2uqvAPYfar8fgym1FW17bPkGbdo03K7A6k30JUnaRnoGzAPA4Ule1O6LHAHcBSwB1n2qazHwuba9BFjUPhl2AIOb+Te0abQ1SQ5v/Rw/ps26vo4Fvtju01wNLEyyW7uSWtjKJEnbyOxeHVfV9UmuBG4G1gK3AOcDOwNXJDmBQQi9q9W/I8kVwJ2t/slV9XTr7iTgImBH4Kr2ALgAuCTJcgZXLotaX6uTnAnc2OqdUVWre71XSdKzdQsYgKo6ncHHhYc9yeBqZlT9s4CzRpQvA147ovwJWkCN2HchcOFmDll63jnllFNYuXIl++yzD2efffZkD0cat64BI2nrrVy5koceemiyhyFtNpeKkSR1YcBIkrpwimwLHfo7n5zsIYzbi7+7hlnAA99dM6XGfdOfHT/ZQ5C0FbyCkSR1YcBIkrowYCRJXRgwkqQuDBhJUhcGjCSpCwNGktSFASNJ6sKAkSR1YcBIkrowYCRJXRgwkqQuDBhJUhcGjCSpCwNGktSFASNJ6sKAkSR1YcBIkrowYCRJXRgwkqQuDBhJUhcGjCSpCwNGktSFASNJ6sKAkSR1YcBIkrqYPdkDUH/PbL/TBs+StC0YMDPAD+cvnOwhSJqBnCKTJHVhwEiSujBgJEldGDCSpC4MGElSF10DJslLklyZ5BtJ7krypiS7J1ma5N72vNtQ/dOSLE9yd5Ijh8oPTXJ72/fRJGnlOyS5vJVfn2TeUJvF7Rj3Jlnc831Kkp6t9xXMXwGfr6pXAz8J3AWcClxTVfOBa9prkhwELAIOBo4Czk0yq/VzHnAiML89jmrlJwCPVdWBwDnAh1tfuwOnA28EDgNOHw4ySVJ/3QImyS7AzwAXAFTVv1bV94GjgYtbtYuBY9r20cBlVfVkVd0HLAcOS7IvsEtVXVdVBXxyTJt1fV0JHNGubo4EllbV6qp6DFjK+lCSJG0DPa9gXgGsAj6R5JYkf5NkJ2DvqnoEoD3v1erPBR4car+ilc1t22PLN2hTVWuBx4E9NtHXBpKcmGRZkmWrVq3amvcqSRqjZ8DMBg4BzquqNwA/pE2HbURGlNUmyre0zfqCqvOrakFVLZgzZ84mhiZJ2lw9A2YFsKKqrm+vr2QQON9p016050eH6u8/1H4/4OFWvt+I8g3aJJkN7Aqs3kRfkqRtpFvAVNVK4MEkr2pFRwB3AkuAdZ/qWgx8rm0vARa1T4YdwOBm/g1tGm1NksPb/ZXjx7RZ19exwBfbfZqrgYVJdms39xe2MknSNtJ7scv3Apcm2R74FvBfGITaFUlOAB4A3gVQVXckuYJBCK0FTq6qp1s/JwEXATsCV7UHDD5AcEmS5QyuXBa1vlYnORO4sdU7o6pW93yjkqQNdQ2YqroVWDBi1xEbqX8WcNaI8mXAa0eUP0ELqBH7LgQu3IzhSpImkN/klyR1YcBIkrowYCRJXRgwkqQu/MlkzTgPnPETkz2EzbJ29e7AbNau/vaUGfvL/vD2yR6Cnge8gpEkdWHASJK6MGAkSV0YMJKkLgwYSVIXBowkqQsDRpLUhQEjSerCgJEkdWHASJK6eM6ASbJ3kguSXNVeH9R+LEySpI0azxXMRQx+bvil7fU9wAc6jUeSNE2MJ2D2rKorgGcAqmot8PSmm0iSZrrxBMwPk+wBFECSw4HHu45KkjTljWe5/t8ElgD/JslXgTnAsV1HJUma8p4zYKrq5iT/HngVEODuqnqq+8gkSVPacwZMkuPHFB2ShKr6ZKcxSZKmgfFMkf3U0PYLgSOAmwEDRpK0UeOZInvv8OskuwKXdBuRJGla2JJv8v8ImD/RA5EkTS/juQfzD7SPKDMIpIOAK3oOSpI09Y3nHsxHhrbXAt+uqhWdxiNJmibGcw/my9tiIJKk6WWjAZNkDeunxjbYBVRV7dJtVJKkKW+jAVNVL96WA5EkTS/juQcDQJK9GHwPBoCqeqDLiCRJ08J4PkX2DuDPGSzX/yjwcuAu4OC+Q5OkZ3vzx9482UOY9r763q9OSD/j+R7MmcDhwD1VdQCDb/JPzNElSdPWeALmqar6HrBdku2q6kvA6/sOS5I01Y3nHsz3k+wM/F/g0iSPMvg+jCRJGzWeK5h/Al4CvB/4PPBN4O0dxyRJmgbGEzABrgauBXYGLm9TZpIkbdRzBkxV/VFVHQyczOCTZF9O8n+6j0ySNKVtzmrKjwIrge8Be423UZJZSW5J8o/t9e5Jlia5tz3vNlT3tCTLk9yd5Mih8kOT3N72fTRJWvkOSS5v5dcnmTfUZnE7xr1JFm/G+5QkTYDnDJgkJyW5FrgG2BP4r1X1us04xvsZfG9mnVOBa6pqfuvz1Hacg4BFDL5fcxRwbpJZrc15wIkMfiZgftsPcALwWFUdCJwDfLj1tTtwOvBG4DDg9OEgkyT1N54rmJcDH6iqg6vq9Kq6c7ydJ9kPeBvwN0PFRwMXt+2LgWOGyi+rqier6j5gOXBYkn2BXarquqoqBr+kecyIvq4EjmhXN0cCS6tqdVU9BixlfShJkraB8aymfOpW9P+XwCnA8Lpme1fVI63vR9oSNABzga8N1VvRyp5q22PL17V5sPW1NsnjwB7D5SPa/H9JTmRwZcTLXvayzX93kqSN2pJftByXJL8APFpVN423yYiy2kT5lrZZX1B1flUtqKoFc+bMGecwJUnj0S1ggDcD70hyP3AZ8LNJPgV8p0170Z4fbfVXAPsPtd8PeLiV7zeifIM2SWYDuwKrN9GXJGkb6RYwVXVaVe1XVfMY3Lz/YlUdBywB1n2qazHwuba9BFjUPhl2AIOb+Te06bQ1SQ5v91eOH9NmXV/HtmMUg+/tLEyyW7u5v7CVSZK2kXEv1z+B/hS4IskJwAPAuwCq6o4kVwB3MliK5uSqerq1OQm4CNgRuKo9AC4ALkmynMGVy6LW1+okZwI3tnpnVNXq3m9MkrTeNgmYqrqWwUoAtFUAjthIvbOAs0aULwNeO6L8CVpAjdh3IXDhlo5ZkrR1et6DkSTNYJMxRSZpM+z5wmeAte1ZmjoMGOl57rdf9/3JHoK0RZwikyR1YcBIkrowYCRJXRgwkqQuDBhJUhcGjCSpCwNGktSFASNJ6sKAkSR1YcBIkrowYCRJXRgwkqQuDBhJUhcGjCSpCwNGktSFASNJ6sKAkSR1YcBIkrowYCRJXRgwkqQuDBhJUhcGjCSpCwNGktSFASNJ6sKAkSR1YcBIkrowYCRJXRgwkqQuDBhJUhcGjCSpCwNGktSFASNJ6sKAkSR1YcBIkrroFjBJ9k/ypSR3Jbkjyftb+e5Jlia5tz3vNtTmtCTLk9yd5Mih8kOT3N72fTRJWvkOSS5v5dcnmTfUZnE7xr1JFvd6n5Kk0XpewawFfquqXgMcDpyc5CDgVOCaqpoPXNNe0/YtAg4GjgLOTTKr9XUecCIwvz2OauUnAI9V1YHAOcCHW1+7A6cDbwQOA04fDjJJUn/dAqaqHqmqm9v2GuAuYC5wNHBxq3YxcEzbPhq4rKqerKr7gOXAYUn2BXapquuqqoBPjmmzrq8rgSPa1c2RwNKqWl1VjwFLWR9KkqRtYJvcg2lTV28Argf2rqpHYBBCwF6t2lzgwaFmK1rZ3LY9tnyDNlW1Fngc2GMTfY0d14lJliVZtmrVqq14h5KksboHTJKdgc8AH6iqf9lU1RFltYnyLW2zvqDq/KpaUFUL5syZs4mhSZI2V9eASfICBuFyaVV9thV/p0170Z4fbeUrgP2Hmu8HPNzK9xtRvkGbJLOBXYHVm+hLkrSN9PwUWYALgLuq6i+Gdi0B1n2qazHwuaHyRe2TYQcwuJl/Q5tGW5Pk8Nbn8WParOvrWOCL7T7N1cDCJLu1m/sLW5kkaRuZ3bHvNwPvBm5Pcmsr+z3gT4ErkpwAPAC8C6Cq7khyBXAng0+gnVxVT7d2JwEXATsCV7UHDALskiTLGVy5LGp9rU5yJnBjq3dGVa3u9D4lSSN0C5iq+gqj74UAHLGRNmcBZ40oXwa8dkT5E7SAGrHvQuDC8Y5XkjSx/Ca/JKkLA0aS1IUBI0nqwoCRJHVhwEiSujBgJEldGDCSpC4MGElSFwaMJKkLA0aS1IUBI0nqwoCRJHVhwEiSujBgJEldGDCSpC4MGElSFwaMJKkLA0aS1IUBI0nqwoCRJHVhwEiSujBgJEldGDCSpC4MGElSFwaMJKkLA0aS1IUBI0nqwoCRJHVhwEiSujBgJEldGDCSpC4MGElSFwaMJKkLA0aS1IUBI0nqwoCRJHVhwEiSujBgJEldTOuASXJUkruTLE9y6mSPR5JmkmkbMElmAf8d+HngIOCXkhw0uaOSpJlj2gYMcBiwvKq+VVX/ClwGHD3JY5KkGSNVNdlj6CLJscBRVfWr7fW7gTdW1a8P1TkROLG9fBVw9zYf6LazJ/DdyR6Etpjnb+qa7ufu5VU1Z9SO2dt6JNtQRpRtkKZVdT5w/rYZzuRKsqyqFkz2OLRlPH9T10w+d9N5imwFsP/Q6/2AhydpLJI040zngLkRmJ/kgCTbA4uAJZM8JkmaMabtFFlVrU3y68DVwCzgwqq6Y5KHNZlmxFTgNOb5m7pm7Lmbtjf5JUmTazpPkUmSJpEBI0nqwoCZApLsn+RLSe5KckeS97fy3ZMsTXJve96tle/R6v8gyV9vpM8lSb6+iWOe1pbYuTvJkX3e2cwwkecvybXtnNzaHntt5Jievwkywedv+yTnJ7knyTeSvHMjx5wW58+AmRrWAr9VVa8BDgdObsvenApcU1XzgWvaa4AngA8Cvz2qsyT/CfjBxg7W+l4EHAwcBZzblt7RlpnQ8wf8clW9vj0eHbvT8zfhJvL8/T7waFW9ksESVl8eW2E6nT8DZgqoqkeq6ua2vQa4C5jLYOmbi1u1i4FjWp0fVtVXGPxF30CSnYHfBP54E4c8Grisqp6sqvuA5QyW3tEWmMjzN06evwk0wefvV4APtXrPVNWob/hPm/NnwEwxSeYBbwCuB/auqkdg8B8BMHK6ZIwzgT8HfrSJOnOBB4der2hl2koTcP4APtGmxz6YZNSKFZ6/Trbm/CV5Sds8M8nNST6dZO8RVafN+TNgppB29fEZ4ANV9S9b0P71wIFV9ffPVXVEmZ9n30pbe/6aX66qnwB+uj3ePepQI8o8f1tpAs7fbAYriny1qg4BrgM+MupQI8qm5PkzYKaIJC9g8Jf70qr6bCv+TpJ92/59gWfNx4/xJuDQJPcDXwFemeTaEfVcZmeCTdD5o6oeas9rgL9l9NSJ52+CTdD5+x6DmYN1/4P3aeCQEfWmzfkzYKaANg1yAXBXVf3F0K4lwOK2vRj43Kb6qarzquqlVTUP+HfAPVX1lhFVlwCLkuyQ5ABgPnDD1r2LmWuizl+S2Un2bNsvAH4BGPVJQM/fBJrA//4K+AfgLa3oCODOEVWnz/mrKh/P8weDMCjgNuDW9viPwB4MPr1yb3vefajN/cBqBp8WWwEcNKbPecDXh16/Azhj6PXvA99k8BMGPz/ZfwZT+TFR5w/YCbip9XMH8FfALM/f1Dh/rfzlwD+1vq4BXjadz59LxUiSunCKTJLUhQEjSerCgJEkdWHASJK6MGAkSV0YMNI0l+Q9SV462ePQzGPASNPfe4CuAZNk2v78uracASM1Sf5Xkpvab36cmGRWkouSfD3J7Ul+o9V7X5I7k9yW5LJWtlOSC5PcmOSWJEe38oOT3NAWp7wtyfxW938n+efW9y+2uvcn+ZMk1yVZluSQJFcn+WaSXxsa5++049yW5I9a2bz2eyUfb+P/QpIdkxwLLAAubWN4W5K/H+rr55J8tm0vbMdetxDjzq38D9vxvt5+yySt/No23i8D798Gp0hTzWR/09OHj+fLg/ZNbGBHBkuwHAosHdr/kvb8MLDDmLI/AY5bVwbcw+Cb9x9jsEAlwPat73cCHx/qd9f2fD9wUts+h8G3vV8MzGHwGyIAC4HzGSyIuB3wj8DPMFiZYS3w+lbviqHxXAssaNsBvgHMaa//Fng7sCeDb5jv1Mp/F/jD4T+Xtn0J8Pahfs+d7PPm4/n78ApGWu99Sf4Z+BqDxQa3B16R5GNJjgLWraB7G4MrguMY/KMOg3/4T01yK4N/eF8IvIzBirm/l+R3gZdX1Y+B24G3Jvlwkp+uqseHxrCkPd8OXF9Va6pqFfBEW+59YXvcAtwMvJrBWlUA91XVrW37Jgahs4GqKgYhcVzr703AVQx+SOsg4KvtPSxmsKwJwH9Icn2S24GfZfBDWOtcvtE/Tc14zptKQJK3AG8F3lRVP2qrTO8A/CRwJHAy8J8Z/GDU2xhcNbwD+GCSgxlcGbyzqu4e0/VdSa5vba5O8qtV9cUkhzJYz+pDSb5QVWe0+k+252eGtte9nt2O86Gq+p9jxj9vTP2nGVwtjfIJBosuPgF8uqrWtmmvpVX1S2P6fSFwLoMroAeT/DcG4bnODzdyDMkrGKnZFXishcurGfwf/Z7AdlX1GQY/gXtIku2A/avqS8ApDKbDdgauBt47dH/iDe35FcC3quqjDK5OXtc+0fWjqvoUg98DGbVk+8ZcDfzK0P2RuUme64fK1jCYagOgqh5mMM33B8BFrfhrwJuTHNj6fVGSV7I+TL7bjnnsZoxVM5xXMNLA54FfS3IbgxVsv8bgVwSvbaECcBowC/hUkl0ZXE2cU1XfT3Im8JfAbS1k7mewnP4vMpiOegpYCZwB/BTwZ0meAZ4CThrvIKvqC0leA1zXsuwHwHEMrlg25iLgfyT5MYMrtB8DlzK4D3Nn63dVkvcAf5dkh9buD6rqniQfZzBldz9w43jHKrmasjQDJflr4JaqumCyx6Lpy4CRZpgkNzG4d/JzVfXkc9WXtpQBI0nqwpv8kqQuDBhJUhcGjCSpCwNGktSFASNJ6uL/AQWmEutPkFuhAAAAAElFTkSuQmCC\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.barplot(x='assessmentyear', y='value', data =train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "id": "0ebd5855",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "2016.0    1186111\n",
       "2014.0        290\n",
       "2015.0         23\n",
       "Name: assessmentyear, dtype: int64"
      ]
     },
     "execution_count": 21,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train.assessmentyear.value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "id": "040e4987",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<AxesSubplot:title={'center':'Feature Correlation to Tax Value'}>"
      ]
     },
     "execution_count": 22,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 1296x1080 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.figure(figsize=(18,15))\n",
    "plt.title(\"Feature Correlation to Tax Value\")\n",
    "corr_chart = train.drop(\"value\", axis=1).corrwith(train['value']).sort_values().plot.barh()\n",
    "corr_chart"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "id": "8cce0dc0",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 454.375x360 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.relplot(x=\"land_value\", y=\"value\", data=train, hue='county')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "id": "f5b9e83c",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.barplot(x=\"bathrooms\", y=\"value\", data=train)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "id": "9eda1409",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 454.375x360 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.relplot(x=\"tax\", y=\"value\", data=train, hue='county')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "id": "fb8c625d",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>bedrooms</th>\n",
       "      <th>bathrooms</th>\n",
       "      <th>square_feet</th>\n",
       "      <th>value</th>\n",
       "      <th>year</th>\n",
       "      <th>tax</th>\n",
       "      <th>fips</th>\n",
       "      <th>assessmentyear</th>\n",
       "      <th>land_value</th>\n",
       "      <th>lot_square_feet</th>\n",
       "      <th>latitude</th>\n",
       "      <th>longitude</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>county</th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>los_angeles</th>\n",
       "      <td>3.190846</td>\n",
       "      <td>2.126276</td>\n",
       "      <td>1750.645893</td>\n",
       "      <td>416560.875221</td>\n",
       "      <td>1954.950027</td>\n",
       "      <td>5256.943537</td>\n",
       "      <td>6037.0</td>\n",
       "      <td>2015.999416</td>\n",
       "      <td>254389.173148</td>\n",
       "      <td>11367.453872</td>\n",
       "      <td>3.408725e+07</td>\n",
       "      <td>-1.182287e+08</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>orange</th>\n",
       "      <td>3.495463</td>\n",
       "      <td>2.439148</td>\n",
       "      <td>2040.722583</td>\n",
       "      <td>515008.127557</td>\n",
       "      <td>1971.946921</td>\n",
       "      <td>5947.721101</td>\n",
       "      <td>6059.0</td>\n",
       "      <td>2015.999740</td>\n",
       "      <td>332176.611295</td>\n",
       "      <td>7412.626831</td>\n",
       "      <td>3.372818e+07</td>\n",
       "      <td>-1.178520e+08</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>ventura</th>\n",
       "      <td>3.570747</td>\n",
       "      <td>2.395010</td>\n",
       "      <td>2006.917953</td>\n",
       "      <td>436585.604940</td>\n",
       "      <td>1973.986461</td>\n",
       "      <td>4987.887829</td>\n",
       "      <td>6111.0</td>\n",
       "      <td>2015.999307</td>\n",
       "      <td>222450.852865</td>\n",
       "      <td>12040.170617</td>\n",
       "      <td>3.424675e+07</td>\n",
       "      <td>-1.189946e+08</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "             bedrooms  bathrooms  square_feet          value         year  \\\n",
       "county                                                                      \n",
       "los_angeles  3.190846   2.126276  1750.645893  416560.875221  1954.950027   \n",
       "orange       3.495463   2.439148  2040.722583  515008.127557  1971.946921   \n",
       "ventura      3.570747   2.395010  2006.917953  436585.604940  1973.986461   \n",
       "\n",
       "                     tax    fips  assessmentyear     land_value  \\\n",
       "county                                                            \n",
       "los_angeles  5256.943537  6037.0     2015.999416  254389.173148   \n",
       "orange       5947.721101  6059.0     2015.999740  332176.611295   \n",
       "ventura      4987.887829  6111.0     2015.999307  222450.852865   \n",
       "\n",
       "             lot_square_feet      latitude     longitude  \n",
       "county                                                    \n",
       "los_angeles     11367.453872  3.408725e+07 -1.182287e+08  \n",
       "orange           7412.626831  3.372818e+07 -1.178520e+08  \n",
       "ventura         12040.170617  3.424675e+07 -1.189946e+08  "
      ]
     },
     "execution_count": 26,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train.groupby(train.county).mean()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "id": "04025901",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "327613.5"
      ]
     },
     "execution_count": 27,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "np.quantile(train.value, .5)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "id": "62427d55",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>bedrooms</th>\n",
       "      <th>bathrooms</th>\n",
       "      <th>square_feet</th>\n",
       "      <th>value</th>\n",
       "      <th>year</th>\n",
       "      <th>tax</th>\n",
       "      <th>fips</th>\n",
       "      <th>assessmentyear</th>\n",
       "      <th>land_value</th>\n",
       "      <th>lot_square_feet</th>\n",
       "      <th>latitude</th>\n",
       "      <th>longitude</th>\n",
       "      <th>county</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>1828910</th>\n",
       "      <td>1.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>938</td>\n",
       "      <td>1982995</td>\n",
       "      <td>1947</td>\n",
       "      <td>21424.90</td>\n",
       "      <td>6059</td>\n",
       "      <td>2015.0</td>\n",
       "      <td>1332502.0</td>\n",
       "      <td>3541.0</td>\n",
       "      <td>33596263.0</td>\n",
       "      <td>-117868889.0</td>\n",
       "      <td>orange</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1486712</th>\n",
       "      <td>3.0</td>\n",
       "      <td>3.0</td>\n",
       "      <td>3154</td>\n",
       "      <td>641356</td>\n",
       "      <td>2004</td>\n",
       "      <td>7312.12</td>\n",
       "      <td>6111</td>\n",
       "      <td>2015.0</td>\n",
       "      <td>165616.0</td>\n",
       "      <td>243065.0</td>\n",
       "      <td>34251372.0</td>\n",
       "      <td>-118758217.0</td>\n",
       "      <td>ventura</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1410568</th>\n",
       "      <td>2.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>886</td>\n",
       "      <td>230780</td>\n",
       "      <td>1953</td>\n",
       "      <td>2803.10</td>\n",
       "      <td>6037</td>\n",
       "      <td>2015.0</td>\n",
       "      <td>216367.0</td>\n",
       "      <td>3243.0</td>\n",
       "      <td>33987629.0</td>\n",
       "      <td>-118453238.0</td>\n",
       "      <td>los_angeles</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>818347</th>\n",
       "      <td>3.0</td>\n",
       "      <td>2.5</td>\n",
       "      <td>1688</td>\n",
       "      <td>327750</td>\n",
       "      <td>2002</td>\n",
       "      <td>329.94</td>\n",
       "      <td>6059</td>\n",
       "      <td>2015.0</td>\n",
       "      <td>136476.0</td>\n",
       "      <td>3043.0</td>\n",
       "      <td>33877166.0</td>\n",
       "      <td>-117988611.0</td>\n",
       "      <td>orange</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1572339</th>\n",
       "      <td>4.0</td>\n",
       "      <td>3.0</td>\n",
       "      <td>1938</td>\n",
       "      <td>542280</td>\n",
       "      <td>1979</td>\n",
       "      <td>5882.66</td>\n",
       "      <td>6111</td>\n",
       "      <td>2015.0</td>\n",
       "      <td>271140.0</td>\n",
       "      <td>9732.0</td>\n",
       "      <td>34230500.0</td>\n",
       "      <td>-119032528.0</td>\n",
       "      <td>ventura</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>968187</th>\n",
       "      <td>4.0</td>\n",
       "      <td>2.0</td>\n",
       "      <td>1313</td>\n",
       "      <td>1954386</td>\n",
       "      <td>1952</td>\n",
       "      <td>21298.92</td>\n",
       "      <td>6059</td>\n",
       "      <td>2015.0</td>\n",
       "      <td>1061740.0</td>\n",
       "      <td>5970.0</td>\n",
       "      <td>33535022.0</td>\n",
       "      <td>-117775350.0</td>\n",
       "      <td>orange</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>118726</th>\n",
       "      <td>4.0</td>\n",
       "      <td>2.5</td>\n",
       "      <td>2360</td>\n",
       "      <td>93686</td>\n",
       "      <td>1965</td>\n",
       "      <td>1041.98</td>\n",
       "      <td>6111</td>\n",
       "      <td>2015.0</td>\n",
       "      <td>19363.0</td>\n",
       "      <td>10259.0</td>\n",
       "      <td>34285937.0</td>\n",
       "      <td>-119191356.0</td>\n",
       "      <td>ventura</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1758322</th>\n",
       "      <td>2.0</td>\n",
       "      <td>2.0</td>\n",
       "      <td>1219</td>\n",
       "      <td>655743</td>\n",
       "      <td>1932</td>\n",
       "      <td>515.73</td>\n",
       "      <td>6037</td>\n",
       "      <td>2015.0</td>\n",
       "      <td>524597.0</td>\n",
       "      <td>7334.0</td>\n",
       "      <td>34101398.0</td>\n",
       "      <td>-118619628.0</td>\n",
       "      <td>los_angeles</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1141995</th>\n",
       "      <td>3.0</td>\n",
       "      <td>2.5</td>\n",
       "      <td>2092</td>\n",
       "      <td>661000</td>\n",
       "      <td>2005</td>\n",
       "      <td>7021.46</td>\n",
       "      <td>6111</td>\n",
       "      <td>2015.0</td>\n",
       "      <td>435000.0</td>\n",
       "      <td>34400.0</td>\n",
       "      <td>34163927.0</td>\n",
       "      <td>-118846286.0</td>\n",
       "      <td>ventura</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>218</th>\n",
       "      <td>4.0</td>\n",
       "      <td>3.0</td>\n",
       "      <td>3207</td>\n",
       "      <td>710970</td>\n",
       "      <td>1991</td>\n",
       "      <td>7607.58</td>\n",
       "      <td>6111</td>\n",
       "      <td>2015.0</td>\n",
       "      <td>171740.0</td>\n",
       "      <td>52172.0</td>\n",
       "      <td>34271805.0</td>\n",
       "      <td>-118987939.0</td>\n",
       "      <td>ventura</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1432246</th>\n",
       "      <td>4.0</td>\n",
       "      <td>3.5</td>\n",
       "      <td>2534</td>\n",
       "      <td>110074</td>\n",
       "      <td>1963</td>\n",
       "      <td>1220.76</td>\n",
       "      <td>6111</td>\n",
       "      <td>2015.0</td>\n",
       "      <td>24437.0</td>\n",
       "      <td>13199.0</td>\n",
       "      <td>34283367.0</td>\n",
       "      <td>-119268966.0</td>\n",
       "      <td>ventura</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1626371</th>\n",
       "      <td>4.0</td>\n",
       "      <td>3.0</td>\n",
       "      <td>2428</td>\n",
       "      <td>559608</td>\n",
       "      <td>1964</td>\n",
       "      <td>6130.14</td>\n",
       "      <td>6111</td>\n",
       "      <td>2015.0</td>\n",
       "      <td>322998.0</td>\n",
       "      <td>14435.0</td>\n",
       "      <td>34277894.0</td>\n",
       "      <td>-118902559.0</td>\n",
       "      <td>ventura</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>560004</th>\n",
       "      <td>4.0</td>\n",
       "      <td>3.5</td>\n",
       "      <td>2683</td>\n",
       "      <td>716143</td>\n",
       "      <td>1965</td>\n",
       "      <td>7479.50</td>\n",
       "      <td>6111</td>\n",
       "      <td>2015.0</td>\n",
       "      <td>289141.0</td>\n",
       "      <td>34630.0</td>\n",
       "      <td>34289386.0</td>\n",
       "      <td>-119207165.0</td>\n",
       "      <td>ventura</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1841208</th>\n",
       "      <td>3.0</td>\n",
       "      <td>1.5</td>\n",
       "      <td>2072</td>\n",
       "      <td>104769</td>\n",
       "      <td>1950</td>\n",
       "      <td>1134.14</td>\n",
       "      <td>6111</td>\n",
       "      <td>2015.0</td>\n",
       "      <td>29558.0</td>\n",
       "      <td>33006.0</td>\n",
       "      <td>34248938.0</td>\n",
       "      <td>-119043436.0</td>\n",
       "      <td>ventura</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1001417</th>\n",
       "      <td>3.0</td>\n",
       "      <td>2.5</td>\n",
       "      <td>2033</td>\n",
       "      <td>239922</td>\n",
       "      <td>2002</td>\n",
       "      <td>329.94</td>\n",
       "      <td>6059</td>\n",
       "      <td>2015.0</td>\n",
       "      <td>7586.0</td>\n",
       "      <td>2106.0</td>\n",
       "      <td>33877234.0</td>\n",
       "      <td>-117991103.0</td>\n",
       "      <td>orange</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>713420</th>\n",
       "      <td>3.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>1007</td>\n",
       "      <td>180666</td>\n",
       "      <td>1949</td>\n",
       "      <td>2707.72</td>\n",
       "      <td>6037</td>\n",
       "      <td>2015.0</td>\n",
       "      <td>106276.0</td>\n",
       "      <td>6019.0</td>\n",
       "      <td>33935094.0</td>\n",
       "      <td>-118331329.0</td>\n",
       "      <td>los_angeles</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1981286</th>\n",
       "      <td>5.0</td>\n",
       "      <td>2.5</td>\n",
       "      <td>2640</td>\n",
       "      <td>560988</td>\n",
       "      <td>1965</td>\n",
       "      <td>6438.56</td>\n",
       "      <td>6111</td>\n",
       "      <td>2015.0</td>\n",
       "      <td>364642.0</td>\n",
       "      <td>25000.0</td>\n",
       "      <td>34259250.0</td>\n",
       "      <td>-118784561.0</td>\n",
       "      <td>ventura</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>323630</th>\n",
       "      <td>2.0</td>\n",
       "      <td>2.0</td>\n",
       "      <td>1162</td>\n",
       "      <td>248966</td>\n",
       "      <td>2005</td>\n",
       "      <td>3009.36</td>\n",
       "      <td>6111</td>\n",
       "      <td>2015.0</td>\n",
       "      <td>87757.0</td>\n",
       "      <td>3000.0</td>\n",
       "      <td>34260126.0</td>\n",
       "      <td>-118668308.0</td>\n",
       "      <td>ventura</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>527787</th>\n",
       "      <td>3.0</td>\n",
       "      <td>2.0</td>\n",
       "      <td>1636</td>\n",
       "      <td>288948</td>\n",
       "      <td>1970</td>\n",
       "      <td>3082.94</td>\n",
       "      <td>6111</td>\n",
       "      <td>2015.0</td>\n",
       "      <td>166065.0</td>\n",
       "      <td>8235.0</td>\n",
       "      <td>34286002.0</td>\n",
       "      <td>-119191085.0</td>\n",
       "      <td>ventura</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1249078</th>\n",
       "      <td>3.0</td>\n",
       "      <td>1.5</td>\n",
       "      <td>1268</td>\n",
       "      <td>430333</td>\n",
       "      <td>1912</td>\n",
       "      <td>4631.70</td>\n",
       "      <td>6111</td>\n",
       "      <td>2015.0</td>\n",
       "      <td>322751.0</td>\n",
       "      <td>10450.0</td>\n",
       "      <td>34279214.0</td>\n",
       "      <td>-119280606.0</td>\n",
       "      <td>ventura</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2078569</th>\n",
       "      <td>2.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>807</td>\n",
       "      <td>90825</td>\n",
       "      <td>1925</td>\n",
       "      <td>1393.45</td>\n",
       "      <td>6037</td>\n",
       "      <td>2015.0</td>\n",
       "      <td>89438.0</td>\n",
       "      <td>9027.0</td>\n",
       "      <td>34206426.0</td>\n",
       "      <td>-118456235.0</td>\n",
       "      <td>los_angeles</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>625034</th>\n",
       "      <td>3.0</td>\n",
       "      <td>3.0</td>\n",
       "      <td>2126</td>\n",
       "      <td>389975</td>\n",
       "      <td>1980</td>\n",
       "      <td>4285.06</td>\n",
       "      <td>6111</td>\n",
       "      <td>2015.0</td>\n",
       "      <td>168824.0</td>\n",
       "      <td>2450.0</td>\n",
       "      <td>34159017.0</td>\n",
       "      <td>-119220159.0</td>\n",
       "      <td>ventura</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1432386</th>\n",
       "      <td>4.0</td>\n",
       "      <td>3.0</td>\n",
       "      <td>2046</td>\n",
       "      <td>1108158</td>\n",
       "      <td>1984</td>\n",
       "      <td>11875.18</td>\n",
       "      <td>6111</td>\n",
       "      <td>2015.0</td>\n",
       "      <td>695626.0</td>\n",
       "      <td>113518.0</td>\n",
       "      <td>34249012.0</td>\n",
       "      <td>-119044177.0</td>\n",
       "      <td>ventura</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "         bedrooms  bathrooms  square_feet    value  year       tax  fips  \\\n",
       "1828910       1.0        1.0          938  1982995  1947  21424.90  6059   \n",
       "1486712       3.0        3.0         3154   641356  2004   7312.12  6111   \n",
       "1410568       2.0        1.0          886   230780  1953   2803.10  6037   \n",
       "818347        3.0        2.5         1688   327750  2002    329.94  6059   \n",
       "1572339       4.0        3.0         1938   542280  1979   5882.66  6111   \n",
       "968187        4.0        2.0         1313  1954386  1952  21298.92  6059   \n",
       "118726        4.0        2.5         2360    93686  1965   1041.98  6111   \n",
       "1758322       2.0        2.0         1219   655743  1932    515.73  6037   \n",
       "1141995       3.0        2.5         2092   661000  2005   7021.46  6111   \n",
       "218           4.0        3.0         3207   710970  1991   7607.58  6111   \n",
       "1432246       4.0        3.5         2534   110074  1963   1220.76  6111   \n",
       "1626371       4.0        3.0         2428   559608  1964   6130.14  6111   \n",
       "560004        4.0        3.5         2683   716143  1965   7479.50  6111   \n",
       "1841208       3.0        1.5         2072   104769  1950   1134.14  6111   \n",
       "1001417       3.0        2.5         2033   239922  2002    329.94  6059   \n",
       "713420        3.0        1.0         1007   180666  1949   2707.72  6037   \n",
       "1981286       5.0        2.5         2640   560988  1965   6438.56  6111   \n",
       "323630        2.0        2.0         1162   248966  2005   3009.36  6111   \n",
       "527787        3.0        2.0         1636   288948  1970   3082.94  6111   \n",
       "1249078       3.0        1.5         1268   430333  1912   4631.70  6111   \n",
       "2078569       2.0        1.0          807    90825  1925   1393.45  6037   \n",
       "625034        3.0        3.0         2126   389975  1980   4285.06  6111   \n",
       "1432386       4.0        3.0         2046  1108158  1984  11875.18  6111   \n",
       "\n",
       "         assessmentyear  land_value  lot_square_feet    latitude    longitude  \\\n",
       "1828910          2015.0   1332502.0           3541.0  33596263.0 -117868889.0   \n",
       "1486712          2015.0    165616.0         243065.0  34251372.0 -118758217.0   \n",
       "1410568          2015.0    216367.0           3243.0  33987629.0 -118453238.0   \n",
       "818347           2015.0    136476.0           3043.0  33877166.0 -117988611.0   \n",
       "1572339          2015.0    271140.0           9732.0  34230500.0 -119032528.0   \n",
       "968187           2015.0   1061740.0           5970.0  33535022.0 -117775350.0   \n",
       "118726           2015.0     19363.0          10259.0  34285937.0 -119191356.0   \n",
       "1758322          2015.0    524597.0           7334.0  34101398.0 -118619628.0   \n",
       "1141995          2015.0    435000.0          34400.0  34163927.0 -118846286.0   \n",
       "218              2015.0    171740.0          52172.0  34271805.0 -118987939.0   \n",
       "1432246          2015.0     24437.0          13199.0  34283367.0 -119268966.0   \n",
       "1626371          2015.0    322998.0          14435.0  34277894.0 -118902559.0   \n",
       "560004           2015.0    289141.0          34630.0  34289386.0 -119207165.0   \n",
       "1841208          2015.0     29558.0          33006.0  34248938.0 -119043436.0   \n",
       "1001417          2015.0      7586.0           2106.0  33877234.0 -117991103.0   \n",
       "713420           2015.0    106276.0           6019.0  33935094.0 -118331329.0   \n",
       "1981286          2015.0    364642.0          25000.0  34259250.0 -118784561.0   \n",
       "323630           2015.0     87757.0           3000.0  34260126.0 -118668308.0   \n",
       "527787           2015.0    166065.0           8235.0  34286002.0 -119191085.0   \n",
       "1249078          2015.0    322751.0          10450.0  34279214.0 -119280606.0   \n",
       "2078569          2015.0     89438.0           9027.0  34206426.0 -118456235.0   \n",
       "625034           2015.0    168824.0           2450.0  34159017.0 -119220159.0   \n",
       "1432386          2015.0    695626.0         113518.0  34249012.0 -119044177.0   \n",
       "\n",
       "              county  \n",
       "1828910       orange  \n",
       "1486712      ventura  \n",
       "1410568  los_angeles  \n",
       "818347        orange  \n",
       "1572339      ventura  \n",
       "968187        orange  \n",
       "118726       ventura  \n",
       "1758322  los_angeles  \n",
       "1141995      ventura  \n",
       "218          ventura  \n",
       "1432246      ventura  \n",
       "1626371      ventura  \n",
       "560004       ventura  \n",
       "1841208      ventura  \n",
       "1001417       orange  \n",
       "713420   los_angeles  \n",
       "1981286      ventura  \n",
       "323630       ventura  \n",
       "527787       ventura  \n",
       "1249078      ventura  \n",
       "2078569  los_angeles  \n",
       "625034       ventura  \n",
       "1432386      ventura  "
      ]
     },
     "execution_count": 29,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train[train.assessmentyear == 2015]"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "86789b17",
   "metadata": {},
   "source": [
    "I think I should look at dropping assessment year 2015 as there are not many records, but the mean value is noteably different. There may have been different protcols to assess value. Perhaps something to research"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "id": "9091b37c",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<seaborn.axisgrid.FacetGrid at 0x7f833f554070>"
      ]
     },
     "execution_count": 30,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 360x360 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.relplot(x=\"bedrooms\", y=\"bathrooms\", data=train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "id": "378afe9a",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "Int64Index: 1186424 entries, 451021 to 457514\n",
      "Data columns (total 13 columns):\n",
      " #   Column           Non-Null Count    Dtype  \n",
      "---  ------           --------------    -----  \n",
      " 0   bedrooms         1186424 non-null  float64\n",
      " 1   bathrooms        1186424 non-null  float64\n",
      " 2   square_feet      1186424 non-null  int64  \n",
      " 3   value            1186424 non-null  int64  \n",
      " 4   year             1186424 non-null  int64  \n",
      " 5   tax              1186424 non-null  float64\n",
      " 6   fips             1186424 non-null  int64  \n",
      " 7   assessmentyear   1186424 non-null  float64\n",
      " 8   land_value       1186424 non-null  float64\n",
      " 9   lot_square_feet  1186424 non-null  float64\n",
      " 10  latitude         1186424 non-null  float64\n",
      " 11  longitude        1186424 non-null  float64\n",
      " 12  county           1186424 non-null  object \n",
      "dtypes: float64(8), int64(4), object(1)\n",
      "memory usage: 159.0+ MB\n"
     ]
    }
   ],
   "source": [
    "train.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "2d52ba11",
   "metadata": {},
   "outputs": [],
   "source": [
    "alpha = .05\n",
    "r, p = stats.spearmanr(train.square_feet, train.lot_square_feet)\n",
    "r, p"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "01f9691d",
   "metadata": {},
   "outputs": [],
   "source": [
    "stats.spearmanr(train.value, train.lot_square_feet)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "a31e491f",
   "metadata": {},
   "outputs": [],
   "source": [
    "stats.spearmanr(train.value, train.square_feet)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "f4d287b8",
   "metadata": {},
   "outputs": [],
   "source": [
    "stats.spearmanr(train.value, train.tax)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "61346301",
   "metadata": {},
   "outputs": [],
   "source": [
    "stats.spearmanr(train.value, train.land_value)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "4ae2f1bc",
   "metadata": {},
   "outputs": [],
   "source": [
    "stats.spearmanr(train.bedrooms, train.bathrooms)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "773ae05f",
   "metadata": {},
   "outputs": [],
   "source": [
    "stats.spearmanr(train.bedrooms, train.square_feet)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "66849a45",
   "metadata": {},
   "outputs": [],
   "source": [
    "stats.spearmanr(train.bathrooms, train.square_feet)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "ead76f97",
   "metadata": {},
   "outputs": [],
   "source": [
    "stats.spearmanr(train.tax, train.land_value)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "e2f7ac17",
   "metadata": {},
   "outputs": [],
   "source": [
    "stats.spearmanr(train.lot_square_feet, train.land_value)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "94f16260",
   "metadata": {},
   "outputs": [],
   "source": [
    "stats.spearmanr(train.year, train.land_value)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "4a100265",
   "metadata": {},
   "outputs": [],
   "source": [
    "stats.spearmanr(train.year, train.value)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "ac55900f",
   "metadata": {},
   "outputs": [],
   "source": [
    "vv = train[train.county == 'ventura'].value\n",
    "lav = train[train.county == 'los_angeles'].value\n",
    "ov = train[train.county == 'orange'].value\n",
    "diffv = train[train.county != 'ventura'].value\n",
    "diffl = train[train.county != 'los_angeles'].value\n",
    "diffo = train[train.county != 'orange'].value"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "2c606adc",
   "metadata": {},
   "outputs": [],
   "source": [
    "vv.var(), lav.var(), ov.var()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "fb26e19c",
   "metadata": {},
   "outputs": [],
   "source": [
    "stats.kruskal(vv, lav, ov)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "722f0f55",
   "metadata": {},
   "outputs": [],
   "source": [
    "stats.f_oneway(vv, lav, ov)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "a3629f25",
   "metadata": {},
   "outputs": [],
   "source": [
    "t, p = stats.ttest_ind(vv, diffv, equal_var=False)\n",
    "t, p"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "9296faee",
   "metadata": {},
   "outputs": [],
   "source": [
    "t, p = stats.ttest_ind(lav, diffl, equal_var=False)\n",
    "t, p"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "99a181ed",
   "metadata": {},
   "outputs": [],
   "source": [
    "t, p = stats.ttest_ind(ov, diffo, equal_var=False)\n",
    "t, p"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "9b22f3b3",
   "metadata": {},
   "outputs": [],
   "source": [
    "train.county.value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "bde45fc4",
   "metadata": {},
   "outputs": [],
   "source": [
    "vt = train[train.county == 'ventura'].tax\n",
    "lat = train[train.county == 'los_angeles'].tax\n",
    "ot = train[train.county == 'orange'].tax\n",
    "diffvt = train[train.county != 'ventura'].tax\n",
    "difflat = train[train.county != 'los_angeles'].tax\n",
    "diffot = train[train.county != 'orange'].tax"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "26da91a7",
   "metadata": {},
   "outputs": [],
   "source": [
    "t, p = stats.ttest_ind(vt, diffvt, equal_var=False)\n",
    "t, p"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "20c9400b",
   "metadata": {},
   "outputs": [],
   "source": [
    "t, p = stats.ttest_ind(lat, difflat, equal_var=False)\n",
    "t, p"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "aa47197b",
   "metadata": {},
   "outputs": [],
   "source": [
    "t, p = stats.ttest_ind(ot, diffot, equal_var=False)\n",
    "t, p"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "9f657475",
   "metadata": {},
   "outputs": [],
   "source": [
    "vv = train[train.county == 'ventura'].value\n",
    "lav = train[train.county == 'los_angeles'].value\n",
    "ov = train[train.county == 'orange'].value"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "6853c785",
   "metadata": {},
   "outputs": [],
   "source": [
    "vv.std(), lav.std(), ov.std()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "72d5dbf3",
   "metadata": {},
   "outputs": [],
   "source": [
    "vv.mean(), lav.mean(), ov.mean()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "fa83c974",
   "metadata": {},
   "outputs": [],
   "source": [
    "vv.min(), lav.min(), ov.min()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "60b406c4",
   "metadata": {},
   "outputs": [],
   "source": [
    "vv.max(), lav.max(), ov.max()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "29d2c880",
   "metadata": {},
   "outputs": [],
   "source": [
    "vt = train[train.county == 'ventura'].tax\n",
    "lat = train[train.county == 'los_angeles'].tax\n",
    "ot = train[train.county == 'orange'].tax"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "768f2c87",
   "metadata": {},
   "outputs": [],
   "source": [
    "vt.std(), lat.std(), ot.std()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "ad98f296",
   "metadata": {},
   "outputs": [],
   "source": [
    "vt.mean(), lat.mean(), ot.mean()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "a4b611ec",
   "metadata": {},
   "outputs": [],
   "source": [
    "vt.min(), lat.min(), ot.min()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "c48dcb2f",
   "metadata": {},
   "outputs": [],
   "source": [
    "vt.max(), lat.max(), ot.max()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "7a993814",
   "metadata": {},
   "outputs": [],
   "source": [
    "train.groupby(train.county == 'ventura').mean()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "f08fde93",
   "metadata": {},
   "outputs": [],
   "source": [
    "vl = train[train.county == 'ventura'].land_value\n",
    "lal = train[train.county == 'los_angeles'].land_value\n",
    "ol = train[train.county == 'orange'].land_value\n",
    "notvl = train[train.county != 'ventura'].land_value\n",
    "notlal = train[train.county != 'los_angeles'].land_value\n",
    "notol = train[train.county != 'orange'].land_value\n",
    "overall = train.land_value.mean()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "c06dfac0",
   "metadata": {},
   "outputs": [],
   "source": [
    "t, p = stats.ttest_ind(vl, notvl, equal_var=False)\n",
    "t, p"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "d77d27b3",
   "metadata": {},
   "outputs": [],
   "source": [
    "t, p = stats.ttest_ind(lal, notlal, equal_var=False)\n",
    "t, p"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "519ab88a",
   "metadata": {},
   "outputs": [],
   "source": [
    "t, p = stats.ttest_ind(ol, notol, equal_var=False)\n",
    "t, p"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "1e386db6",
   "metadata": {},
   "outputs": [],
   "source": [
    "df.county.value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "101b6ac5",
   "metadata": {},
   "outputs": [],
   "source": [
    "train.groupby(train.county == 'ventura').mean()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "id": "a31761ad",
   "metadata": {},
   "outputs": [],
   "source": [
    "import geopy\n",
    "\n",
    "# def get_zipcode(train, geolocator, latitude longitude):\n",
    "#     try:\n",
    "#         location = geolocator.reverse((df[lat_field], df[lon_field]))\n",
    "#         return location.raw['address']['postcode']\n",
    "#     except (AttributeError, KeyError, ValueError):\n",
    "#         print(repr(e), df[lat_field], df[lon_field])\n",
    "#         return None"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 84,
   "id": "82094739",
   "metadata": {},
   "outputs": [],
   "source": [
    "def get_zipcode(train, geolocator, long_lat):\n",
    "        location = geolocator.reverse((train['latitude'], train['longitude']))\n",
    "        return location.raw['address']['postcode']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 89,
   "id": "f84bbfd1",
   "metadata": {},
   "outputs": [],
   "source": [
    "geolocator = geopy.Nominatim(user_agent=\"mshiben\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 65,
   "id": "4a655bce",
   "metadata": {},
   "outputs": [],
   "source": [
    "from shapely.geometry import Point"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 67,
   "id": "af57a7a7",
   "metadata": {},
   "outputs": [],
   "source": [
    "import fiona"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 77,
   "id": "805b798a",
   "metadata": {},
   "outputs": [],
   "source": [
    "import shapely"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 90,
   "id": "90e4b72f",
   "metadata": {},
   "outputs": [],
   "source": [
    "train[\"long_lat\"] = list(zip(train[\"longitude\"], train[\"latitude\"]))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 113,
   "id": "eeb1f51d",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "451021     -117812468.0\n",
       "2047707    -118158914.0\n",
       "48623      -118129195.0\n",
       "919293     -118247790.0\n",
       "254952     -118039414.0\n",
       "               ...     \n",
       "102510     -118151030.0\n",
       "1976504    -117941158.0\n",
       "1202344    -117951411.0\n",
       "1632452    -117977301.0\n",
       "457514     -118104928.0\n",
       "Name: longitude, Length: 1186424, dtype: object"
      ]
     },
     "execution_count": 113,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "long = train.longitude.astype(str)\n",
    "\n",
    "long"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 145,
   "id": "1b7f8a70",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>latitude</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>451021</th>\n",
       "      <td>33634776.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2047707</th>\n",
       "      <td>34715061.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>48623</th>\n",
       "      <td>33812878.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>919293</th>\n",
       "      <td>34071617.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>254952</th>\n",
       "      <td>34086298.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>102510</th>\n",
       "      <td>33850953.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1976504</th>\n",
       "      <td>33949457.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1202344</th>\n",
       "      <td>34049046.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1632452</th>\n",
       "      <td>34002291.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>457514</th>\n",
       "      <td>33962004.0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>1186424 rows × 1 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "           latitude\n",
       "451021   33634776.0\n",
       "2047707  34715061.0\n",
       "48623    33812878.0\n",
       "919293   34071617.0\n",
       "254952   34086298.0\n",
       "...             ...\n",
       "102510   33850953.0\n",
       "1976504  33949457.0\n",
       "1202344  34049046.0\n",
       "1632452  34002291.0\n",
       "457514   33962004.0\n",
       "\n",
       "[1186424 rows x 1 columns]"
      ]
     },
     "execution_count": 145,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "lat = pd.DataFrame(train['latitude'].astype(str))\n",
    "lat"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 146,
   "id": "5109618e",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>latitude</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>451021</th>\n",
       "      <td>33.634776.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2047707</th>\n",
       "      <td>34.715061.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>48623</th>\n",
       "      <td>33.812878.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>919293</th>\n",
       "      <td>34.071617.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>254952</th>\n",
       "      <td>34.086298.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>102510</th>\n",
       "      <td>33.850953.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1976504</th>\n",
       "      <td>33.949457.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1202344</th>\n",
       "      <td>34.049046.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1632452</th>\n",
       "      <td>34.002291.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>457514</th>\n",
       "      <td>33.962004.0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>1186424 rows × 1 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "            latitude\n",
       "451021   33.634776.0\n",
       "2047707  34.715061.0\n",
       "48623    33.812878.0\n",
       "919293   34.071617.0\n",
       "254952   34.086298.0\n",
       "...              ...\n",
       "102510   33.850953.0\n",
       "1976504  33.949457.0\n",
       "1202344  34.049046.0\n",
       "1632452  34.002291.0\n",
       "457514   33.962004.0\n",
       "\n",
       "[1186424 rows x 1 columns]"
      ]
     },
     "execution_count": 146,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "for c in lat:\n",
    "     lat[c] = (lat[c].str[:2] + '.' + lat[c].str[2:])\n",
    "lat"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 194,
   "id": "7594ae79",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>longitude</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>451021</th>\n",
       "      <td>-117812468.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2047707</th>\n",
       "      <td>-118158914.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>48623</th>\n",
       "      <td>-118129195.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>919293</th>\n",
       "      <td>-118247790.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>254952</th>\n",
       "      <td>-118039414.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>102510</th>\n",
       "      <td>-118151030.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1976504</th>\n",
       "      <td>-117941158.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1202344</th>\n",
       "      <td>-117951411.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1632452</th>\n",
       "      <td>-117977301.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>457514</th>\n",
       "      <td>-118104928.0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>1186424 rows × 1 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "            longitude\n",
       "451021   -117812468.0\n",
       "2047707  -118158914.0\n",
       "48623    -118129195.0\n",
       "919293   -118247790.0\n",
       "254952   -118039414.0\n",
       "...               ...\n",
       "102510   -118151030.0\n",
       "1976504  -117941158.0\n",
       "1202344  -117951411.0\n",
       "1632452  -117977301.0\n",
       "457514   -118104928.0\n",
       "\n",
       "[1186424 rows x 1 columns]"
      ]
     },
     "execution_count": 194,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "long = pd.DataFrame(train['longitude'].astype(str))\n",
    "long"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 195,
   "id": "94d8fa9a",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>longitude</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>451021</th>\n",
       "      <td>-117.17812468.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2047707</th>\n",
       "      <td>-118.18158914.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>48623</th>\n",
       "      <td>-118.18129195.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>919293</th>\n",
       "      <td>-118.18247790.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>254952</th>\n",
       "      <td>-118.18039414.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>102510</th>\n",
       "      <td>-118.18151030.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1976504</th>\n",
       "      <td>-117.17941158.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1202344</th>\n",
       "      <td>-117.17951411.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1632452</th>\n",
       "      <td>-117.17977301.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>457514</th>\n",
       "      <td>-118.18104928.0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>1186424 rows × 1 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "               longitude\n",
       "451021   -117.17812468.0\n",
       "2047707  -118.18158914.0\n",
       "48623    -118.18129195.0\n",
       "919293   -118.18247790.0\n",
       "254952   -118.18039414.0\n",
       "...                  ...\n",
       "102510   -118.18151030.0\n",
       "1976504  -117.17941158.0\n",
       "1202344  -117.17951411.0\n",
       "1632452  -117.17977301.0\n",
       "457514   -118.18104928.0\n",
       "\n",
       "[1186424 rows x 1 columns]"
      ]
     },
     "execution_count": 195,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "for c in long:\n",
    "     long[c] = (long[c].str[:4] + '.' + long[c].str[2:])\n",
    "long"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 201,
   "id": "0af6ae97",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "451021     -117.17812468\n",
       "2047707    -118.18158914\n",
       "48623      -118.18129195\n",
       "919293      -118.1824779\n",
       "254952     -118.18039414\n",
       "               ...      \n",
       "102510      -118.1815103\n",
       "1976504    -117.17941158\n",
       "1202344    -117.17951411\n",
       "1632452    -117.17977301\n",
       "457514     -118.18104928\n",
       "Name: longitude, Length: 1186424, dtype: object"
      ]
     },
     "execution_count": 201,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "long = long.longitude.str.rstrip('.0') \n",
    "long"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 202,
   "id": "16039eb9",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "451021     33.634776\n",
       "2047707    34.715061\n",
       "48623      33.812878\n",
       "919293     34.071617\n",
       "254952     34.086298\n",
       "             ...    \n",
       "102510     33.850953\n",
       "1976504    33.949457\n",
       "1202344    34.049046\n",
       "1632452    34.002291\n",
       "457514     33.962004\n",
       "Name: latitude, Length: 1186424, dtype: object"
      ]
     },
     "execution_count": 202,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "lat = lat.latitude.str.rstrip('.0')\n",
    "lat"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 153,
   "id": "1b99c2df",
   "metadata": {},
   "outputs": [
    {
     "ename": "TypeError",
     "evalue": "reverse() takes 2 positional arguments but 3 were given",
     "output_type": "error",
     "traceback": [
      "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[0;31mTypeError\u001b[0m                                 Traceback (most recent call last)",
      "\u001b[0;32m/var/folders/s4/hmz6ljm533vgpm_bhv59yw0m0000gn/T/ipykernel_4524/2857273035.py\u001b[0m in \u001b[0;36m<module>\u001b[0;34m\u001b[0m\n\u001b[1;32m      1\u001b[0m \u001b[0mgeo_locator\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mgeopy\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mNominatim\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0muser_agent\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;34m'mshiben'\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      2\u001b[0m                         \u001b[0;31m# Latitude, Longitude\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m----> 3\u001b[0;31m \u001b[0mgeolocator\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mreverse\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m\"-118.151030\"\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m\"33.850953\"\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m",
      "\u001b[0;31mTypeError\u001b[0m: reverse() takes 2 positional arguments but 3 were given"
     ]
    }
   ],
   "source": [
    "geo_locator = geopy.Nominatim(user_agent='mshiben')\n",
    "                        # Latitude, Longitude"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 156,
   "id": "fcf8d545",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Location(Downey Avenue, Lakewood, Los Angeles County, California, 90712, United States, (33.850952807641285, -118.15098112973604, 0.0))"
      ]
     },
     "execution_count": 156,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "geolocator.reverse([33.850953, -118.151030])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 205,
   "id": "99927709",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "451021     33.634776\n",
       "2047707    34.715061\n",
       "48623      33.812878\n",
       "919293     34.071617\n",
       "254952     34.086298\n",
       "             ...    \n",
       "102510     33.850953\n",
       "1976504    33.949457\n",
       "1202344    34.049046\n",
       "1632452    34.002291\n",
       "457514     33.962004\n",
       "Name: latitude, Length: 1186424, dtype: float64"
      ]
     },
     "execution_count": 205,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "lat.astype('float')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 206,
   "id": "e37570a5",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "451021    -117.178125\n",
       "2047707   -118.181589\n",
       "48623     -118.181292\n",
       "919293    -118.182478\n",
       "254952    -118.180394\n",
       "              ...    \n",
       "102510    -118.181510\n",
       "1976504   -117.179412\n",
       "1202344   -117.179514\n",
       "1632452   -117.179773\n",
       "457514    -118.181049\n",
       "Name: longitude, Length: 1186424, dtype: float64"
      ]
     },
     "execution_count": 206,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "long.astype('float')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 208,
   "id": "9058afb0",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([['33.634776', '-117.17812468'],\n",
       "       ['34.715061', '-118.18158914'],\n",
       "       ['33.812878', '-118.18129195'],\n",
       "       ...,\n",
       "       ['34.049046', '-117.17951411'],\n",
       "       ['34.002291', '-117.17977301'],\n",
       "       ['33.962004', '-118.18104928']], dtype=object)"
      ]
     },
     "execution_count": 208,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Creating a zip with latitudes and longitudes\n",
    "coords = np.stack((lat, long), axis = 1)\n",
    "coords"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 226,
   "id": "3568b58d",
   "metadata": {},
   "outputs": [
    {
     "ename": "AttributeError",
     "evalue": "'NoneType' object has no attribute 'raw'",
     "output_type": "error",
     "traceback": [
      "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[0;31mAttributeError\u001b[0m                            Traceback (most recent call last)",
      "\u001b[0;32m/var/folders/s4/hmz6ljm533vgpm_bhv59yw0m0000gn/T/ipykernel_4524/4196061166.py\u001b[0m in \u001b[0;36m<module>\u001b[0;34m\u001b[0m\n\u001b[1;32m      1\u001b[0m \u001b[0mlocation\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mgeolocator\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mgeocode\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mcoords\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m----> 2\u001b[0;31m \u001b[0mlocation\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mraw\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0maddress\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0mpostcode\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m",
      "\u001b[0;31mAttributeError\u001b[0m: 'NoneType' object has no attribute 'raw'"
     ]
    }
   ],
   "source": [
    "location = geolocator.geocode(coords)\n",
    "location.raw[address][postcode]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 237,
   "id": "e75ef08a",
   "metadata": {},
   "outputs": [
    {
     "ename": "AttributeError",
     "evalue": "'Nominatim' object has no attribute 'getlocation'",
     "output_type": "error",
     "traceback": [
      "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[0;31mAttributeError\u001b[0m                            Traceback (most recent call last)",
      "\u001b[0;32m/var/folders/s4/hmz6ljm533vgpm_bhv59yw0m0000gn/T/ipykernel_4524/568089287.py\u001b[0m in \u001b[0;36m<module>\u001b[0;34m\u001b[0m\n\u001b[0;32m----> 1\u001b[0;31m geolocator.getlocation.getCurrentPosition(enableHighAccuracy= False,\n\u001b[0m\u001b[1;32m      2\u001b[0m \u001b[0mtimeout\u001b[0m\u001b[0;34m=\u001b[0m \u001b[0;36m5000\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      3\u001b[0m maximumAge= Infinity)\n",
      "\u001b[0;31mAttributeError\u001b[0m: 'Nominatim' object has no attribute 'getlocation'"
     ]
    }
   ],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 238,
   "id": "3ba2a1d2",
   "metadata": {},
   "outputs": [
    {
     "ename": "KeyboardInterrupt",
     "evalue": "",
     "output_type": "error",
     "traceback": [
      "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[0;31mKeyboardInterrupt\u001b[0m                         Traceback (most recent call last)",
      "\u001b[0;32m/var/folders/s4/hmz6ljm533vgpm_bhv59yw0m0000gn/T/ipykernel_4524/3468745673.py\u001b[0m in \u001b[0;36m<module>\u001b[0;34m\u001b[0m\n\u001b[1;32m      1\u001b[0m \u001b[0mzipcode\u001b[0m \u001b[0;34m=\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      2\u001b[0m \u001b[0;32mfor\u001b[0m \u001b[0mi\u001b[0m \u001b[0;32min\u001b[0m \u001b[0mrange\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mlen\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mcoords\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m----> 3\u001b[0;31m     \u001b[0mlocation\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mgeolocator\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mreverse\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mcoords\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0mi\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m      4\u001b[0m     \u001b[0mzipcode\u001b[0m \u001b[0;34m=\u001b[0m\u001b[0mlocation\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mraw\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      5\u001b[0m \u001b[0;31m#Creating dataframe with all the addresses\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m/usr/local/anaconda3/lib/python3.9/site-packages/geopy/geocoders/nominatim.py\u001b[0m in \u001b[0;36mreverse\u001b[0;34m(self, query, exactly_one, timeout, language, addressdetails, zoom)\u001b[0m\n\u001b[1;32m    360\u001b[0m         \u001b[0mlogger\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mdebug\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m\"%s.reverse: %s\"\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m__class__\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m__name__\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0murl\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    361\u001b[0m         \u001b[0mcallback\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mpartial\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_parse_json\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mexactly_one\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mexactly_one\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 362\u001b[0;31m         \u001b[0;32mreturn\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_call_geocoder\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0murl\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mcallback\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mtimeout\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mtimeout\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    363\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    364\u001b[0m     \u001b[0;32mdef\u001b[0m \u001b[0m_parse_code\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mplace\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m/usr/local/anaconda3/lib/python3.9/site-packages/geopy/geocoders/base.py\u001b[0m in \u001b[0;36m_call_geocoder\u001b[0;34m(self, url, callback, timeout, is_json, headers)\u001b[0m\n\u001b[1;32m    366\u001b[0m         \u001b[0;32mtry\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    367\u001b[0m             \u001b[0;32mif\u001b[0m \u001b[0mis_json\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 368\u001b[0;31m                 \u001b[0mresult\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0madapter\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mget_json\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0murl\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mtimeout\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mtimeout\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mheaders\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mreq_headers\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    369\u001b[0m             \u001b[0;32melse\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    370\u001b[0m                 \u001b[0mresult\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0madapter\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mget_text\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0murl\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mtimeout\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mtimeout\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mheaders\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mreq_headers\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m/usr/local/anaconda3/lib/python3.9/site-packages/geopy/adapters.py\u001b[0m in \u001b[0;36mget_json\u001b[0;34m(self, url, timeout, headers)\u001b[0m\n\u001b[1;32m    436\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    437\u001b[0m     \u001b[0;32mdef\u001b[0m \u001b[0mget_json\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0murl\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m*\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mtimeout\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mheaders\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 438\u001b[0;31m         \u001b[0mresp\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_request\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0murl\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mtimeout\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mtimeout\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mheaders\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mheaders\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    439\u001b[0m         \u001b[0;32mtry\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    440\u001b[0m             \u001b[0;32mreturn\u001b[0m \u001b[0mresp\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mjson\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m/usr/local/anaconda3/lib/python3.9/site-packages/geopy/adapters.py\u001b[0m in \u001b[0;36m_request\u001b[0;34m(self, url, timeout, headers)\u001b[0m\n\u001b[1;32m    446\u001b[0m     \u001b[0;32mdef\u001b[0m \u001b[0m_request\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0murl\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m*\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mtimeout\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mheaders\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    447\u001b[0m         \u001b[0;32mtry\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 448\u001b[0;31m             \u001b[0mresp\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0msession\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mget\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0murl\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mtimeout\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mtimeout\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mheaders\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mheaders\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    449\u001b[0m         \u001b[0;32mexcept\u001b[0m \u001b[0mException\u001b[0m \u001b[0;32mas\u001b[0m \u001b[0merror\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    450\u001b[0m             \u001b[0mmessage\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mstr\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0merror\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m/usr/local/anaconda3/lib/python3.9/site-packages/requests/sessions.py\u001b[0m in \u001b[0;36mget\u001b[0;34m(self, url, **kwargs)\u001b[0m\n\u001b[1;32m    553\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    554\u001b[0m         \u001b[0mkwargs\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0msetdefault\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m'allow_redirects'\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;32mTrue\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 555\u001b[0;31m         \u001b[0;32mreturn\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mrequest\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m'GET'\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0murl\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m**\u001b[0m\u001b[0mkwargs\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    556\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    557\u001b[0m     \u001b[0;32mdef\u001b[0m \u001b[0moptions\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0murl\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m**\u001b[0m\u001b[0mkwargs\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m/usr/local/anaconda3/lib/python3.9/site-packages/requests/sessions.py\u001b[0m in \u001b[0;36mrequest\u001b[0;34m(self, method, url, params, data, headers, cookies, files, auth, timeout, allow_redirects, proxies, hooks, stream, verify, cert, json)\u001b[0m\n\u001b[1;32m    540\u001b[0m         }\n\u001b[1;32m    541\u001b[0m         \u001b[0msend_kwargs\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mupdate\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0msettings\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 542\u001b[0;31m         \u001b[0mresp\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0msend\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mprep\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m**\u001b[0m\u001b[0msend_kwargs\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    543\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    544\u001b[0m         \u001b[0;32mreturn\u001b[0m \u001b[0mresp\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m/usr/local/anaconda3/lib/python3.9/site-packages/requests/sessions.py\u001b[0m in \u001b[0;36msend\u001b[0;34m(self, request, **kwargs)\u001b[0m\n\u001b[1;32m    653\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    654\u001b[0m         \u001b[0;31m# Send the request\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 655\u001b[0;31m         \u001b[0mr\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0madapter\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0msend\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mrequest\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m**\u001b[0m\u001b[0mkwargs\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    656\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    657\u001b[0m         \u001b[0;31m# Total elapsed time of the request (approximately)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m/usr/local/anaconda3/lib/python3.9/site-packages/requests/adapters.py\u001b[0m in \u001b[0;36msend\u001b[0;34m(self, request, stream, timeout, verify, cert, proxies)\u001b[0m\n\u001b[1;32m    437\u001b[0m         \u001b[0;32mtry\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    438\u001b[0m             \u001b[0;32mif\u001b[0m \u001b[0;32mnot\u001b[0m \u001b[0mchunked\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 439\u001b[0;31m                 resp = conn.urlopen(\n\u001b[0m\u001b[1;32m    440\u001b[0m                     \u001b[0mmethod\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mrequest\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mmethod\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    441\u001b[0m                     \u001b[0murl\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0murl\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m/usr/local/anaconda3/lib/python3.9/site-packages/urllib3/connectionpool.py\u001b[0m in \u001b[0;36murlopen\u001b[0;34m(self, method, url, body, headers, retries, redirect, assert_same_host, timeout, pool_timeout, release_conn, chunked, body_pos, **response_kw)\u001b[0m\n\u001b[1;32m    697\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    698\u001b[0m             \u001b[0;31m# Make the request on the httplib connection object.\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 699\u001b[0;31m             httplib_response = self._make_request(\n\u001b[0m\u001b[1;32m    700\u001b[0m                 \u001b[0mconn\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    701\u001b[0m                 \u001b[0mmethod\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m/usr/local/anaconda3/lib/python3.9/site-packages/urllib3/connectionpool.py\u001b[0m in \u001b[0;36m_make_request\u001b[0;34m(self, conn, method, url, timeout, chunked, **httplib_request_kw)\u001b[0m\n\u001b[1;32m    443\u001b[0m                     \u001b[0;31m# Python 3 (including for exceptions like SystemExit).\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    444\u001b[0m                     \u001b[0;31m# Otherwise it looks like a bug in the code.\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 445\u001b[0;31m                     \u001b[0msix\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mraise_from\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0me\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;32mNone\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    446\u001b[0m         \u001b[0;32mexcept\u001b[0m \u001b[0;34m(\u001b[0m\u001b[0mSocketTimeout\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mBaseSSLError\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mSocketError\u001b[0m\u001b[0;34m)\u001b[0m \u001b[0;32mas\u001b[0m \u001b[0me\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    447\u001b[0m             \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_raise_timeout\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0merr\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0me\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0murl\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0murl\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mtimeout_value\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mread_timeout\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m/usr/local/anaconda3/lib/python3.9/site-packages/urllib3/packages/six.py\u001b[0m in \u001b[0;36mraise_from\u001b[0;34m(value, from_value)\u001b[0m\n",
      "\u001b[0;32m/usr/local/anaconda3/lib/python3.9/site-packages/urllib3/connectionpool.py\u001b[0m in \u001b[0;36m_make_request\u001b[0;34m(self, conn, method, url, timeout, chunked, **httplib_request_kw)\u001b[0m\n\u001b[1;32m    438\u001b[0m                 \u001b[0;31m# Python 3\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    439\u001b[0m                 \u001b[0;32mtry\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 440\u001b[0;31m                     \u001b[0mhttplib_response\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mconn\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mgetresponse\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    441\u001b[0m                 \u001b[0;32mexcept\u001b[0m \u001b[0mBaseException\u001b[0m \u001b[0;32mas\u001b[0m \u001b[0me\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    442\u001b[0m                     \u001b[0;31m# Remove the TypeError from the exception chain in\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m/usr/local/anaconda3/lib/python3.9/http/client.py\u001b[0m in \u001b[0;36mgetresponse\u001b[0;34m(self)\u001b[0m\n\u001b[1;32m   1369\u001b[0m         \u001b[0;32mtry\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   1370\u001b[0m             \u001b[0;32mtry\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m-> 1371\u001b[0;31m                 \u001b[0mresponse\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mbegin\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m   1372\u001b[0m             \u001b[0;32mexcept\u001b[0m \u001b[0mConnectionError\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   1373\u001b[0m                 \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mclose\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m/usr/local/anaconda3/lib/python3.9/http/client.py\u001b[0m in \u001b[0;36mbegin\u001b[0;34m(self)\u001b[0m\n\u001b[1;32m    317\u001b[0m         \u001b[0;31m# read until we get a non-100 response\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    318\u001b[0m         \u001b[0;32mwhile\u001b[0m \u001b[0;32mTrue\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 319\u001b[0;31m             \u001b[0mversion\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mstatus\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mreason\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_read_status\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    320\u001b[0m             \u001b[0;32mif\u001b[0m \u001b[0mstatus\u001b[0m \u001b[0;34m!=\u001b[0m \u001b[0mCONTINUE\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    321\u001b[0m                 \u001b[0;32mbreak\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m/usr/local/anaconda3/lib/python3.9/http/client.py\u001b[0m in \u001b[0;36m_read_status\u001b[0;34m(self)\u001b[0m\n\u001b[1;32m    278\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    279\u001b[0m     \u001b[0;32mdef\u001b[0m \u001b[0m_read_status\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 280\u001b[0;31m         \u001b[0mline\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mstr\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mfp\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mreadline\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0m_MAXLINE\u001b[0m \u001b[0;34m+\u001b[0m \u001b[0;36m1\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m\"iso-8859-1\"\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    281\u001b[0m         \u001b[0;32mif\u001b[0m \u001b[0mlen\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mline\u001b[0m\u001b[0;34m)\u001b[0m \u001b[0;34m>\u001b[0m \u001b[0m_MAXLINE\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    282\u001b[0m             \u001b[0;32mraise\u001b[0m \u001b[0mLineTooLong\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m\"status line\"\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m/usr/local/anaconda3/lib/python3.9/socket.py\u001b[0m in \u001b[0;36mreadinto\u001b[0;34m(self, b)\u001b[0m\n\u001b[1;32m    702\u001b[0m         \u001b[0;32mwhile\u001b[0m \u001b[0;32mTrue\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    703\u001b[0m             \u001b[0;32mtry\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 704\u001b[0;31m                 \u001b[0;32mreturn\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_sock\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mrecv_into\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mb\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    705\u001b[0m             \u001b[0;32mexcept\u001b[0m \u001b[0mtimeout\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    706\u001b[0m                 \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_timeout_occurred\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0;32mTrue\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m/usr/local/anaconda3/lib/python3.9/ssl.py\u001b[0m in \u001b[0;36mrecv_into\u001b[0;34m(self, buffer, nbytes, flags)\u001b[0m\n\u001b[1;32m   1239\u001b[0m                   \u001b[0;34m\"non-zero flags not allowed in calls to recv_into() on %s\"\u001b[0m \u001b[0;34m%\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   1240\u001b[0m                   self.__class__)\n\u001b[0;32m-> 1241\u001b[0;31m             \u001b[0;32mreturn\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mread\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mnbytes\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mbuffer\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m   1242\u001b[0m         \u001b[0;32melse\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   1243\u001b[0m             \u001b[0;32mreturn\u001b[0m \u001b[0msuper\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mrecv_into\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mbuffer\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mnbytes\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mflags\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m/usr/local/anaconda3/lib/python3.9/ssl.py\u001b[0m in \u001b[0;36mread\u001b[0;34m(self, len, buffer)\u001b[0m\n\u001b[1;32m   1097\u001b[0m         \u001b[0;32mtry\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   1098\u001b[0m             \u001b[0;32mif\u001b[0m \u001b[0mbuffer\u001b[0m \u001b[0;32mis\u001b[0m \u001b[0;32mnot\u001b[0m \u001b[0;32mNone\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m-> 1099\u001b[0;31m                 \u001b[0;32mreturn\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_sslobj\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mread\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mlen\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mbuffer\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m   1100\u001b[0m             \u001b[0;32melse\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   1101\u001b[0m                 \u001b[0;32mreturn\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_sslobj\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mread\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mlen\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;31mKeyboardInterrupt\u001b[0m: "
     ]
    }
   ],
   "source": [
    "zipcode =[]\n",
    "for i in range(len(coords)):\n",
    "    location = geolocator.reverse(coords[i])\n",
    "    zipcode =location.raw\n",
    "#Creating dataframe with all the addresses\n",
    "zipcodes =pd.DataFrame(data=zipcode)\n",
    "zipcodes\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 92,
   "id": "39b7e11e",
   "metadata": {},
   "outputs": [
    {
     "ename": "SyntaxError",
     "evalue": "positional argument follows keyword argument (3291110479.py, line 1)",
     "output_type": "error",
     "traceback": [
      "\u001b[0;36m  File \u001b[0;32m\"/var/folders/s4/hmz6ljm533vgpm_bhv59yw0m0000gn/T/ipykernel_4524/3291110479.py\"\u001b[0;36m, line \u001b[0;32m1\u001b[0m\n\u001b[0;31m    zipcodes = train.apply(get_zipcode, geolocator=geolocator, train['long_lat'])\u001b[0m\n\u001b[0m                                                                                ^\u001b[0m\n\u001b[0;31mSyntaxError\u001b[0m\u001b[0;31m:\u001b[0m positional argument follows keyword argument\n"
     ]
    }
   ],
   "source": [
    "zipcodes = train.apply(get_zipcode, geolocator=geolocator, train['long_lat'])"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "0355b80c",
   "metadata": {},
   "source": [
    "sns.swarmplot(x= train.zipcode.where(train.county == 'ventura') , y='value', data=train)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
