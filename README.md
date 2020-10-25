
# Time Series

![Clocks](https://media.giphy.com/media/xTiTnEeKtzw4zJyFsQ/giphy.gif)


```python
from src.student_caller import one_random_student, three_random_students
from src.student_list import student_first_names
```


```python
"Name 3 assumptions of linear regression?"

one_random_student(student_first_names)
```

    William


<a id='section_1'></a>

# Time Series vs. Linear

For linear regression, we attempted to explain the variance of a continuous target variable via a set of **independent predictor features**. We assumed that there was no **autocorrelation** amongst our records.  Whereas multicolinearity describes two features whose linear increase or decrease is correlated,  autocorrelation describes whether there is a relationship between values of the same variable at different times.   


In linear regression, we make the assumption that each record is independent of the others.  In time series models, we make the opposite assumption.  We assume that a given value can best be predicted by its **past values**.

The main idea with time series is to replace our independent features with past values of our target. 

The models we will cover in lecture include endogenous variables.
<em>Endogenous</em> means caused by factors within the system. 

<em>Exogenous</em>, caused by factors outside the system. 

Many statsmodels tools use <tt>endog</tt> to represent the incoming time series data in place of the constant <tt>y</tt>.<br>

For more information visit http://www.statsmodels.org/stable/endog_exog.html

# Applications
> informed by [Practical Time Series Analysis](https://www.oreilly.com/library/view/practical-time-series/9781492041641/), Nielson)


## Healthcare
> With new methods of personalized data collection, the opportunity for time series analysis is growing.  Take health care,  where new wearable technology is producing individualized records of medical data. With a smartwatch or phone, heartrate, bloodpressure, sleep and activity records, can all be recorded easily. All of these datapoints can be timestamped precisely, and easily exported for analysis.  

> Time series are used to predict weekly flu rates

## Finance
> High frequency traders use large quantities to train time series models that trade on the microsecond level. 
> Long term time series look to model over longer periods (hours, days, months) are still relevant and employed by traditional trading firms.

## Government
> Government databases, which serve an important purpose of gathering data related to the wellfare of its citizens, are a rich source for time series data.  These databases contain time series related to:
   - Unemployment
   - Global warming
   - Crime (gun crime will be the example of today's lessons)

## A few examples visualized



```python
import pandas as pd
import numpy as np
pd.set_option('display.max_rows', 1000)
```


```python
# Define a function that will help us load and
# clean up a dataset.

def load_trend(trend_name='football', country_code='us'):
    df = pd.read_csv('data/google-trends_'
                     + trend_name + '_'
                     + country_code
                     + '.csv').iloc[1:, :]
    df.columns = ['counts']
    df['counts'] = df['counts'].str.replace('<1', '0').astype(int)
    return df
```


```python
df = load_trend(**{'trend_name': 'data-science', 'country_code': 'us'})

```


```python
trends = [
    {'trend_name': 'data-science', 'country_code': 'us'},
    {'trend_name': 'football', 'country_code': 'us'},
    {'trend_name': 'football', 'country_code': 'uk'},
    {'trend_name': 'game-of-thrones', 'country_code': 'us'},
    {'trend_name': 'pokemon', 'country_code': 'us'},
    {'trend_name': 'taxes', 'country_code': 'us'},   
]
```


```python
np.random.shuffle(trends)

```


```python
trend_dfs = [load_trend(**trend) for trend in trends]

```


```python
import matplotlib.pyplot as plt
# Let's see if we can guess which is which just by looking
# at their graphs.


import matplotlib; matplotlib.style.use('ggplot')

fig, axs = plt.subplots(len(trend_dfs), 1, figsize=(8, 10))
plt.tight_layout()
for i, trend_df in enumerate(trend_dfs):
    ax = axs[i]
    #ax.set_title(str(trends[i]))
    ax.plot(np.array(trend_df.index), trend_df['counts'])
    ticks = ax.get_xticks()
    ax.set_ylim((0, 100))
    ax.set_xticks([tick for tick in ticks if tick%24 == 0])
```


![png](index_files/index_19_0.png)


## Agenda

1. [Date Time Objects](#section_2)
2. [Time Series Preprocessing Techniques](#section_3)
 - [Resampling](#resampling)
 - [Interpolating](#interpolation)
4. [Components of Time Series Data and Stationarity](#components)
 - [Decomposition](#decomposition)
 - [Stationarity](#stationarity)
 - [Dickey-Fuller](#dickey-fuller)
    

<a id='section_2'></a>

# 1: Datetime objects

Datetime objects make our time series modeling lives easier.  They will allow us to perform essential data prep tasks with a few lines of code.  

We need our time series **index** to be datetime objects, since our models will rely on being able to identify the previous chronological value.

There is a datetime [library](https://docs.python.org/2/library/datetime.html), and inside pandas there is a datetime module as well as a to_datetime() function.


Let's import some data on **gun violence in Chicago**.

[source](https://data.cityofchicago.org/Public-Safety/Gun-Crimes-Heat-Map/iinq-m3rg)


```python
# import the Gun_Crimes_Heat_Map.csv into a data frame
ts = None
```


```python
# inspect the ts dataframe
```

Let's look at some summary stats:


```python
print(f"There are {ts.shape[0]} records in our timeseries")
```

    There are 85267 records in our timeseries



```python
import matplotlib.pyplot as plt
import seaborn as sns

# Let's look at the Description of the different types of reported events
# There is some messy input in this data set.

```


```python
height = ts['Description'].value_counts()[:10]
offense_names = ts['Description'].value_counts()[:10].index

fig, ax = plt.subplots()
sns.barplot(height, offense_names, color='r', ax=ax)
ax.set_title('Mostly Handgun offenses')
```




    Text(0.5, 1.0, 'Mostly Handgun offenses')




![png](index_files/index_30_1.png)



```python
# Let's look at the percentage of events that are related to domestic violence
# by using value counts on the Domestic feature
per_domestic_violence = None
```


```python
# Mostly non-domestic offenses

fig, ax = plt.subplots()
sns.barplot( ts['Domestic'].value_counts().index, 
             ts['Domestic'].value_counts(),  
             palette=[ 'r', 'b'], ax=ax
           )

ax.set_title("Overwhelmingly Non-Domestic Offenses");
```


![png](index_files/index_32_0.png)



```python
# Look at the arrest rates by taking the value counts of the Arrest feature
arrest_rate = None
```


```python
# Just above 30% of the arrests result in arrests

fig, ax = plt.subplots()

sns.barplot( ts['Arrest'].value_counts().index, 
             ts['Arrest'].value_counts(), 
             palette=['r', 'g'], ax=ax
           )

ax.set_title(f'{arrest_rate: .2%} of Total Cases\n Result in Arrest');
```


![png](index_files/index_34_0.png)


The data extracts the year of offense as its own columns.


```python
fig, ax = plt.subplots()
sns.barplot( ts['Year'].value_counts().index, 
             ts['Year'].value_counts(),  
             color= 'r', ax=ax
           )

ax.set_title("Offenses By Year");
```


![png](index_files/index_36_0.png)


While this does show some interesting information that will be relevant to our time series analysis, we are going to get more granular.

# Date Time Objects

For time series modeling, the first step is to make sure that the index is a date time object.


```python
print(f"The original data, if we import with standard read_csv, is a {type(ts.index)}")
```

    The original data, if we import with standard read_csv, is a <class 'pandas.core.indexes.range.RangeIndex'>


There are a few ways to **reindex** our series to datetime. 

We can use the pd.to_datetime() method


```python
# set_index to a datetime index.  
# Set drop = True to drop the original index and inplace=True to modify the dataframe object. 
```

Or, we can parse the dates directly on import


```python
ts =  pd.read_csv('data/Gun_Crimes_Heat_Map.csv', index_col='Date', parse_dates=True)
```


```python
print(f"Now our index is a {type(ts.index)}")
```

    Now our index is a <class 'pandas.core.indexes.datetimes.DatetimeIndex'>


We've covered some of the fun abilities of datetime objects, including being able to extract components of the date like so:


```python
# extract the month component from the index of the first record

```


```python
# extract the year

```


```python
# There are so many cool attributes and methods.  How can we inspect them? Use the ?

```

We can easily see now see whether offenses happen, for example, during business hours.



```python


ts['hour'] = ts.index
ts['hour'] = ts.hour.apply(lambda x: x.hour)
ts['business_hours'] = ts.hour.apply(lambda x: 9 <= x <= 16 )

ts.business_hours.value_counts()
```




    False    60863
    True     24404
    Name: business_hours, dtype: int64




```python
fig, ax = plt.subplots()
bh_ratio = ts.business_hours.value_counts()[1]/len(ts)

x = ts.business_hours.value_counts().index
y = ts.business_hours.value_counts()
sns.barplot(x=x, y=y)

ax.set_title(f'{bh_ratio: .2%} of Offenses\n Happen Btwn 9 and 5')
```




    Text(0.5, 1.0, ' 28.62% of Offenses\n Happen Btwn 9 and 5')




![png](index_files/index_52_1.png)


### With a partner, take five minutes ot play around with the datetime object, and make a plot that answers a time based question about our data.


```python
# What is the distribution of gun crime across different days of the week

```


```python
# What is the distribution of gun crime across different quarters

```


```python
# What is the distribution of gun crime across months

```

![pair](https://media.giphy.com/media/SvulfW0MQncFYzQEMT/giphy.gif)

<a id='section_3'></a>

# 2: Time Series Preprocessing Techniques

<a id='resampling'></a>

## Resampling
We have new abilities associated with the datetime index, such as **resampling**

Resampling allows us to zoom in on or zoom out from the time specification associated with data collection.

For example, our gun data is collected with a time stamp including the minute of the incident.  Of course, we will not be interested in predicting the minute a gun crime occured, so we will eventually zoom out from our data.  


Take a moment to familiarize yourself with the difference between resampling aliases

<table style="display: inline-block">
    <caption style="text-align: center"><strong>TIME SERIES OFFSET ALIASES</strong></caption>
<tr><th>ALIAS</th><th>DESCRIPTION</th></tr>
<tr><td>B</td><td>business day frequency</td></tr>
<tr><td>C</td><td>custom business day frequency (experimental)</td></tr>
<tr><td>D</td><td>calendar day frequency</td></tr>
<tr><td>W</td><td>weekly frequency</td></tr>
<tr><td>M</td><td>month end frequency</td></tr>
<tr><td>SM</td><td>semi-month end frequency (15th and end of month)</td></tr>
<tr><td>BM</td><td>business month end frequency</td></tr>
<tr><td>CBM</td><td>custom business month end frequency</td></tr>
<tr><td>MS</td><td>month start frequency</td></tr>
<tr><td>SMS</td><td>semi-month start frequency (1st and 15th)</td></tr>
<tr><td>BMS</td><td>business month start frequency</td></tr>
<tr><td>CBMS</td><td>custom business month start frequency</td></tr>
<tr><td>Q</td><td>quarter end frequency</td></tr>
<tr><td></td><td><font color=white>intentionally left blank</font></td></tr></table>

<table style="display: inline-block; margin-left: 40px">
<caption style="text-align: center"></caption>
<tr><th>ALIAS</th><th>DESCRIPTION</th></tr>
<tr><td>BQ</td><td>business quarter endfrequency</td></tr>
<tr><td>QS</td><td>quarter start frequency</td></tr>
<tr><td>BQS</td><td>business quarter start frequency</td></tr>
<tr><td>A</td><td>year end frequency</td></tr>
<tr><td>BA</td><td>business year end frequency</td></tr>
<tr><td>AS</td><td>year start frequency</td></tr>
<tr><td>BAS</td><td>business year start frequency</td></tr>
<tr><td>BH</td><td>business hour frequency</td></tr>
<tr><td>H</td><td>hourly frequency</td></tr>
<tr><td>T, min</td><td>minutely frequency</td></tr>
<tr><td>S</td><td>secondly frequency</td></tr>
<tr><td>L, ms</td><td>milliseconds</td></tr>
<tr><td>U, us</td><td>microseconds</td></tr>
<tr><td>N</td><td>nanoseconds</td></tr></table>

**To upsample** is to increase the frequency of the data of interest.  
**To downsample** is to decrease the frequency of the data of interest.

Let's downsample, and create a time series of gun offenses reported per day. 


```python
# Code: Use the resample method with the 'D' parameter
```

When resampling, we have to provide a rule to resample by, and an **aggregate function**.

For our purposes, we will downsample, and  count the number of occurences per day.


```python
# Code: Add the aggregate function count()
```

Our time series will consist of a series of counts of gun reports per day.


```python
# ID is unimportant. We could have chosen any column, since the counts are the same.
ts_day = None
```


```python
ts_day
```




    Date
    2014-01-01    50
    2014-01-02    33
    2014-01-03    24
    2014-01-04    32
    2014-01-05    17
                  ..
    2020-06-21    52
    2020-06-22    66
    2020-06-23    48
    2020-06-24    58
    2020-06-25    46
    Freq: D, Name: ID, Length: 2368, dtype: int64



Let's visualize our timeseries with a plot.


```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10,5))
ax.plot(ts_day.index, ts_day.values)
ax.set_title('Gun Crimes per day in Chicago')
ax.set_ylabel('Reported Gun Crimes')
```




    Text(0, 0.5, 'Reported Gun Crimes')




![png](index_files/index_74_1.png)


There seems to be some abnormal activity happening towards the end of our series.

**[sun-times](https://chicago.suntimes.com/crime/2020/6/8/21281998/chicago-deadliest-day-violence-murder-history-police-crime)**


```python
ts_day.sort_values(ascending=False)[:10]
```




    Date
    2020-05-31    130
    2020-06-02    109
    2020-06-01     97
    2020-06-03     95
    2020-05-25     93
    2020-06-20     82
    2020-05-24     77
    2018-05-28     74
    2019-05-26     72
    2019-07-20     72
    Name: ID, dtype: int64



Let's treat the span of days from 5-31 to 6-03 as outliers. 

There are several ways to do this, but let's first remove the outliers, and populate an an empty array with the original date range.  That will introduce us to the pd.date_range method.


```python
# Remove outlier counts over 90
daily_count = ts_day[ts_day < 90]

# Use pd.date_range that makes a full date range from index[0] to index[-1]
# 2014-01-01 to 2020-06-25

ts_daterange = pd.date_range(daily_count.index[0], daily_count.index[-1])

# Use np.empty to create an empty array that spans the daterange
ts_daily = np.empty(shape=len(ts_daterange))

# Convert to a series and reindex with the 
ts_daily = pd.Series(ts_daily)
ts_daily = ts_daily.reindex(ts_daterange)

ts_day = ts_daily.fillna(daily_count)

```


```python
fig, ax = plt.subplots(figsize=(10,5))
ts_day.plot(ax=ax)
ax.set_title('Gun Crimes in Chicago with Deadliest Days Removed');
```


![png](index_files/index_79_0.png)


Let's zoom in on that week again


```python
fig, ax = plt.subplots()
ax.plot(ts_day[(ts_day.index > '2020-05-20') 
                 & (ts_day.index < '2020-06-07')]
       )
ax.tick_params(rotation=45)
ax.set_title('We have some gaps now')
```




    Text(0.5, 1.0, 'We have some gaps now')




![png](index_files/index_81_1.png)


The datetime object allows us several options of how to fill those gaps:


```python
ts_day[(ts_day.index > '2020-05-20') 
                 & (ts_day.index < '2020-06-07')]
```




    2020-05-21    46.0
    2020-05-22    48.0
    2020-05-23    68.0
    2020-05-24    77.0
    2020-05-25     NaN
    2020-05-26    56.0
    2020-05-27    47.0
    2020-05-28    58.0
    2020-05-29    54.0
    2020-05-30    51.0
    2020-05-31     NaN
    2020-06-01     NaN
    2020-06-02     NaN
    2020-06-03     NaN
    2020-06-04    64.0
    2020-06-05    60.0
    2020-06-06    59.0
    Freq: D, dtype: float64



# Forward Fill

A simple way to deal with the missing data is to simply roll forward the most recent entry prior to the gap.


```python
# Take the date range above and call the ffill() method
ts_day[(ts_day.index > '2020-05-20') 
                 & (ts_day.index < '2020-06-07')]
```




    2020-05-21    46.0
    2020-05-22    48.0
    2020-05-23    68.0
    2020-05-24    77.0
    2020-05-25     NaN
    2020-05-26    56.0
    2020-05-27    47.0
    2020-05-28    58.0
    2020-05-29    54.0
    2020-05-30    51.0
    2020-05-31     NaN
    2020-06-01     NaN
    2020-06-02     NaN
    2020-06-03     NaN
    2020-06-04    64.0
    2020-06-05    60.0
    2020-06-06    59.0
    Freq: D, dtype: float64




```python
# Below you will find forward fill visualized
fig, (ax1,ax2) = plt.subplots(1,2, figsize = (10,5))
ax1.plot(ts_day[(ts_day.index > '2020-05-20') 
                 & (ts_day.index < '2020-06-07')].ffill()
       )
ax1.tick_params(rotation=45)
ax1.set_title('Forward Fill')

ax2.plot(ts_day[(ts_day.index > '2020-05-20') 
                 & (ts_day.index < '2020-06-07')]
       )
ax2.tick_params(rotation=45)
ax2.set_title('Original')

```




    Text(0.5, 1.0, 'Original')




![png](index_files/index_87_1.png)


## Backward Fill

We can also fill backward, but doing so is more risky, since you are incorporating future information into prior data.  This is a so-called **lookahead**, which is a type of time series data leakage.  If we backfill, we would expect our models to perform unreasonably well predicting data points whose previous values have been backfilled.


```python
# Take the date range above and call the bfill() method

ts_day[(ts_day.index > '2020-05-20') 
                 & (ts_day.index < '2020-06-07')]
```




    2020-05-21    46.0
    2020-05-22    48.0
    2020-05-23    68.0
    2020-05-24    77.0
    2020-05-25     NaN
    2020-05-26    56.0
    2020-05-27    47.0
    2020-05-28    58.0
    2020-05-29    54.0
    2020-05-30    51.0
    2020-05-31     NaN
    2020-06-01     NaN
    2020-06-02     NaN
    2020-06-03     NaN
    2020-06-04    64.0
    2020-06-05    60.0
    2020-06-06    59.0
    Freq: D, dtype: float64




```python
fig, (ax1,ax2) = plt.subplots(1,2, figsize = (10,5))
ax1.plot(ts_day.bfill()[(ts_day.index > '2020-05-20') 
                 & (ts_day.index < '2020-06-07')]
       )
ax1.tick_params(rotation=45)
ax1.set_title('Back Fill')

ax2.plot(ts_day[(ts_day.index > '2020-05-20') 
                 & (ts_day.index < '2020-06-07')]
       )
ax2.tick_params(rotation=45)
ax2.set_title('Original')
```




    Text(0.5, 1.0, 'Original')




![png](index_files/index_91_1.png)


<a id='interpolation'></a>

# Interpolate 
Fills the values according to a specified method. The default linear, assumes the data area evenly spaced along the line connecting the real values surrounding the NaN values.


```python
# Call interpolate on the date range from above
ts_day[(ts_day.index > '2020-05-20') 
                 & (ts_day.index < '2020-06-07')]
```




    2020-05-21    46.0
    2020-05-22    48.0
    2020-05-23    68.0
    2020-05-24    77.0
    2020-05-25     NaN
    2020-05-26    56.0
    2020-05-27    47.0
    2020-05-28    58.0
    2020-05-29    54.0
    2020-05-30    51.0
    2020-05-31     NaN
    2020-06-01     NaN
    2020-06-02     NaN
    2020-06-03     NaN
    2020-06-04    64.0
    2020-06-05    60.0
    2020-06-06    59.0
    Freq: D, dtype: float64




```python
fig, (ax1,ax2) = plt.subplots(1,2, figsize = (10,5))
ax1.plot(ts_day.interpolate()[(ts_day.index > '2020-05-20') 
                 & (ts_day.index < '2020-06-07')]
       )
ax1.tick_params(rotation=45)
ax1.set_title('Interpolation')

ax2.plot(ts_day[(ts_day.index > '2020-05-20') 
                 & (ts_day.index < '2020-06-07')]
       )
ax2.tick_params(rotation=45)
ax2.set_title('Original')
```




    Text(0.5, 1.0, 'Original')




![png](index_files/index_95_1.png)


<a id='components'></a>

## Components of Time Series Data
A time series in general is supposed to be affected by four main components, which can be separated from the observed data. These components are: *Trend, Cyclical, Seasonal and Irregular* components.

- **Trend** : The long term movement of a time series. For example, series relating to population growth, number of houses in a city etc. show upward trend.
- **Seasonality** : Fluctuation in the data set that follow a regular pattern due to outside influences. For example sales of ice-cream increase in summer, or daily web traffic.
- **Cyclical** : When data exhibit rises and falls that are not of fixed period.  Think of business cycles which usually last several years, but where the length of the current cycle is unknown beforehand.
- **Irregular**: Are caused by unpredictable influences, which are not regular and also do not repeat in a particular pattern. These variations are caused by incidences such as war, strike, earthquake, flood, revolution, etc. There is no defined statistical technique for measuring random fluctuations in a time series.


*Note: Many people confuse cyclic behaviour with seasonal behaviour, but they are really quite different. If the fluctuations are not of fixed period then they are cyclic; if the period is unchanging and associated with some aspect of the calendar, then the pattern is seasonal.*

We can use the seasonal_decompose function to show trends the components of our time series.

Our modeling will aim to predict the weekly gun crime counts.
We will treat the outliers with interpolated interpolation.


```python
ts_int = ts_day.interpolate()
```


```python
# Downsample to a weekly count using resample with the 'W' argument and a mean aggregate
ts_weekly = None
```

<a id='decomposition'></a>


```python
from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(ts_weekly)
fig = plt.figure()
fig = decomposition.plot()
fig.set_size_inches(15, 8)
```


    <Figure size 432x288 with 0 Axes>



![png](index_files/index_104_1.png)


We can also get a sense of the patterns in our data using **smoothing**. Noise makes it difficult to see patterns in our time series. Smoothing techniques to see the patterns more clearly.

Common smoothing techniques are simple moving averages and exponentially weighted moving averages.  

 - Simple moving average simply calculates the average of a specified number of points close to the point in question.
 - Exponentially weighted average does the same thing, but gives more weight to points closer in time.

We can call the rolling function plus an aggregate (mean) to calculate the simple moving average.


```python
# call rolling and pass 16 as an argument.  
```


```python
# if we add the above script to the plot below, we can see trends across 16 weeks
# This is a clear display of how gun crime increases in the summer months 
fig, ax = plt.subplots(figsize=(10,5))

ts_weekly<fill_in>.plot(ax=ax, label='16 Week SMA')
ts_weekly.plot(ax=ax, alpha=.5, label='Original Data')
ax.set_title("30 Day SMA Shows Seasonality")
ax.legend();
```


      File "<ipython-input-86-b59af50b0fe7>", line 5
        ts_weekly<fill_in>.plot(ax=ax, label='16 Week SMA')
                          ^
    SyntaxError: invalid syntax




```python
# if we look at the sma across 52 weeks, we can see a clear trend upwards
# This is a clear display of how gun crime increases in the summer months 
fig, ax = plt.subplots(figsize=(10,5))

ts_weekly<fill_in>.plot(ax=ax, label='52 Week SMA')
ts_weekly.plot(ax=ax, alpha=.5, label='Original Data')
ax.set_title("52 Week SMA Shows Trend Upwards though 2017,\n then Slight Decrease Thereafter ")
ax.legend();
```


      File "<ipython-input-88-516ab18ce490>", line 5
        ts_weekly<fill_in>.plot(ax=ax, label='52 Week SMA')
                          ^
    SyntaxError: invalid syntax



<a id='stationarity'></a>

### Statistical stationarity: 

When building our models, we will want to account for these trends somehow.  Time series whose mean and variance have trends across time will be difficult to predict out into the future. 

A **stationary time series** is one whose statistical properties such as mean, variance, autocorrelation, etc. are all constant over time. Most statistical forecasting methods are based on the assumption that the time series can be rendered approximately stationary (i.e., "stationarized") through the use of mathematical transformations. A stationarized series is relatively easy to predict: you simply predict that its statistical properties will be the same in the future as they have been in the past!  


<h3 style="text-align: center;">Constant Mean</p>



<img src='img/mean_nonstationary.webp'/>

<h3 style="text-align: center;">Constant Variance</p>


<img src='img/variance_nonstationary.webp'/>


<h3 style="text-align: center;">Constant Covariance</p>


<img src='img/covariance_nonstationary.webp'/>

While we can get a sense of how stationary our data is with visuals, the Dickey Fuller test gives us a quantitatitive measure.

Here the null hypothesis is that the TS is non-stationary. If the ‘Test Statistic’ is less than the ‘Critical Value’, we can reject the null hypothesis and say that the series is stationary.


```python
from statsmodels.tsa.stattools import adfuller

#create a function that will help us to quickly 
def test_stationarity(timeseries, window):
    
    #Determing rolling statistics
    rolmean = timeseries.rolling(window=window).mean()
    rolstd = timeseries.rolling(window=window).std()

    #Plot rolling statistics:
    fig = plt.figure(figsize=(12, 8))
    orig = plt.plot(timeseries.iloc[window:], color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show()
    
    #Perform Dickey-Fuller test:
    print ('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print (dfoutput)

```


```python
test_stationarity(ts_weekly, 52)
```


![png](index_files/index_120_0.png)


    Results of Dickey-Fuller Test:
    Test Statistic                  -2.562238
    p-value                          0.101056
    #Lags Used                       4.000000
    Number of Observations Used    334.000000
    Critical Value (1%)             -3.450081
    Critical Value (5%)             -2.870233
    Critical Value (10%)            -2.571401
    dtype: float64


As we concluded visually, our original timeseries does not pass the test of stationarity.

### How to stationarize time series data

A series of steps can be taken to stationarize your data - also known -  as removing trends (linear trends, seasonaility/periodicity, etc - more details on transformations <a href='http://people.duke.edu/~rnau/whatuse.htm'>here</a>).


One way to remove trends is to difference our data.  
Differencing is performed by subtracting the previous observation (lag=1) from the current observation, thereby creating a timeseries of differences.  


```python
# Call .diff() on our ts_weekly time series

```


```python
# drop na's and plot

```


```python
ts_weekly.diff(2).dropna()[:5]
```




    2014-01-19   -6.628571
    2014-01-26    5.571429
    2014-02-02   -2.285714
    2014-02-09   -7.428571
    2014-02-16   -6.142857
    Freq: W-SUN, dtype: float64



Sometimes, we have to difference the differenced data (known as a second difference) to achieve stationary data. <b>The number of times we have to difference our data is the order of differencing</b> - we will use this information when building our model.


```python
#Second order difference:

ts_weekly.diff().diff().dropna()[:5]
```




    2014-01-19    17.771429
    2014-01-26    -5.571429
    2014-02-02    -2.285714
    2014-02-09    -2.857143
    2014-02-16     4.142857
    Freq: W-SUN, dtype: float64




```python
# We can also apply seasonal differences by passing 52, i.e. the number of weeks in a year. 
    
ts_weekly.diff(52).dropna()[:10]
```




    2015-01-04   -3.771429
    2015-01-11    1.571429
    2015-01-18    0.428571
    2015-01-25    6.428571
    2015-02-01   -0.285714
    2015-02-08    1.142857
    2015-02-15    0.714286
    2015-02-22    2.714286
    2015-03-01    3.714286
    2015-03-08    5.857143
    Freq: W-SUN, dtype: float64



<a id='dickey-fuller'></a>

Let's difference our data and see if it improves Dickey-Fuller Test


```python
from statsmodels.tsa.stattools import adfuller

#create a function that will help us to quickly 
def test_stationarity(timeseries, window):
    
    #Determing rolling statistics
    rolmean = timeseries.rolling(window=window).mean()
    rolstd = timeseries.rolling(window=window).std()

    #Plot rolling statistics:
    fig = plt.figure(figsize=(12, 8))
    orig = plt.plot(timeseries.iloc[window:], color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show()
    
    #Perform Dickey-Fuller test:
    print ('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print (dfoutput)

```


```python
test_stationarity(ts_weekly.diff().dropna(), 52)
```


![png](index_files/index_134_0.png)


    Results of Dickey-Fuller Test:
    Test Statistic                -1.322052e+01
    p-value                        1.003933e-24
    #Lags Used                     3.000000e+00
    Number of Observations Used    3.340000e+02
    Critical Value (1%)           -3.450081e+00
    Critical Value (5%)           -2.870233e+00
    Critical Value (10%)          -2.571401e+00
    dtype: float64


One we have achieved stationarity the next step in fitting a model to address any autocorrelation that remains in the differenced series. 

Sometimes, we have to difference the differenced data (known as a second difference) to achieve stationary data. <b>The number of times we have to difference our data is the order of differencing</b> - we will use this information when building our model.

One we have achieved stationarity the next step in fitting a model is to address any autocorrelation that remains in the differenced series. 
