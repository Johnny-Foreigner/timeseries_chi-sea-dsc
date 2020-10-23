
<a id='section_1'></a>

# Time Series vs. Linear

For linear regression, we attempted to explain the variance of a continuous target variable via a set of **independent predictor features**. We assumed that there was no **autocorrelation** amongst our records.  In other words, we did not use the target variable of one row to predict that of another.

In time series models, we make the opposite assumption.  We assume that a given value can best be predicted by its **past values**.

The main idea with time series is to replace our independent features with past values of our target. 

The models we will cover in lecture include endogenous variables.
<em>Endogenous</em> means caused by factors within the system. 

<em>Exogenous</em>, caused by factors outside the system. 

Many statsmodels tools use <tt>endog</tt> to represent the incoming time series data in place of the constant <tt>y</tt>.<br>

For more information and a nice **mneumonic**, visit http://www.statsmodels.org/stable/endog_exog.html

Time series analysis has many applications.  With new methods of personalized data collection, the opportunity for time series analysis is growing.  Take health care,  where new wearable technology is producing individualized records of medical data. With a smartwatch or phone, heartrate, bloodpressure, sleep and activity records, can all be recorded easily. All of these datapoints can be timestamped precisely, and easily exported for analysis.

There is also plenty of opportunities to apply time series models in other fields.  In finance, time series data is plentiful, collected by both the government and private industry.  Data scientest can use financial data to build models not only for personal financial betterment, but for forecasting economic cycles and unemployment rate.

# Time Series

## Agenda

1. [Time Series Models vs. Linear Models](#section_1)
2. [Date Time Objects](#section_2)
3. [Time Series Preprocessing Techniques](#section_3)
 - [Resampling](#resampling)
 - Interpolating
4. [Visual Diagnostics](#moving_avg)
 - Moving Average and Exponentially Weighted Moving Average
5. [Components of Time Series Data and Stationarity](#stationarity)
 - Decomposition
 - Dickey-Fuller
    

To begin, let's look at some time series data plots.

The ** syntax is used to pass keywords and values in dictionary form to a function. For more on * and ** (*args and **kwargs), see this page.

<a id='section_2'></a>

# 2: Datetime objects

Datetime objects make our time series modeling lives easier.  They will allow us to perform essential data prep tasks with a few lines of code.  

We need our timeseries **index** to be datetime objects, since our models will rely on being able to identify the previous chronological value.

There is a datetime [library](https://docs.python.org/2/library/datetime.html), and inside pandas there is a datetime module as well as a to_datetime() function.


Let's import some data on **gun violence in Chicago**.

[source](https://data.cityofchicago.org/Public-Safety/Gun-Crimes-Heat-Map/iinq-m3rg)


```python
ts = pd.read_csv('data/Gun_Crimes_Heat_Map.csv')
```

Let's look at some summary stats:


```python
ts['Description'].value_counts()
```


```python
ts.Domestic.value_counts()[1]/len(ts)
```


```python
arrest_rate = ts['Arrest'].value_counts()[1]/len(ts)
```

The data extracts the year of offense as its own columns.

While this does show some interesting information that will be relevant to our time series analysis, we are going to get more granular.

# Date Time Objects

For time series modeling, the first step is to make sure that the index is a date time object.

There are a few ways to **reindex** our series to datetime. 

We can use the pd.to_datetime() method


```python
ts.set_index(pd.to_datetime(ts['Date']), drop=True, inplace=True)
```

Or, we can parse the dates directly on import

We've covered some of the fun abilities of datetime objects, including being able to extract components of the date like so:


```python
ts.index[0].month
```


```python
ts.index[0].year
```


```python
ts.index?
```

We can easily see now see whether offenses happen, for example, during business hours.


### With a partner, take five minutes ot play around with the datetime object, and make a plot that answers a time based question about our data.


```python
ts['dow'] = ts.index

# 0 aligns with Monday
ts['dow'] = ts.dow.apply(lambda x: x.dayofweek)

x = ts.dow.value_counts(sort=False).index

height = ts.dow.value_counts(sort=False)

fig, ax = plt.subplots()

sns.barplot(x, height, color='r', ax=ax)
ax.set_title('Gun Crime Across the Months')
ax.set_ylabel('Gun Crime Count')
ax.set_xlabel('Day of Week')
ax.set_xticklabels(['M', 'T', 'W', 'Th', 'F', 'Sa', 'Su']);
```


```python
ts['quarter'] = ts.index
ts['quarter'] = ts.quarter.apply(lambda x: x.quarter)

x = ts.quarter.value_counts(sort=False).index

height = ts.quarter.value_counts(sort=False)

fig, ax = plt.subplots()

sns.barplot(x, height, color='r', ax=ax)
ax.set_title('Gun Crime Across Quarters')
ax.set_ylabel('Gun Crime Count')
ax.set_xlabel('Quarter')

```


```python
ts['month'] = ts.index
ts['month'] = ts.month.apply(lambda x: x.month)


x = ts.month.value_counts(sort=False).index

height = ts.month.value_counts(sort=False)

fig, ax = plt.subplots()

sns.barplot(x, height, color='r', ax=ax)
ax.set_title('Gun Crime Across the Months')
ax.set_ylabel('Gun Crime Count')
ax.set_xlabel('Month')
```

![pair](https://media.giphy.com/media/SvulfW0MQncFYzQEMT/giphy.gif)

<a id='section_3'></a>

# 3: Time Series Preprocessing Techniques

<a id='resampling'></a>

## Resampling
We have new abilities, such as **resampling**

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

To create our timeseries, we will count the number of gun offenses reported per day.


```python
ts.resample('D')
```

When resampling, we have to provide a rule to resample by, and an **aggregate function**.

**To upsample** is to increase the frequency of the data of interest.  
**To downsample** is to decrease the frequency of the data of interest.

For our purposes, we will downsample, and  count the number of occurences per day.

Our time series will consist of a series of counts of gun reports per day.


```python
ts_day = ts.resample('D').count()['ID']
```

Let's visualize our timeseries with a plot.

There seems to be some abnormal activity happening towards the end of our series.

**[sun-times](https://chicago.suntimes.com/crime/2020/6/8/21281998/chicago-deadliest-day-violence-murder-history-police-crime)**

Let's treat the span of days from 5-31 to 6-03 as outliers. 

There are several ways to do this, but let's first remove the outliers, and populate an an empty array with the original date range.  That will introduce us to the pd.date_range method.

Let's zoom in on that week again

The datetime object allows us several options of how to fill those gaps:

## Forward Fill


```python
# Forward Fill
# Take the date range above and call the ffill() method
ts_day[(ts_day.index > '2020-05-20') 
                 & (ts_day.index < '2020-06-07')].ffill()
```

## Backward Fill


```python
ts_day[(ts_day.index > '2020-05-20') 
                 & (ts_day.index < '2020-06-07')].bfill()
```

# Interpolate 
Fills the values according to a specified method. The default linear, assumes the data area evenly spaced along the line connecting the real values surrounding the NaN values.


```python
ts_day[(ts_day.index > '2020-05-20') 
                 & (ts_day.index < '2020-06-07')].interpolate()
```

<a id='moving_avg'></a>

# SMA and EWMA

We could also proceed by smoothing our data.

- Smoothing is replacing the measured value on each day with an average of a moving window across near values.  Applying smoothing, with for example a simple moving average an exponentially weighted average, can be used to minimize the effect of outliers. It can then serve as an alternative to dropping outliers to reduce measurement spikes and errors of measurement (Practical Time Series Analysis, Nielson, p. 55)

- Moving averages can also provide a clearer picture of trends in noisy data. 



# Simple Moving Average

A simple moving average consists of an average across a specified window of time. 

The datetime index allows us to calculate simple moving averages via the rolling function.

The rolling function calculates a statistic across a moving **window**, which we can change with the window paraamter.


```python
ts_to_smooth.rolling(window=7)
```

The rolling method requires we specify an aggregate function. For moving average, we call mean.

This is simply the avarage of a datapoint and the previous seven data points:

If we increase the window even more, the data **smooths** out in a way to help visualize the underlying seasonal pattern. 

If we plot the moving average across 365 days, we can see a smooth trend across a year.  The SMA reaches back 365 weeks, showing that the steepest growth of gun crime started around the beginning of 2016 and leveled out at the beginning of 2017.

# EWMA
## Exponentially Weighted Moving Average 

An alternative to SMA is the EWMA. The exponentially weighted average gives more weight to the points closer to the date in question.  With EWMA, the average will track more closely to the peaks and valleys. If there are extreme historical values in the dataset, the EWMA will be less skewed than the SMA.


$\large X_{t} = \beta * X_{t-1} * (1-\beta)*X_t$

Which after recursion, breaks resolves into:

$\large \beta^3 * X_{t-3} + \beta^2 * X_{t-2} + \beta * X_{t-1}+ (1-\beta)*X_t$

For this equation, X_t gives us an approximation of the last $1/(1-\beta)$ days.

>Andrew Ng, [EWMA](https://www.coursera.org/learn/deep-neural-network/lecture/duStO/exponentially-weighted-averages)

The higher the $\alpha$ parameter, the closer the EWMA will be to the actual value of the point.

Let's plot our rolling statistics with some different windows

Again, if we zoom in to the year level, we can see peaks and valleys according to the seasons.  

We can also plot rolling averages for the variance and standard deviation.

If we zoom in on our standard deviation, we can the variance of our data has quite a fluctuation at different moments in time.  When we are building our models, we will want to remove this variability, or our models will have different performance at different times.  We will be unable, then to be confident our model will perform well at an arbitrary point in the future.


<a id='stationarity'></a>

### Components of Time Series Data
A time series in general is supposed to be affected by four main components, which can be separated from the observed data. These components are: *Trend, Cyclical, Seasonal and Irregular* components.

- **Trend** : The long term movement of a time series. For example, series relating to population growth, number of houses in a city etc. show upward trend.
- **Seasonality** : Fluctuation in the data set that follow a regular pattern due to outside influences. For example sales of ice-cream increase in summer, or daily web traffic.
- **Cyclical** : When data exhibit rises and falls that are not of fixed period.  Think of business cycles which usually last several years, but where the length of the current cycle is unknown beforehand.
- **Irregular**: Are caused by unpredictable influences, which are not regular and also do not repeat in a particular pattern. These variations are caused by incidences such as war, strike, earthquake, flood, revolution, etc. There is no defined statistical technique for measuring random fluctuations in a time series.


*Note: Many people confuse cyclic behaviour with seasonal behaviour, but they are really quite different. If the fluctuations are not of fixed period then they are cyclic; if the period is unchanging and associated with some aspect of the calendar, then the pattern is seasonal.*

The statsmodels seasonal decompose can also help show us the trends in our data.

Our modeling will aim to predict the weekly gun crime counts.
We will treat the outliers with interpolated interpolation.


```python
ts_weekly = ts_int.resample('W').mean()
```

We can now use the seasonal_decompose function to show trends the components of our time series.

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

As we concluded visually, our original timeseries does not pass the test of stationarity.

### How to stationarize time series data

A series of steps can be taken to stationarize your data - also known -  as removing trends (linear trends, seasonaility/periodicity, etc - more details on transformations <a href='http://people.duke.edu/~rnau/whatuse.htm'>here</a>).


One way to remove trends is to difference our data.  
Differencing is performed by subtracting the previous observation (lag=1) from the current observation.


```python
ts_weekly.diff().dropna()[:5]
```


```python
ts_weekly.diff().dropna().plot()
```

Sometimes, we have to difference the differenced data (known as a second difference) to achieve stationary data. <b>The number of times we have to difference our data is the order of differencing</b> - we will use this information when building our model.

Let's difference our data and see if it improves Dickey-Fuller Test

One we have achieved stationarity the next step in fitting a model to address any autocorrelation that remains in the differenced series. 

Sometimes, we have to difference the differenced data (known as a second difference) to achieve stationary data. <b>The number of times we have to difference our data is the order of differencing</b> - we will use this information when building our model.

One we have achieved stationarity the next step in fitting a model is to address any autocorrelation that remains in the differenced series. 
