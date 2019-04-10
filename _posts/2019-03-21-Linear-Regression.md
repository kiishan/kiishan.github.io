---
layout: post
title: Simple Linear Regression
date: 2019-03-21T00:00:00.000Z
published: true
categories: Machine Learning
read_time: true
---
Regression is a technique used to model and analyze the relationships between variables and often times how they contribute and are related to producing a particular outcome together. 

<img src="/images/regression/smpl_linear_reg.png" width="35%" style="float: right;"/>

This technique is used for forecasting, time series modelling and finding the causal effect relationship between the variables.

Establishes a relationship between dependent variable (Y - continuous) and one or more independent variables (X - continuous or discrete) using a best fit straight line (also known as regression line).



It is represented by an equation <img src="http://latex.codecogs.com/svg.latex?Y=c + m * X + e" border="0"/> , where `c` is intercept, `m` is slope of the line and `e` is error term. This equation can be used to predict the value of target variable based on given predictor variable(s).

__How to obtain best fit line (Value of c and m)?__

1. Least Square Method
2. Gradient Descent Method


### When, why, and how you should use linear regression?
---
__What__:  

1. Form of predictive modelling technique.
2. Technique used to model and analyze the relationships between variables 

__Where__: 

1. Forecasting
2. Time series modelling
3. finding the causal effect relationship between the variables

__When__:   

1. The relationship between the variables is linear.
2. The data is homoskedastic, meaning the variance in the residuals (the difference in the real and predicted values) is more or less constant.
3. The residuals are independent, meaning the residuals are distributed randomly and not influenced by the residuals in previous observations. If the residuals are not independent of each other, they’re considered to be autocorrelated.
4. The residuals are normally distributed. This assumption means the probability density function of the residual values is normally distributed at each x value. I leave this assumption for last because I don’t consider it to be a hard requirement for the use of linear regression, although if this isn’t true, some manipulations must be made to the model.

__Why__: when you want to predict a continuous dependent variable from a number of independent variables.  
__How__:  we fit a curve / line to the data points, in such a manner that the differences between the distances of data points from the curve or line is minimized.



### Assumptions of Linear Regression
---

1. There should be a __linear and additive relationship__ between dependent (response) variable and independent (predictor) variable(s)

    - A linear relationship suggests that a change in response Y due to one unit change in X¹ is constant, regardless of 
         the value of X¹. 
    - An additive relationship suggests that the effect of X¹ on Y is independent of other variables.
    - The linearity assumption can best be tested with scatter plots.
    
<img src="/images/regression/qqplot-comparison.png" width="60%" style="float: right;"/>

2. The linear regression analysis requires all variables to be __multivariate normal__. 

    - One definition is that a random vector is said to be k-variate normally distributed if every linear combination  of its k components has a univariate normal distribution.
    - This assumption can best be __checked with a histogram or a Quantile -Quantile(Q-Q) Plot.__
    - Normality can be checked with a goodness of fit test, e.g., the Kolmogorov-Smirnov test.  
    - When the data is not normally distributed a non-linear transformation (e.g., log-transformation) might fix this issue.


3. Linear regression assumes that there is little or __no multicollinearity__ in the data. Multicollinearity occurs when the independent variables are highly correlated with each other.
    
    - Multicollinearity can increase the variance of the coefficient estimates and make the estimates very sensitive to minor changes in the model. The result is that the coefficient estimates are unstable.
    - Multicollinearity may be tested with three central criteria:

    (i) __Correlation matrix__ – when computing the matrix of Pearson’s Bivariate Correlation among all independent variables the correlation coefficients need to be smaller than 1.

    (ii) __Tolerance__ – the tolerance measures the influence of one independent variable on all other independent variables; the tolerance is calculated with an initial linear regression analysis.  Tolerance is defined as T = 1 – R² for these first step regression analysis.  With T < 0.1 there might be multicollinearity in the data and with T < 0.01 there certainly is.

    (iii) __Variance Inflation Factor (VIF)__ – the variance inflation factor of the linear regression is defined as VIF = 1/T. With VIF > 10 there is an indication that multicollinearity may be present; with VIF > 100 there is certainly multicollinearity among the variables.
               
    (iv) __Condition Index__ – the condition index is calculated using a factor analysis on the independent variables. Values of 10-30 indicate a mediocre multicollinearity in the linear regression variables, values > 30 indicate strong multicollinearity.

    - If multicollinearity is found in the data, centering the data (that is deducting the mean of the variable from each score) might help to solve the problem.  However, the simplest way to address the problem is to remove independent variables with high VIF values.

4. Linear regression analysis requires that there is little or __no autocorrelation__ in the data.  

    - Autocorrelation occurs when the residuals are not independent from each other. 
    - For instance, this typically occurs in time series data like stock prices, where the price is not independent from the previous price.
    - While a scatterplot allows you to check for autocorrelations, you can test the linear regression model for autocorrelation with the Durbin-Watson test.  Durbin-Watson’s d tests the null hypothesis that the residuals are not linearly auto-correlated.  While d can assume values between 0 and 4, values around 2 indicate no autocorrelation.  As a rule of thumb values of 1.5 < d < 2.5 show that there is no auto-correlation in the data. However, the Durbin-Watson test only analyses linear autocorrelation and only between direct neighbors, which are first order effects.

<img src="/images/regression/homoscedasticity_heteroscedasticity.jpg" width="60%" style="float: right;"/>

5. The last assumption of the linear regression analysis is __homoscedasticity__ (homogeneity of variance).  

    - The scatter plot is good way to check whether the data are homoscedastic (meaning the residuals are equal across the regression line).  The following scatter plots show examples of data that are not homoscedastic.

    - The Goldfeld-Quandt Test can also be used to test for heteroscedasticity. The test splits the data into two groups and tests to see if the variances of the residuals are similar across the groups.  If homoscedasticity is present, a non-linear correction might fix the problem.


---
***Note*** 

1. Linear Regression is very sensitive to __outliers__. It can terribly affect the regression line and eventually the forecasted values.   
2. In case of multiple independent variables, we can go with forward selection, backward elimination and step wise approach for selection of most significant independent variables.

---

### Ordinary Least Squares Method
---

Aim is to model equation of line : <img src="http://latex.codecogs.com/svg.latex?y(pred) = b_0 + b_1x" border="0"/>  
step 1: calculate mean of independent variable (x) : <img src="http://latex.codecogs.com/svg.latex?\bar{x} = \frac{\sum_{i=1}^n (x_i)}{n}" border="0"/>  
step 2: calculate mean of dependent variable (y) &nbsp;  : <img src="http://latex.codecogs.com/svg.latex?\bar{y} = \frac{\sum_{i=1}^n (y_i)}{n}" border="0"/>    
setp 3: calculate slope of line $b_1$) &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; : <img src="http://latex.codecogs.com/svg.latex?b_1 = \frac{\sum_{i=1}^n (x_i - \bar{x})(y_i - \bar{y})}{\sum_{i=1}^n (x_i - \bar{x})^2}" border="0"/>    
Step 4: calculate intercept of line ($b_0$) &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;: <img src="http://latex.codecogs.com/svg.latex?b_0 = \bar{y} - b_1\bar{x}" border="0"/>  

Ordinary Least Square method looks simple and computation is easy. But, this OLS method will only work for a univariate dataset which is single independent variables and single dependent variables. Multi-variate dataset contains a single independent variables set and multiple dependent variables sets, require us to use a machine learning algorithm called &nbsp; “Gradient Descent”.


### Gradient Descent Method
---
__Loss Function__

The loss is the error in our predicted value of m and c. Our goal is to minimize this error to obtain the most accurate value of m and c.
We will use the Mean Squared Error function to calculate the loss. There are three steps in this function:

1. Find the difference between the actual y and predicted y value(y = mx + c), for a given x.
2. Square this difference.
3. Find the mean of the squares for every value in X.

<img src="http://latex.codecogs.com/svg.latex?E = \frac{1}{n} \sum_{i=0}^n (y_i - \bar{y_i})^2" border="0"/>

Here yᵢ is the actual value and ȳᵢ is the predicted value. Lets substitute the value of ȳᵢ:

<img src="http://latex.codecogs.com/svg.latex?E = \frac{1}{n} \sum_{i=0}^n (y_i - (mx_i+c))^2 " border="0"/>

So we square the error and find the mean. hence the name Mean Squared Error. Now that we have defined the loss function, lets get into the interesting part — minimizing it and finding m and c.


Let’s try applying gradient descent to m and c and approach it step by step:

1. Initially let m = 0 and c = 0. Let L be our learning rate. This controls how much the value of m changes with each step. L could be a small value like 0.0001 for good accuracy.

2. Calculate the partial derivative of the loss function with respect to m, and plug in the current values of x, y, m and c in it to obtain the derivative value D.

<img src="http://latex.codecogs.com/svg.latex?D_m = \frac{1}{n} \sum_{i=0}^n 2(y_i - (mx_i+c))(-x_i)" border="0"/>

<img src="http://latex.codecogs.com/svg.latex?D_m = \frac{-2}{n} \sum_{i=0}^n x_i(y_i - \bar{y_i})" border="0"/>
    - Dₘ is the value of the partial derivative with respect to m. Similarly lets find the partial derivative with respect to c, Dc :

<img src="http://latex.codecogs.com/svg.latex?D_c = \frac{-2}{n} \sum_{i=0}^n (y_i - \bar{y_i})" border="0"/>

3. Now we update the current value of m and c using the following equation:

<img src="http://latex.codecogs.com/svg.latex?m= m-l*D_m" border="0"/>
<img src="http://latex.codecogs.com/svg.latex?c= c-l*D_c" border="0"/>

4. We repeat this process until our loss function is a very small value or ideally 0 (which means 0 error or 100% accuracy). The value of m and c that we are left with now will be the optimum values.



### Metrics for model evaluation
---

__1. Regression sum of squares (SSR)__

This gives information about how far estimated regression line is from the horizontal ‘no relationship’ line (average of actual output).

<img src="http://latex.codecogs.com/svg.latex?Error = \sum_{i=1}^n (Predected\:output - Average\:of\:actual\:output)^2" border="0"/>

__2. Sum of Squared error (SSE)__

How much the target value varies around the regression line (predicted value).
 
<img src="http://latex.codecogs.com/svg.latex?Error = \sum_{i=1}^n (Actual\:output - Predected\:output)^2" border="0"/>

__3. Total sum of squares (SSTO)__

This tells how much the data point move around the mean.
 
<img src="http://latex.codecogs.com/svg.latex?Error = \sum_{i=1}^n (Actual\:output - Average\:of\:actual\:output)^2" border="0"/>

<img src="http://latex.codecogs.com/svg.latex?R^2 = 1- \frac{SSE}{SSTO}" border="0"/>

__4. The Coefficient of Determination (<img src="http://latex.codecogs.com/svg.latex?R^2" border="0"/>) R-Square__   

The coefficient of determination (denoted by R2) is a key output of regression analysis. It is interpreted as the proportion of the variance in the dependent variable that is predictable from the independent variable.

- The coefficient of determination ranges from 0 to 1.
- An R2 of 0 means that the dependent variable cannot be predicted from the independent variable.
- An R2 of 1 means the dependent variable can be predicted without error from the independent variable.
- An R2 between 0 and 1 indicates the extent to which the dependent variable is predictable. An R2 of 0.10 means that 10 percent of the variance in Y is predictable from X; an R2 of 0.20 means that 20 percent is predictable; and so on.

The formula for computing the coefficient of determination for a linear regression model with one independent variable is given below.

<img src="http://latex.codecogs.com/svg.latex?R^2 =  \frac{2}{N} * \sum \frac{(x_i - x) * (y_i - y)}{(\sigma x * \sigma y)}" border="0"/>

where N is the number of observations used to fit the model, Σ is the summation symbol, xi is the x value for observation i, x is the mean x value, yi is the y value for observation i, y is the mean y value, σx is the standard deviation of x, and σy is the standard deviation of y.


__5. Correlation co-efficient (r)__

This is related to value of ‘r-squared’ which can be observed from the notation itself. It ranges from -1 to 1.

<img src="http://latex.codecogs.com/svg.latex?r = \pm \sqrt{R^2}" border="0"/>

If the value of b1 is negative, then ‘r’ is negative whereas if the value of ‘b1’ is positive then, ‘r’ is positive. It is unitless.


__Is the range of R-Square always between 0 to 1?__

Value of R2 may end up being negative if the regression line is made to pass through a point forcefully. This will lead to forcefully making regression line to pass through the origin (no intercept) giving an error higher than the error produced by the horizontal line. This will happen if the data is far away from the origin.
