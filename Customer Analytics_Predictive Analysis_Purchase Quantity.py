### Libraries ###

import numpy as np
import pandas as pd 
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans 
import pickle 
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt  
import matplotlib.axes as axs
import seaborn as sns
sns.set()
from sklearn.linear_model import LinearRegression

### Data Preparation ###

# load data
df_purchase = pd.read_csv('D:/DATA ANALYST/belajar_python/CUSTOMER ANALYTICS/purchase data.csv')

# import scaler
scaler = pickle.load(open('scaler_purchase.pickle', 'rb'))

# import PCA
pca = pickle.load(open('pca_purchase.pickle', 'rb'))

# import K-Means
kmeans_pca = pickle.load(open('kmeans_pca_purchase.pickle', 'rb'))

# standardization
features = df_purchase[['Sex', 'Marital status', 'Age', 'Education', 'Income', 'Occupation', 'Settlement size']]
df_purchase_segm_std = scaler.transform(features)

# apply PCA
df_purchase_segm_pca = pca.transform(df_purchase_segm_std)
## df_purchase_segm_pca will contain duplicates beacuse 1 person can do more than one transaction

# segment data with k-means
purchase_segm_kmeans_pca = kmeans_pca.predict(df_purchase_segm_pca)

# create a copy of the data frame
df_purchase_predictors = df_purchase.copy()

# add segment labels
df_purchase_predictors['Segment'] = purchase_segm_kmeans_pca
segment_dummies = pd.get_dummies(purchase_segm_kmeans_pca, prefix='Segment', prefix_sep='_')
df_purchase_predictors = pd.concat([df_purchase_predictors, segment_dummies], axis=1)
print(df_purchase_predictors)

df_pa = df_purchase_predictors

# membuat dataframe baru untuk predict
print(df_pa[['Price_1', 'Price_2', 'Price_3', 'Price_4', 'Price_5']].describe())
price_range = np.arange(0.5, 3.5, 0.01)

# membuat master data untuk price elasticities
df_price_elasticities = pd.DataFrame(price_range)
df_price_elasticities = df_price_elasticities.rename(columns={0:"Price_Point"})


### ----- PRICE ELASTICITY OF PURCHASE QUANTITY BY AVERAGE ----- ###

df_purchase_quantity = df_pa[df_pa['Incidence'] == 1]
df_purchase_quantity = pd.get_dummies(df_purchase_quantity, columns=['Brand'], prefix='Brand', prefix_sep='_')
print(df_purchase_quantity.head())

# explore dependent variable quantity
print(df_purchase_quantity.describe())
## There are fourteen thousand six hundred thirty eight observations in total, the minimum quantity purchased is one as we've excluded purchase occasions where no chocolate
#  candy bars were bought and the maximum quantity is 15. We can see the mean is two point seventy seven with a standard deviation of one point eight. Therefore, we expect
#  that on many purchase occasions, more than one chocolate candy bar was purchased. All right, what about the independent variables in our model, what features can influence
#  purchase quantity?

# explore independent variable
print(df_purchase_quantity.columns.values)
## First, we've got ID, ideas like a name, so it doesn't contain any useful information for predictive purposes, then we've got day. Day is the time stamp of the purchase if we
#  consider time series dependencies in a model. This would be an important piece of data. However, for our purposes, we will limit it. We filtered the data by incidence and
#  quantity in the dependent variable. Thus these two are not potential predictors. Then we've got last incidence brand. I cannot find a clear, logical relationship between
#  the quantity purchased today and the brand chosen last time. Therefore, I'll skip that one to. What about the next one? Well, it is very logical that the last purchase
#  quantity will be indicative of the purchase behavior of this particular customer. The problem is that from the descriptive statistics, it is evident that this variable
#  has a maximum of one. In fact, in this dataset, it seems like a binary variable indicating whether a purchase has occurred during the previous shopping trip, as such, it
#  shows whether the purchase quantity was equal to zero. Unfortunately, it does not show the precise quantity purchased on the last shopping trip. The data set needs to be
#  further preprocessed to obtain the exact values. We'll forego the use of the last incremented quantity here. However you can pre-process the data and include it in your
#  model if you wish. OK, then we come to price, this is perhaps the most important factor that will influence purchase quantity, but which price? Well, at this stage of the
#  customer journey, the customer has already selected which bran d to buy. So the decision, how many units is influenced only by the price of the chosen brand and not the other
#  brands. Therefore, we need to filter this information somehow.

df_purchase_quantity['Price_Incidence'] = (df_purchase_quantity['Brand_1'] + df_purchase_quantity['Price_1'] +
                                           df_purchase_quantity['Brand_2'] + df_purchase_quantity['Price_2'] +
                                           df_purchase_quantity['Brand_3'] + df_purchase_quantity['Price_3'] +
                                           df_purchase_quantity['Brand_4'] + df_purchase_quantity['Price_4'] +
                                           df_purchase_quantity['Brand_5'] + df_purchase_quantity['Price_5'] )

df_purchase_quantity['Promotion_Incidence'] = (df_purchase_quantity['Brand_1'] + df_purchase_quantity['Promotion_1'] +
                                           df_purchase_quantity['Brand_2'] + df_purchase_quantity['Promotion_2'] +
                                           df_purchase_quantity['Brand_3'] + df_purchase_quantity['Promotion_3'] +
                                           df_purchase_quantity['Brand_4'] + df_purchase_quantity['Promotion_4'] +
                                           df_purchase_quantity['Brand_5'] + df_purchase_quantity['Promotion_5'] )

## let's move on sex, marital status, age, education, income, occupation and settlement size, we're all used to predict the segment. They will not be a part of this model.


### Model Estimation Linier Regression ###
X = df_purchase_quantity[['Price_Incidence', 'Promotion_Incidence']]
print(X)
Y = df_purchase_quantity['Quantity']
print(Y)

model_quantity = LinearRegression()
model_quantity.fit(X, Y)
print(model_quantity.coef_)

## Hasil: The coefficient shows the change in the dependent variable that is going to occur with a unit change in the respective independent variable. The first coefficient
#  refers to price incidents while the second two promotion incidents. Here's the interpretation. For every dollar increase in price, about zero point eighty two units less
#  chocolate candy bars would be bought naturally. As price increases, purchase quantity decreases. Then if there is a promotion, about zero point eleven units less will be
#  bought. Interestingly, people would buy a bit less if there is a promotion. This may be explained by the fact that our model is quite simplified, given that we've only got
#  two predictors. Maybe some important information is missing in our data set. Another plausible explanation is related to the fact that we are considering the average
#  customer and not any particular segment. Or it may well be that a promotion is prompting a customer to try out a new brand. In that case, we would not expect them to buy
#  many chocolate bars, but a single one.


### Price Elasticity Estimation With Promotion ###
# membuat input data baru untuk predict
df_price_elasticity_quantity = pd.DataFrame(index = np.arange(price_range.size))
df_price_elasticity_quantity['Price_Incidence'] = price_range
df_price_elasticity_quantity['Promotion_Incidence'] = 1

# ekstract koefisien variabel price from model
beta_quantity = model_quantity.coef_[0]
print(beta_quantity)

# predict purchase quantity
predict_quantity = model_quantity.predict(df_price_elasticity_quantity)

# menghitung price elasticity of purchase quantity
price_elasticity_quantity_promo = beta_quantity * price_range / predict_quantity
df_price_elasticities['PE_Quantity_Promotion_1'] = price_elasticity_quantity_promo
print(df_price_elasticities)

# membuat grafik
plt.figure(figsize=(9,6))
plt.plot(price_range, price_elasticity_quantity_promo)
plt.xlabel('Price')
plt.ylabel('Elasticity')
plt.title('Price Elasticity of Purchase Quantity with Promotion')
plt.show()

### Price Elasticity Estimation Without Promotion ###
# membuat input data baru untuk predict
df_price_elasticity_quantity = pd.DataFrame(index = np.arange(price_range.size))
df_price_elasticity_quantity['Price_Incidence'] = price_range
df_price_elasticity_quantity['Promotion_Incidence'] = 0

# ekstract koefisien variabel price from model
beta_quantity = model_quantity.coef_[0]
print(beta_quantity)

# predict purchase quantity
predict_quantity = model_quantity.predict(df_price_elasticity_quantity)

# menghitung price elasticity of purchase quantity
price_elasticity_quantity_no_promo = beta_quantity * price_range / predict_quantity
df_price_elasticities['PE_Quantity_Promotion_0'] = price_elasticity_quantity_no_promo
print(df_price_elasticities)

# membuat grafik
plt.figure(figsize=(9,6))
plt.plot(price_range, price_elasticity_quantity_promo, color='orange')
plt.plot(price_range, price_elasticity_quantity_no_promo, color='blue')
plt.xlabel('Price')
plt.ylabel('Elasticity')
plt.title('Price Elasticity of Purchase Quantity with and Without Promotion')
plt.show()

## Hasil: What we can see is that customers are a tiny bit more elastic when there is a promotion. However, overall, customers are inelastic towards purchase quantity for all
#  prices from zero point five dollars to around two point seventy dollars. As you remember, our most expensive brand cost two point eighty dollars at most.
#  Furthermore, promotion doesn't look like such a big factor either, the two lines practically overlap in many of the price points. Let's think about this for a second.
#  The reason might be that the variables we included in our model hold no predictive value. Therefore, it might seem like it doesn't really make sense to focus too much on
#  the purchase quantity. Neither price nor promotion shifts appear to affect the customer's decision. A different explanation could be that our methodology is imperfect.
#  The main concern here is that we are estimating a model based on the average customer and we already know that there are four distinct and quite different segments.
#  Maybe we could refine our model further to amend that one idea is to calculate price elasticity of demand for each brand, you can just filter all transactions relating to
#  brand five, for instance, and explore the price elasticity of purchase quantity for brand five.


