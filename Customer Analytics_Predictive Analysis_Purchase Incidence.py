### Libraries ###
import numpy as np
import pandas as pd 
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans 
import pickle 
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt 
import seaborn as sns
sns.set()

### Data Preparation ###

# load data
df_purchase = pd.read_csv('D:/DATA ANALYST/belajar_python/CUSTOMER ANALYTICS/purchase data.csv')

# import scaler
scaler = pickle.load(open('scaler_purchase.pickle', 'rb'))

#import PCA
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

### ----- Purchase Probability Model ----- ###

Y = df_pa['Incidence']

## variabel X yaitu rata-rata harga per row (row disini yaitu per kunjungan konsumen)
# We will opt for the mean and it will be the average price of chocolate candy bars for each row in our data frame.
## Prie can be represent with minimum price, maximum price, mean price, and median price, tergantung variabel dan kondisi data
X = pd.DataFrame()
X['Mean_Price'] = (df_pa['Price_1'] +
                   df_pa['Price_2'] +
                   df_pa['Price_3'] +
                   df_pa['Price_4'] +
                   df_pa['Price_5'] ) / 5

model_purchase = LogisticRegression(solver='sag')
model_purchase.fit(X,Y)
print(model_purchase.coef_)

## adding a solver argument we will choose SAG as it is optimal for simple problems with large datasets like ours.
## hasil: We observed that the coefficient for price has a value of minus two point three four. Just by the sign, we know that a decrease in price would lead to an increase in
#  purchase probability, this model quantifies the exact relationship between price and probability of purchase. In the next lecture, we'll observe not only the direction but
#  also the magnitude of the effect of price on purchase probability.

### Price Elasticity of Purchase Probability ###

## We have information about the change in purchase probability, given a price. That's what the logistic regression coefficient is showing. Therefore, we want to check different
#  values for the mean price and see how they affect the purchase probability. Since we are programming, we can define a range of prices and analyze each one of them.

print(df_pa[['Price_1', 'Price_2', 'Price_3', 'Price_4', 'Price_5']].describe())
price_range = np.arange(0.5, 3.5, 0.01)
df_price_range = pd.DataFrame(price_range)

# price probability by customer
Y_pr = model_purchase.predict_proba(df_price_range)
print(Y_pr)

## hasil: what we've obtained are the probabilities for our two classes, zero and one, zero implies no purchase and one purchase.
## The first value in a row shows the probability of no purchase, while the second the probability of purchase.
## Since we are interested only in the probability of purchase, we can simply take the second column of that array.

purchase_pr = Y_pr[:][:, 1]

## even from the initial data frame, you can see that with lower prices, the purchase probability is higher, while with higher prices lower, as we already noted, that's normal.
## But the real question is, how does demand for the product change with a given change in price?
## Well, time to calculate the elasticities, then they will provide us with exactly this insight.

## Normally we would have to do the math in order to simplify it into something we can code here. We will give you the simplified result directly and you can find their derivation
#  in the course notes. The simplified formula for elasticity is the coefficient from the model times price times one minus the purchase probability.

## price elasticity of purchase probability by customer
pe = model_purchase.coef_[:, 0] * price_range * (1 - purchase_pr)

## result price elasticity of purchase probability
df_price_elasticities = pd.DataFrame(price_range)
df_price_elasticities = df_price_elasticities.rename(columns={0:"Price_Point"})
df_price_elasticities['Mean_PE'] = pe
print(df_price_elasticities)

#pd.options.display.max_rows = None
print(df_price_elasticities)

plt.figure(figsize=(9,6))
plt.plot(price_range, pe, color='grey')
plt.xlabel('Price')
plt.ylabel('Elasticity')
plt.title('Price Elasticity of Purchase Probability')
plt.show()

### Hasil: the price elasticity decreases as price increases. No surprise here, the higher the price of a product becomes, the less likely it will be for people to want to buy it.
# Let's see, the decrease in price is slow in the range between zero point five and one point one, and then it becomes steeper after the one point one mark.
# The other important observation we make is that the price elasticities are all negative. We use the models price coefficient, which is negative two point thirty five, thus
#  indicating the inverse proportionality between price and purchase probability. Hence, our price elasticities are negative as well.
# By definition, elasticity measures the percent change in an output variable, in our case purchase probability, given a percent change in an input variable or in our case, price.
# Now, if this percent change is greater than 100 percent, we say that the output or purchase probability is called elastic. On the other hand, for changes less than 100 percent,
#  it is inelastic. In other words, if the elasticity has a value smaller than one, in absolute terms, we say it is inelastic. If it is greater than one, we say it is elastic.
# At one point ten price, the average customer has an elasticity of minus zero point six nine. This means that for each increase in price by one percent, the probability of
#  purchase will change by minus zero point six nine percent. Therefore, we expect it to decrease by zero point six nine percent. The important observation here is that an
#  increase of one percent in elasticity leads to a decrease of less than one percent. Therefore, purchase probability at this point is inelastic.
# Let's examine another elasticity, this time at a higher price point. For instance, if we look at the one point fifty price, the elasticity is minus one point seven. An
#  increase of one percent in price would translate into a decline of minus one point seven percent of purchase probability. Therefore, an increase of one percent will lead
#  to a decrease of almost two percent in purchase probability. In this case, the elasticity of purchase probability is elastic.
# This is an important distinction, and the reason is that for inelastic values, the general recommendation is to increase the price as it wouldn't cause a significant decrease
#  in the output variable or in our case, purchase probability. On the other hand, if we have the elasticity, which is greater than one in absolute terms, we should decrease our
#  prices. In our graph, it starts from being inelastic and then switches to being elastic. we observed this happens at the one point twenty five mark. And that brings us to the
#  following conclusion, with prices lower than one point twenty five, we can increase our product price without losing too much in terms of purchase, probability for price is
#  higher than one point twenty five. We have more to gain by reducing our prices.

## If we want to develop concrete marketing strategies, we'll need to refine our analysis further. And the way to do that is through a more specific representation of customers
#  behavior. Luckily, we've already achieved this by segmenting our data.


### ----- Price Elasticity of Purchase Probability by Segment ----- ###
# segment 0: standard, segment 1: career focused, segment 2: fewer opportunities, career 3: well off

## Segment 1 - Career-Focused ##
df_pa_segment_1 = df_pa[df_pa['Segment'] == 1]

Y = df_pa_segment_1['Incidence']
X = pd.DataFrame()
X['Mean_Price'] = (df_pa_segment_1['Price_1'] +
                   df_pa_segment_1['Price_2'] +
                   df_pa_segment_1['Price_3'] +
                   df_pa_segment_1['Price_4'] +
                   df_pa_segment_1['Price_5'] ) / 5

model_incidence_segment_1 = LogisticRegression(solver='sag')
model_incidence_segment_1.fit(X, Y)

print(model_incidence_segment_1.coef_)
# Hasil: let's look at the coefficient. It's minus one point seven, so it's lower in absolute terms compared to the average consumer's coefficient, therefore it will have
#  a lower impact when we calculate the elasticities.

# purchase probability by segment 1
Y_segment_1 = model_incidence_segment_1.predict_proba(df_price_range)
purchase_pr_segment_1 = Y_segment_1[:][:, 1]

# price elasticity of purchase probability by segment 1
pe_segment_1 = model_incidence_segment_1.coef_[:, 0] * price_range * (1 - purchase_pr_segment_1)

# results price elasticity of purchase probability 
df_price_elasticities['PE_Segment_1'] = pe_segment_1
print(df_price_elasticities)

plt.figure(figsize=(9,6))
plt.plot(price_range, pe, color='grey')
plt.plot(price_range, pe_segment_1, color='green')
plt.xlabel('Price')
plt.ylabel('Elasticity')
plt.title('Price Elasticity of Purchase Probability')
plt.show()

## Hasil: The two elasticity lines live very close together on the zero point five and one point zero range, from there on, the green curve sits above the gray one.
# As we suspected, the purchase probabilities of the career focused segment are less elastic than average. In fact, we can observe at which point they become inelastic, if
#  we refer to our table again. A dollar, 39 cents (1.39 dollar comapre to 1.25 dollar). That's 14 cents higher than the average turning point. in this case, we'd increase
#  prices if we were in the zero point five to one point three nine range and think about decreasing them afterwards if we want to target the purchase probability of the
#  career focused segment.

## Segment 2 - Fewer-Opportunities ##
# select only customers drom segment 2
df_pa_segment_2 = df_pa[df_pa['Segment'] == 2]

# build logistic regression model
Y = df_pa_segment_2['Incidence']
X = pd.DataFrame()
X['Mean_Price'] = (df_pa_segment_2['Price_1'] +
                   df_pa_segment_2['Price_2'] +
                   df_pa_segment_2['Price_3'] +
                   df_pa_segment_2['Price_4'] +
                   df_pa_segment_2['Price_5'] ) / 5

model_incidence_segment_2 = LogisticRegression(solver='sag')
model_incidence_segment_2.fit(X, Y)

print(model_incidence_segment_2.coef_)

# purchase probability by segment 2
Y_segment_2 = model_incidence_segment_2.predict_proba(df_price_range)
purchase_pr_segment_2 = Y_segment_2[:][:, 1]

# price elasticity of purchase probability by segment 2
pe_segment_2 = model_incidence_segment_2.coef_[:, 0] * price_range * (1 - purchase_pr_segment_2)

# results price elasticity of purchase probability
df_price_elasticities['PE_Segment_2'] = pe_segment_2
print(df_price_elasticities)

plt.figure(figsize=(9,6))
plt.plot(price_range, pe, color='grey')
plt.plot(price_range, pe_segment_1, color='green')
plt.plot(price_range, pe_segment_2, color='r')
plt.xlabel('Price')
plt.ylabel('Elasticity')
plt.title('Price Elasticity of Purchase Probability')
plt.show()

## Hasil: now we have the three elasticities together and straight away we can see that this segment is quite different when compared to the other two, which is why we decided
#  to display all the elasticities. At the same time, we can observe that the fewer opportunities segment is more price sensitive compared to the average and a lot more
#  sensitive compared to the career focused. The line is not only lower than the other two, but it is also much steeper. And this means that with an increase in price, they
#  become more and more elastic, much faster. Lastly, let's see the tipping point between elasticity and inelasticity for the Fewer Opportunities segment. It seems to stand at
#  one point two seven cents, this is extremely interesting because the average tipping point was at one point twenty five. It seems that this segment is more inelastic at lower
#  prices. This is also evident from the graph. The red line is a bit higher than the other two in the beginning. Later on, however, it becomes much steeper.
#  So let's elaborate a bit on that. Now, this may be due to two main reasons. The first one is technical. Since the fewer opportunities cluster is the biggest one, maybe the
#  abundance of data is resulting in a more sophisticated model. The second one is related to purchasing behavior. Maybe the fewer opportunities cluster enjoys chocolate candy
#  bars so much that a price increase in the lower price ranges won't stop them from buying it. However, once it starts becoming expensive, it does not make any financial sense
#  to them to invest in it.

## Segment 0 - Standard ##
# select only customers drom segment 0
df_pa_segment_0 = df_pa[df_pa['Segment'] == 0]

#build logistic regression model
Y = df_pa_segment_0['Incidence']
X = pd.DataFrame()
X['Mean_Price'] = (df_pa_segment_0['Price_1'] +
                   df_pa_segment_0['Price_2'] +
                   df_pa_segment_0['Price_3'] +
                   df_pa_segment_0['Price_4'] +
                   df_pa_segment_0['Price_5'] ) / 5

model_incidence_segment_0 = LogisticRegression(solver='sag')
model_incidence_segment_0.fit(X, Y)

print(model_incidence_segment_0.coef_)

# purchase probability by segment 0
Y_segment_0 = model_incidence_segment_0.predict_proba(df_price_range)
purchase_pr_segment_0 = Y_segment_0[:][:, 1]

# price elasticity of purchase probability by segment 0
pe_segment_0 = model_incidence_segment_0.coef_[:, 0] * price_range * (1 - purchase_pr_segment_0)

# results price elasticity of purchase probability
df_price_elasticities['PE_Segment_0'] = pe_segment_0
print(df_price_elasticities)

plt.figure(figsize=(9,6))
plt.plot(price_range, pe, color='grey')
plt.plot(price_range, pe_segment_1, color='green')
plt.plot(price_range, pe_segment_2, color='r')
plt.plot(price_range, pe_segment_0, color='blue')
plt.xlabel('Price')
plt.ylabel('Elasticity')
plt.title('Price Elasticity of Purchase Probability')
plt.show()

## Segment 3 - Well-Off ##
# select only customers drom segment 3
df_pa_segment_3 = df_pa[df_pa['Segment'] == 3]

# build logistic regression model
Y = df_pa_segment_3['Incidence']
X = pd.DataFrame()
X['Mean_Price'] = (df_pa_segment_3['Price_1'] +
                   df_pa_segment_3['Price_2'] +
                   df_pa_segment_3['Price_3'] +
                   df_pa_segment_3['Price_4'] +
                   df_pa_segment_3['Price_5'] ) / 5

model_incidence_segment_3 = LogisticRegression(solver='sag')
model_incidence_segment_3.fit(X, Y)

print(model_incidence_segment_3.coef_)

# purchase probability by segment 3
Y_segment_3 = model_incidence_segment_3.predict_proba(df_price_range)
purchase_pr_segment_3 = Y_segment_3[:][:, 1]

# price elasticity of purchase probability by segment 3
pe_segment_3 = model_incidence_segment_3.coef_[:, 0] * price_range * (1 - purchase_pr_segment_3)

# results price elasticity of purchase probability
df_price_elasticities['PE_Segment_3'] = pe_segment_3
print(df_price_elasticities)

plt.figure(figsize=(9,6))
plt.plot(price_range, pe, color='grey')
plt.plot(price_range, pe_segment_1, color='green')
plt.plot(price_range, pe_segment_2, color='r')
plt.plot(price_range, pe_segment_0, color='blue')
plt.plot(price_range, pe_segment_3, color='orange')
plt.xlabel('Price')
plt.ylabel('Elasticity')
plt.title('Price Elasticity of Purchase Probability')
plt.show()


### ----- Purchase Probability With Promotion Feature ----- ###

## Data Preparation
Y = df_pa['Incidence']

X = pd.DataFrame()
X['Mean_Price'] = (df_pa['Price_1'] +
                   df_pa['Price_2'] +
                   df_pa['Price_3'] +
                   df_pa['Price_4'] +
                   df_pa['Price_5'] ) / 5

X['Mean_Promotion'] = (df_pa['Promotion_1'] +
                       df_pa['Promotion_2'] +
                       df_pa['Promotion_3'] +
                       df_pa['Promotion_4'] +
                       df_pa['Promotion_5'] ) / 5
print(X) 

## Model Estimation
model_incidence_promotion = LogisticRegression(solver= 'sag')
model_incidence_promotion.fit(X, Y)
print(model_incidence_promotion.coef_)

## hasil: The two coefficient values are minus one point four nine for price and zero point five six for promotion. Again, we have a negative coefficient for the price.
#  On the other hand, the promotion coefficient is positive, meaning that with the increase in promotion, the purchase probability also increases.


### Price Elasticity with Promotion ###

## Create Data Price and Promotion untuk predict 
df_price_elasticity_promotion = pd.DataFrame(price_range)
df_price_elasticity_promotion = df_price_elasticity_promotion.rename(columns={0: "Price_Range"})
df_price_elasticity_promotion['Promotion'] = 1
print(df_price_elasticity_promotion)

## Purchase Probability with variable price and promotionn
Y_promotion = model_incidence_promotion.predict_proba(df_price_elasticity_promotion)
print(Y_promotion)

## Ambil Purchase Probability untuk binary 1=terjadi pembelian produk of interest
promo = Y_promotion[:, 1]
print(promo)

## Price Elasticity with Promotion
price_elasticity_promo = (model_incidence_promotion.coef_[:, 0] * price_range) * (1 - promo)

df_price_elasticities['Elasticity_Promotion_1'] = price_elasticity_promo
print(df_price_elasticities)


### Price Elasticity without Promotion ###

## Create Data Price and Promotion untuk predict 
df_price_elasticity_no_promotion = pd.DataFrame(price_range)
df_price_elasticity_no_promotion = df_price_elasticity_no_promotion.rename(columns={0: "Price_Range"})
df_price_elasticity_no_promotion['Promotion'] = 0
print(df_price_elasticity_no_promotion)

## Purchase Probability with variable price and promotionn
Y_no_promotion = model_incidence_promotion.predict_proba(df_price_elasticity_no_promotion)
print(Y_no_promotion)

## Ambil Purchase Probability untuk binary 1=terjadi pembelian produk of interest
no_promo = Y_no_promotion[:, 1]
print(no_promo)

## Price Elasticity without Promotion
price_elasticity_no_promo = (model_incidence_promotion.coef_[:, 0] * price_range) * (1 - no_promo)

df_price_elasticities['Elasticity_Promotion_0'] = price_elasticity_no_promo
print(df_price_elasticities)

### Grafik Price Elasticities With Promotion and Without Promotion ###
plt.figure(figsize=(9,6))
plt.plot(price_range, price_elasticity_no_promo, color='blue')
plt.plot(price_range, price_elasticity_promo, color='red')
plt.xlabel('Price')
plt.ylabel('Elasticity')
plt.title('Price Elasticity of Purchase Probability With and Without Promotion')
plt.show()

## Hasil: This graph here tells us that the elasticity curve with promotion sits above its respective no promotion counterpart for the entire price range. Additionally, if we
#  consult our master data frame, we can see that in elasticity for no promotion ends at one dollar twenty seven while for promotion at one point forty six. That's a difference
#  of almost 20 cents. So if a product has a regular price of one point thirty dollars, the purchase probability is elastic. However, if a product cost one point fifty dollars
#  on the regular and its price is reduced to the aforementioned one point thirty dollars during a promotion, then our analysis says that purchase probability is still
#  inelastic at this point. This may sound trivial, but it is not. Think about it. People are more willing to buy products at promotional prices, be it because of the large
#  discount signs in some stores, or just because psychologically people feel they're getting a bargain. In any case, customers are less price sensitive to similar price
#  changes when there are promotion activities. In other words, it pays off to offer discounts. According to this model, if we could INC, it would be more beneficial to have
#  a higher original price and constant promotion rather than a lower original price. 


### ----- Price Elasticity of Purchase Probability by Segment With Promotion ----- ###
# segment 0: standard, segment 1: career focused, segment 2: fewer opportunities, career 3: well off

## Segment 1 - Career-Focused ##
# build logistic regression model
df_pa_segment_1 = df_pa[df_pa['Segment'] == 1]

Y = df_pa_segment_1['Incidence']

X = pd.DataFrame()
X['Mean_Price'] = (df_pa_segment_1['Price_1'] +
                   df_pa_segment_1['Price_2'] +
                   df_pa_segment_1['Price_3'] +
                   df_pa_segment_1['Price_4'] +
                   df_pa_segment_1['Price_5'] ) / 5


X['Mean_Promotion'] = (df_pa_segment_1['Promotion_1'] +
                       df_pa_segment_1['Promotion_2'] +
                       df_pa_segment_1['Promotion_3'] +
                       df_pa_segment_1['Promotion_4'] +
                       df_pa_segment_1['Promotion_5'] ) / 5

model_incidence_segment_1_promo = LogisticRegression(solver='sag')
model_incidence_segment_1_promo.fit(X, Y)
print(model_incidence_segment_1_promo.coef_)

## Purchase Probability with variable price and promotionn
Y_segment_1_promo = model_incidence_segment_1_promo.predict_proba(df_price_elasticity_promotion)
print(Y_segment_1_promo)

## Ambil Purchase Probability untuk binary 1=terjadi pembelian produk of interest
segment_1_promo= Y_segment_1_promo[:, 1]
print(segment_1_promo)

## Price Elasticity with Promotion
price_elasticity_segment_1_promo = (model_incidence_segment_1_promo.coef_[:, 0] * price_range) * (1 - segment_1_promo)

df_price_elasticities['PE_Promo_1_Segment_1'] = price_elasticity_segment_1_promo
print(df_price_elasticities)

plt.figure(figsize=(9,6))
plt.plot(price_range, price_elasticity_promo, color='red')
plt.plot(price_range, price_elasticity_segment_1_promo, color='green')
plt.xlabel('Price')
plt.ylabel('Elasticity')
plt.title('Price Elasticity of Purchase Probability With Promotion')
plt.show()

## Segment 2 - Fewer-Opportunities ##
# build logistic regression model
df_pa_segment_2 = df_pa[df_pa['Segment'] == 2]

Y = df_pa_segment_2['Incidence']

X = pd.DataFrame()
X['Mean_Price'] = (df_pa_segment_2['Price_1'] +
                   df_pa_segment_2['Price_2'] +
                   df_pa_segment_2['Price_3'] +
                   df_pa_segment_2['Price_4'] +
                   df_pa_segment_2['Price_5'] ) / 5


X['Mean_Promotion'] = (df_pa_segment_2['Promotion_1'] +
                       df_pa_segment_2['Promotion_2'] +
                       df_pa_segment_2['Promotion_3'] +
                       df_pa_segment_2['Promotion_4'] +
                       df_pa_segment_2['Promotion_5'] ) / 5

model_incidence_segment_2_promo = LogisticRegression(solver='sag')
model_incidence_segment_2_promo.fit(X, Y)

print(model_incidence_segment_2_promo.coef_)

## Purchase Probability with variable price and promotionn
Y_segment_2_promo = model_incidence_segment_2_promo.predict_proba(df_price_elasticity_promotion)
print(Y_segment_2_promo)

## Ambil Purchase Probability untuk binary 1=terjadi pembelian produk of interest
segment_2_promo= Y_segment_2_promo[:, 1]
print(segment_2_promo)

## Price Elasticity with Promotion
price_elasticity_segment_2_promo = (model_incidence_segment_2_promo.coef_[:, 0] * price_range) * (1 - segment_2_promo)

df_price_elasticities['PE_Promo_1_Segment_2'] = price_elasticity_segment_2_promo
print(df_price_elasticities)

plt.figure(figsize=(9,6))
plt.plot(price_range, price_elasticity_promo, color='red')
plt.plot(price_range, price_elasticity_segment_1_promo, color='green')
plt.plot(price_range, price_elasticity_segment_2_promo, color='blue')
plt.xlabel('Price')
plt.ylabel('Elasticity')
plt.title('Price Elasticity of Purchase Probability With Promotion')
plt.show()

## Segment 3 - Well-Off ##
df_pa_segment_3 = df_pa[df_pa['Segment'] == 3]

Y = df_pa_segment_3['Incidence']

X = pd.DataFrame()
X['Mean_Price'] = (df_pa_segment_3['Price_1'] +
                   df_pa_segment_3['Price_2'] +
                   df_pa_segment_3['Price_3'] +
                   df_pa_segment_3['Price_4'] +
                   df_pa_segment_3['Price_5'] ) / 5


X['Mean_Promotion'] = (df_pa_segment_3['Promotion_1'] +
                       df_pa_segment_3['Promotion_2'] +
                       df_pa_segment_3['Promotion_3'] +
                       df_pa_segment_3['Promotion_4'] +
                       df_pa_segment_3['Promotion_5'] ) / 5

model_incidence_segment_3_promo = LogisticRegression(solver='sag')
model_incidence_segment_3_promo.fit(X, Y)

print(model_incidence_segment_3_promo.coef_)

## Purchase Probability with variable price and promotionn
Y_segment_3_promo = model_incidence_segment_3_promo.predict_proba(df_price_elasticity_promotion)
print(Y_segment_3_promo)

## Ambil Purchase Probability untuk binary 1=terjadi pembelian produk of interest
segment_3_promo= Y_segment_3_promo[:, 1]
print(segment_3_promo)

## Price Elasticity with Promotion
price_elasticity_segment_3_promo = (model_incidence_segment_3_promo.coef_[:, 0] * price_range) * (1 - segment_3_promo)

df_price_elasticities['PE_Promo_1_Segment_3'] = price_elasticity_segment_3_promo
print(df_price_elasticities)


plt.figure(figsize=(9,6))
plt.plot(price_range, price_elasticity_promo, color='red')
plt.plot(price_range, price_elasticity_segment_1_promo, color='green')
plt.plot(price_range, price_elasticity_segment_2_promo, color='blue')
plt.plot(price_range, price_elasticity_segment_3_promo, color='orange')
plt.xlabel('Price')
plt.ylabel('Elasticity')
plt.title('Price Elasticity of Purchase Probability With Promotion')
plt.show()

## Segment 0 - Standard ##
df_pa_segment_0 = df_pa[df_pa['Segment'] == 0]

Y = df_pa_segment_0['Incidence']

X = pd.DataFrame()
X['Mean_Price'] = (df_pa_segment_0['Price_1'] +
                   df_pa_segment_0['Price_2'] +
                   df_pa_segment_0['Price_3'] +
                   df_pa_segment_0['Price_4'] +
                   df_pa_segment_0['Price_5'] ) / 5


X['Mean_Promotion'] = (df_pa_segment_0['Promotion_1'] +
                       df_pa_segment_0['Promotion_2'] +
                       df_pa_segment_0['Promotion_3'] +
                       df_pa_segment_0['Promotion_4'] +
                       df_pa_segment_0['Promotion_5'] ) / 5

model_incidence_segment_0_promo = LogisticRegression(solver='sag')
model_incidence_segment_0_promo.fit(X, Y)

print(model_incidence_segment_0_promo.coef_)

## Purchase Probability with variable price and promotionn
Y_segment_0_promo = model_incidence_segment_0_promo.predict_proba(df_price_elasticity_promotion)
print(Y_segment_0_promo)

## Ambil Purchase Probability untuk binary 1=terjadi pembelian produk of interest
segment_0_promo= Y_segment_0_promo[:, 1]
print(segment_0_promo)

## Price Elasticity with Promotion
price_elasticity_segment_0_promo = (model_incidence_segment_0_promo.coef_[:, 0] * price_range) * (1 - segment_0_promo)

df_price_elasticities['PE_Promo_1_Segment_0'] = price_elasticity_segment_0_promo
print(df_price_elasticities)

plt.figure(figsize=(9,6))
plt.plot(price_range, price_elasticity_promo, color='red')
plt.plot(price_range, price_elasticity_segment_1_promo, color='green')
plt.plot(price_range, price_elasticity_segment_2_promo, color='blue')
plt.plot(price_range, price_elasticity_segment_3_promo, color='orange')
plt.plot(price_range, price_elasticity_segment_0_promo, color='yellow')
plt.xlabel('Price')
plt.ylabel('Elasticity')
plt.title('Price Elasticity of Purchase Probability With Promotion')
plt.show()


### ----- Price Elasticity of Purchase Probability by Segment Without Promotion ----- ###
# segment 0: standard, segment 1: career focused, segment 2: fewer opportunities, career 3: well off

## Segment 1 - Career-Focused ##
# build logistic regression model
df_pa_segment_1 = df_pa[df_pa['Segment'] == 1]

Y = df_pa_segment_1['Incidence']

X = pd.DataFrame()
X['Mean_Price'] = (df_pa_segment_1['Price_1'] +
                   df_pa_segment_1['Price_2'] +
                   df_pa_segment_1['Price_3'] +
                   df_pa_segment_1['Price_4'] +
                   df_pa_segment_1['Price_5'] ) / 5


X['Mean_Promotion'] = (df_pa_segment_1['Promotion_1'] +
                       df_pa_segment_1['Promotion_2'] +
                       df_pa_segment_1['Promotion_3'] +
                       df_pa_segment_1['Promotion_4'] +
                       df_pa_segment_1['Promotion_5'] ) / 5

model_incidence_segment_1_no_promo = LogisticRegression(solver='sag')
model_incidence_segment_1_no_promo.fit(X, Y)
print(model_incidence_segment_1_no_promo.coef_)

## Purchase Probability with variable price and promotionn
Y_segment_1_no_promo = model_incidence_segment_1_no_promo.predict_proba(df_price_elasticity_no_promotion)
print(Y_segment_1_no_promo)

## Ambil Purchase Probability untuk binary 1=terjadi pembelian produk of interest
segment_1_no_promo= Y_segment_1_no_promo[:, 1]
print(segment_1_no_promo)

## Price Elasticity without Promotion
price_elasticity_segment_1_no_promo = (model_incidence_segment_1_no_promo.coef_[:, 0] * price_range) * (1 - segment_1_no_promo)

df_price_elasticities['PE_Promo_0_Segment_1'] = price_elasticity_segment_1_no_promo
print(df_price_elasticities)

plt.figure(figsize=(9,6))
plt.plot(price_range, price_elasticity_no_promo, color='red')
plt.plot(price_range, price_elasticity_segment_1_no_promo, color='green')
plt.xlabel('Price')
plt.ylabel('Elasticity')
plt.title('Price Elasticity of Purchase Probability Without Promotion')
plt.show()

## Segment 2 - Fewer-Opportunities ##
# build logistic regression model
df_pa_segment_2 = df_pa[df_pa['Segment'] == 2]

Y = df_pa_segment_2['Incidence']

X = pd.DataFrame()
X['Mean_Price'] = (df_pa_segment_2['Price_1'] +
                   df_pa_segment_2['Price_2'] +
                   df_pa_segment_2['Price_3'] +
                   df_pa_segment_2['Price_4'] +
                   df_pa_segment_2['Price_5'] ) / 5


X['Mean_Promotion'] = (df_pa_segment_2['Promotion_1'] +
                       df_pa_segment_2['Promotion_2'] +
                       df_pa_segment_2['Promotion_3'] +
                       df_pa_segment_2['Promotion_4'] +
                       df_pa_segment_2['Promotion_5'] ) / 5

model_incidence_segment_2_no_promo = LogisticRegression(solver='sag')
model_incidence_segment_2_no_promo.fit(X, Y)

print(model_incidence_segment_2_no_promo.coef_)

## Purchase Probability with variable price and promotionn
Y_segment_2_no_promo = model_incidence_segment_2_no_promo.predict_proba(df_price_elasticity_no_promotion)
print(Y_segment_2_no_promo)

## Ambil Purchase Probability untuk binary 1=terjadi pembelian produk of interest
segment_2_no_promo= Y_segment_2_no_promo[:, 1]
print(segment_2_no_promo)

## Price Elasticity without Promotion
price_elasticity_segment_2_no_promo = (model_incidence_segment_2_no_promo.coef_[:, 0] * price_range) * (1 - segment_2_no_promo)

df_price_elasticities['PE_Promo_0_Segment_2'] = price_elasticity_segment_2_no_promo
print(df_price_elasticities)

plt.figure(figsize=(9,6))
plt.plot(price_range, price_elasticity_no_promo, color='red')
plt.plot(price_range, price_elasticity_segment_1_no_promo, color='green')
plt.plot(price_range, price_elasticity_segment_2_no_promo, color='blue')
plt.xlabel('Price')
plt.ylabel('Elasticity')
plt.title('Price Elasticity of Purchase Probability Without Promotion')
plt.show()

## Segment 3 - Well-Off ##
df_pa_segment_3 = df_pa[df_pa['Segment'] == 3]

Y = df_pa_segment_3['Incidence']

X = pd.DataFrame()
X['Mean_Price'] = (df_pa_segment_3['Price_1'] +
                   df_pa_segment_3['Price_2'] +
                   df_pa_segment_3['Price_3'] +
                   df_pa_segment_3['Price_4'] +
                   df_pa_segment_3['Price_5'] ) / 5


X['Mean_Promotion'] = (df_pa_segment_3['Promotion_1'] +
                       df_pa_segment_3['Promotion_2'] +
                       df_pa_segment_3['Promotion_3'] +
                       df_pa_segment_3['Promotion_4'] +
                       df_pa_segment_3['Promotion_5'] ) / 5

model_incidence_segment_3_no_promo = LogisticRegression(solver='sag')
model_incidence_segment_3_no_promo.fit(X, Y)

print(model_incidence_segment_3_no_promo.coef_)

## Purchase Probability with variable price and promotionn
Y_segment_3_no_promo = model_incidence_segment_3_no_promo.predict_proba(df_price_elasticity_no_promotion)
print(Y_segment_3_no_promo)

## Ambil Purchase Probability untuk binary 1=terjadi pembelian produk of interest
segment_3_no_promo= Y_segment_3_no_promo[:, 1]
print(segment_3_no_promo)

## Price Elasticity without Promotion
price_elasticity_segment_3_no_promo = (model_incidence_segment_3_no_promo.coef_[:, 0] * price_range) * (1 - segment_3_no_promo)

df_price_elasticities['PE_Promo_0_Segment_3'] = price_elasticity_segment_3_no_promo
print(df_price_elasticities)


plt.figure(figsize=(9,6))
plt.plot(price_range, price_elasticity_no_promo, color='red')
plt.plot(price_range, price_elasticity_segment_1_no_promo, color='green')
plt.plot(price_range, price_elasticity_segment_2_no_promo, color='blue')
plt.plot(price_range, price_elasticity_segment_3_no_promo, color='orange')
plt.xlabel('Price')
plt.ylabel('Elasticity')
plt.title('Price Elasticity of Purchase Probability Without Promotion')
plt.show()

## Segment 0 - Standard ##
df_pa_segment_0 = df_pa[df_pa['Segment'] == 0]

Y = df_pa_segment_0['Incidence']

X = pd.DataFrame()
X['Mean_Price'] = (df_pa_segment_0['Price_1'] +
                   df_pa_segment_0['Price_2'] +
                   df_pa_segment_0['Price_3'] +
                   df_pa_segment_0['Price_4'] +
                   df_pa_segment_0['Price_5'] ) / 5


X['Mean_Promotion'] = (df_pa_segment_0['Promotion_1'] +
                       df_pa_segment_0['Promotion_2'] +
                       df_pa_segment_0['Promotion_3'] +
                       df_pa_segment_0['Promotion_4'] +
                       df_pa_segment_0['Promotion_5'] ) / 5

model_incidence_segment_0_no_promo = LogisticRegression(solver='sag')
model_incidence_segment_0_no_promo.fit(X, Y)

print(model_incidence_segment_0_no_promo.coef_)

## Purchase Probability with variable price and promotionn
Y_segment_0_no_promo = model_incidence_segment_0_no_promo.predict_proba(df_price_elasticity_no_promotion)
print(Y_segment_0_no_promo)

## Ambil Purchase Probability untuk binary 1=terjadi pembelian produk of interest
segment_0_no_promo= Y_segment_0_no_promo[:, 1]
print(segment_0_no_promo)

## Price Elasticity without Promotion
price_elasticity_segment_0_no_promo = (model_incidence_segment_0_no_promo.coef_[:, 0] * price_range) * (1 - segment_0_no_promo)

df_price_elasticities['PE_Promo_0_Segment_0'] = price_elasticity_segment_0_no_promo
print(df_price_elasticities)

plt.figure(figsize=(9,6))
plt.plot(price_range, price_elasticity_no_promo, color='red')
plt.plot(price_range, price_elasticity_segment_1_no_promo, color='green')
plt.plot(price_range, price_elasticity_segment_2_no_promo, color='blue')
plt.plot(price_range, price_elasticity_segment_3_no_promo, color='orange')
plt.plot(price_range, price_elasticity_segment_0_no_promo, color='yellow')
plt.xlabel('Price')
plt.ylabel('Elasticity')
plt.title('Price Elasticity of Purchase Probability Without Promotion')
plt.show()
 