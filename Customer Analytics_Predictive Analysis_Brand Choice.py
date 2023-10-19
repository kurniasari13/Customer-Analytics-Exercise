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

# membuat dataframe baru untuk predict
print(df_pa[['Price_1', 'Price_2', 'Price_3', 'Price_4', 'Price_5']].describe())
price_range = np.arange(0.5, 3.5, 0.01)

# membuat master data untuk price elasticities
df_price_elasticities = pd.DataFrame(price_range)
df_price_elasticities = df_price_elasticities.rename(columns={0:"Price_Point"})


### ----- BRAND CHOICE ----- ###

# select only shopping occasion when there was a purchase.
brand_choice = df_pa[df_pa['Incidence'] == 1]
pd.options.display.max_rows = 100
print(brand_choice)

# build the model
Y = brand_choice['Brand']
#print(brand_choice.columns.values)
features =['Price_1', 'Price_2', 'Price_3', 'Price_4', 'Price_5']
X = brand_choice[features]

model_brand_choice = LogisticRegression(solver='sag', multi_class='multinomial')
model_brand_choice.fit(X, Y) 
#print(model_brand_choice.coef_)

bc_coef = pd.DataFrame(model_brand_choice.coef_)
bc_coef = pd.DataFrame(np.transpose(model_brand_choice.coef_))
coefficients = ['Coef_Brand_1', 'Coef_Brand_2', 'Coef_Brand_3', 'Coef_Brand_4', 'Coef_Brand_5']
bc_coef.columns = [coefficients]
prices = ['Price_1', 'Price_2', 'Price_3', 'Price_4', 'Price_5']
bc_coef.index = [prices]
bc_coef = bc_coef.round(2)
print(bc_coef)

## Hasil: Let's start with brand one, the coefficient for the own brand with respect to prices, negative, while it's positive for all other prices except five, we already know
#  that the higher the price of her own product, the lower the probability for it to be purchased. So it makes sense for the brand price coefficient to be negative. On the
#  other hand, the more the price of a competitor increases, the higher the probability of customers switching to our own brand would be. Hence, there is a positive relationship
#  between our own brands purchase probability and a competitor brand increasing their price. Well, at this point, you may have realized that the choice probability for any one
#  brand and the choice probabilities for all the other brands are interrelated. And a marketing mix tool of our brand reflects not only the choice probability for that brand,
#  but the choice probabilities for all other brands as well. These effects are known as own brand effects and cross brand effects. We'll examine these in more detail when we
#  compute their respective elasticities.


### ----- OWN PRICE ELASTICITY BRAND 5 ----- ###

# membuat dataframe baru untuk predict probability brand choice for brand 5
df_own_brand_5 = pd.DataFrame(index= np.arange(price_range.size))
df_own_brand_5['Price_1'] = brand_choice['Price_1'].mean()
df_own_brand_5['Price_2'] = brand_choice['Price_2'].mean()
df_own_brand_5['Price_3'] = brand_choice['Price_3'].mean()
df_own_brand_5['Price_4'] = brand_choice['Price_4'].mean()
df_own_brand_5['Price_5'] = price_range
print(df_own_brand_5)

# predict probability brand choice for brand 5
predict_brand_5 = model_brand_choice.predict_proba(df_own_brand_5)

# extract only probability brand choice for brand 5
pr_own_brand_5 = predict_brand_5[:][:, 4]

# finding the elasticity of probability brand choice for brand 5
beta5 = bc_coef.iloc[4,4]
print(beta5)

own_pe_brand_5 = beta5 * price_range * (1 - pr_own_brand_5)
df_price_elasticities['Brand_5'] = own_pe_brand_5
#pd.options.display.max_rows = None
print(df_price_elasticities)

# build graph
plt.figure(figsize=(9,6))
plt.plot(price_range, own_pe_brand_5, color='grey')
plt.xlabel('Price 5')
plt.ylabel('Elasticity')
plt.title('Own Price Elasticity of Purchase Probability for Brand 5')
plt.show()

### ----- CROSS PRICE ELASTICITY BRAND 5, CROSS BRAND 4 ----- ###

df_brand5_cross_brand4 = pd.DataFrame(index= np.arange(price_range.size))
df_brand5_cross_brand4['Price_1'] = brand_choice['Price_1'].mean()
df_brand5_cross_brand4['Price_2'] = brand_choice['Price_2'].mean()
df_brand5_cross_brand4['Price_3'] = brand_choice['Price_3'].mean()
df_brand5_cross_brand4['Price_4'] = price_range
df_brand5_cross_brand4['Price_5'] = brand_choice['Price_5'].mean()
print(df_brand5_cross_brand4)

# predict probability brand choice for brand 4
predict_brand5_cross_brand4 = model_brand_choice.predict_proba(df_brand5_cross_brand4)

# extract only probability brand choice for brand 4
pr_brand_4 = predict_brand5_cross_brand4[:][:, 3]

# finding the elasticity of probability brand choice for brand 5 cross brand 4
print(beta5) ## koeff dari price brand 5 dan y brand 5

brand5_cross_brand4_price_elasticity = -beta5 * price_range * pr_brand_4
df_price_elasticities['Brand_5_Cross_Brand_4'] = brand5_cross_brand4_price_elasticity
#pd.options.display.max_rows = None
print(df_price_elasticities)

# build graph
plt.figure(figsize=(9,6))
plt.plot(price_range, brand5_cross_brand4_price_elasticity, color='grey')
plt.plot(price_range, own_pe_brand_5, color='red')
plt.xlabel('Price 4')
plt.ylabel('Elasticity')
plt.title('Cross Price Elasticity of Purchase Probability for Brand 5 with respect to Brand 4 (Warna Abu-Abu)')
plt.show()

## Hasil: we observed that the elasticities are positive across the price range. This indicates that if competitor brand 4 increases prices, the purchase probability for our own
#  brand would increase. Our competitor raises prices and consumer start buying our product more. Now, if the cross price elasticity is greater than zero, the two products are
#  considered substitutes. That's logical as brand 4 and brand 5, both chocolate candy bars. If, however, we were looking at brand five cross some type of beer, for instance,
#  the cross price elasticity would not be necessarily positive as the two products have nothing in common. OK, in our example, all across price elasticities will be positive
#  as all brands are substitutes for one another. "Furthermore, if the price elasticity at some price point is greater in absolute terms than our own price elasticity, the
#  alternative brand is considered a strong substitute". So is brand for a strong substitute for brand five? Well, depends on the price point. You can graph the absolute values
#  of the two elasticities or examine the elasticity table instead. "For this case, Brand four is a strong substitute for brand five for all prices up to one dollar sixty five (0-1.65)".
#  """"""aku cek di tabel hasilnya:   1.66 -0.569102  0.564514, di harga 1.66 own price elasticity mulai lebih besar daripada cross elasticity""""""
#  However, we know that these prices are out of the natural domain of brand, for therefore if Brand 4 had a substantially lower price, it would be a very strong competitor for
#  a Brand five. "It is important to mark that the observed price range of Brand 4 lies between one point seventy six (1.76) and two point to six (2.6)", we observed the
#  elasticity is steadily decreasing. This signals that with an increase in price, the purchase probability changes more slowly. Note, however, it is positive. Therefore, our
#  purchase probability still increases with the increase in price of Brand 4, but at a slower rate. "With these observations in mind, we can conclude that when it comes to the
#  average customer, brand 4 is an albeit weak substitute for Brand five". In light of these results, Brand five can create a marketing strategy for targeting customers which
#  choose brand 4 and attract them to buy the own brand. However, we already know that targeting the average customer can be a labor some, if not, next to an impossible task, no
#  brand can make everyone happy, but a brand can make a certain customer segment happy, which is why the next lecture will further refine our model and compute Own and
#  cross price elasticities by customer segments.


### ----- OWN AND CROSS PRICE ELASTICITY FOR BRAND 5 BY SEGMENT ----- ###
# segment 0: standard, segment 1: career focused, segment 2: fewer opportunities, career 3: well off

## SEGMENT 3: WELL OFF ##

# data preparation
brand_choice_s3 = df_pa[df_pa['Incidence'] == 1]
brand_choice_s3 = brand_choice_s3[brand_choice_s3['Segment'] == 3]
print(brand_choice_s3)

# model estimation
Y = brand_choice_s3['Brand']
brand_choice_s3 = pd.get_dummies(brand_choice_s3, columns=['Brand'], prefix='Brand', prefix_sep= '_')
X = brand_choice_s3[features]

model_brand_choice_s3 = LogisticRegression(solver='sag', multi_class='multinomial', max_iter= 300)
model_brand_choice_s3.fit(X, Y)

# coefficient table for Brand 5 by segment 3
bc_coef = pd.DataFrame(np.transpose(model_brand_choice_s3.coef_))
bc_coef.columns = [coefficients]
bc_coef.index = [prices]
bc_coef = bc_coef.round(2)
print(bc_coef)

## Own Brand Price Elasticity for Brand 5 by Segment 3
# membuat dataframe baru untuk predict probability brand choice for brand 5
df_own_brand_5_s3 = pd.DataFrame(index= np.arange(price_range.size))
df_own_brand_5_s3['Price_1'] = brand_choice_s3['Price_1'].mean()
df_own_brand_5_s3['Price_2'] = brand_choice_s3['Price_2'].mean()
df_own_brand_5_s3['Price_3'] = brand_choice_s3['Price_3'].mean()
df_own_brand_5_s3['Price_4'] = brand_choice_s3['Price_4'].mean()
df_own_brand_5_s3['Price_5'] = price_range
print(df_own_brand_5_s3)

# predict probability brand choice for brand 5
predict_own_brand_5_s3 = model_brand_choice_s3.predict_proba(df_own_brand_5_s3)

# extract only probability brand choice for brand 5
pr_own_brand_5_s3 = predict_own_brand_5_s3[:][:, 4]

beta5_s3 = bc_coef.iloc[4,4]
print(beta5_s3)

# finding the elasticity of probability brand choice for brand 5
own_pe_brand_5_s3 = beta5_s3 * price_range * (1 - pr_own_brand_5_s3)
df_price_elasticities['Brand 5 S3'] = own_pe_brand_5_s3
#pd.options.display.max_rows = None
print(df_price_elasticities)

## Cross Brand Price Elasticity for Brand 5 terhadap brand 4 by Segment 3
df_brand5_cross_brand4_s3 = pd.DataFrame(index= np.arange(price_range.size))
df_brand5_cross_brand4_s3['Price_1'] = brand_choice_s3['Price_1'].mean()
df_brand5_cross_brand4_s3['Price_2'] = brand_choice_s3['Price_2'].mean()
df_brand5_cross_brand4_s3['Price_3'] = brand_choice_s3['Price_3'].mean()
df_brand5_cross_brand4_s3['Price_4'] = price_range
df_brand5_cross_brand4_s3['Price_5'] = brand_choice_s3['Price_5'].mean()
print(df_brand5_cross_brand4_s3)

# predict probability brand choice for brand 4
predict_brand5_cross_brand4_s3 = model_brand_choice_s3.predict_proba(df_brand5_cross_brand4_s3)

# extract only probability brand choice for brand 4
pr_cross_brand_5_s3 = predict_brand5_cross_brand4_s3[:][:, 3]

# finding the elasticity of probability brand choice for brand 5 cross brand 4
print(beta5_s3) ## koeff dari price brand 5 dan y brand 5 by segment 3

brand5_cross_brand4_price_elasticity_s3 = -beta5_s3 * price_range * pr_cross_brand_5_s3
df_price_elasticities['Brand_5_Cross_Brand_4_s3'] = brand5_cross_brand4_price_elasticity_s3
#pd.options.display.max_rows = None
print(df_price_elasticities)

# membuat plot own and cross price elasticity brand 5 by segment 3
fig, axs = plt.subplots(1, 2, figsize=(14, 4))
axs[0].plot(price_range, own_pe_brand_5_s3, color='orange')
axs[0].set_title('Brand 5 Segment Well-Off')
axs[0].set_xlabel('Price 5')

axs[1].plot(price_range, brand5_cross_brand4_price_elasticity_s3, color='orange')
axs[1].set_title('Cross Price Elasticity of Brand 5 wrt Brand 4 Segment Well-Off')
axs[1].set_xlabel('Price 4')

for ax in axs.flat:
    ax.set(ylabel='Elasticity')
plt.show()

## Hasil: Let me also remind you that the natural domain of prices of Brand five is from two point eleven to two point eight dollars (2.11-2.8). First, the Own price elasticity
#  indicates that the ""well-off customer is elastic to our own brand"". This was rather expected, as they seem to prefer brand 4. And if we check our descriptive analysis table,
#  we can verify that indeed, over 60 percent of the well-off segment purchased brand 4 at about 20 percent by brand five. Now to cross price elasticities. Again, they are
#  positive, indicating that for the ""well off brand 4 is a substitute for brand five"". We will focus on a novel idea based on both graphs. Let's first highlight the natural
#  domains of the prices of Brand five and brand four. On the two graphs say brand five costs two dollars for so own price elasticity is about minus two. Moreover, brand
#  4 cast two dollars, so the cross price elasticity is about one point five. What would happen if our competitor brand for LOWER prices by one percent? Well, the cross price
#  elasticity is one point five, so the purchase probability of our brand will fall by one point five percent. Sounds like a serious hit on our sales, but we can strike back.
#  We can lower our own price by one percent. In that case, we must look at the own price elasticity of our brand, since it is minus 2, a one percent decrease in our price
#  would be reflected in a two percent increase in purchase probability. The net effect of the two price decreases is two percent minus one point five percent or plus zero
#  point five percent. Therefore, we have reacted to our competitors price range and have actually gained some market share. In a similar way, knowing the price elasticities,
#  we can react to our competitor to keep the purchase probability constant. We establish that if Brant 4 decreases their price by one percent, the purchase probability for
#  our brand would decrease by one point five percent. To match that by our own price decrease, we can make an equation X times two equals one point five percent.
#  In this equation, X is the decrease in price we require to reach a one point five percent increase in purchase probability. The answer is one point five percent divided by
#  two or zero point seventy five percent. Therefore, if Brant 4 decreases their price by one percent, we can decrease ours by zero point seventy five percent and theoretically
#  we won't lose a single customer from the well off segment.


## SEGMENT 0: STANDARD ##

# data preparation
brand_choice_s0 = df_pa[df_pa['Incidence'] == 1]
brand_choice_s0 = brand_choice_s0[brand_choice_s0['Segment'] == 0]
print(brand_choice_s0)

# model estimation
Y = brand_choice_s0['Brand']
brand_choice_s0 = pd.get_dummies(brand_choice_s0, columns=['Brand'], prefix='Brand', prefix_sep= '_')
X = brand_choice_s0[features]

model_brand_choice_s0 = LogisticRegression(solver='sag', multi_class='multinomial', max_iter= 300)
model_brand_choice_s0.fit(X, Y)

# coefficient table for Brand 5 by segment 0
bc_coef = pd.DataFrame(np.transpose(model_brand_choice_s0.coef_))
bc_coef.columns = [coefficients]
bc_coef.index = [prices]
bc_coef = bc_coef.round(2)
print(bc_coef)

## Own Brand Price Elasticity for Brand 5 by Segment 0
# membuat dataframe baru untuk predict probability brand choice for brand 5
df_own_brand_5_s0 = pd.DataFrame(index= np.arange(price_range.size))
df_own_brand_5_s0['Price_1'] = brand_choice_s0['Price_1'].mean()
df_own_brand_5_s0['Price_2'] = brand_choice_s0['Price_2'].mean()
df_own_brand_5_s0['Price_3'] = brand_choice_s0['Price_3'].mean()
df_own_brand_5_s0['Price_4'] = brand_choice_s0['Price_4'].mean()
df_own_brand_5_s0['Price_5'] = price_range
print(df_own_brand_5_s0)

# predict probability brand choice for brand 5
predict_own_brand_5_s0 = model_brand_choice_s0.predict_proba(df_own_brand_5_s0)

# extract only probability brand choice for brand 5
pr_own_brand_5_s0 = predict_own_brand_5_s0[:][:, 4]

beta5_s0 = bc_coef.iloc[4,4]
print(beta5_s0)

# finding the elasticity of probability brand choice for brand 5
own_pe_brand_5_s0 = beta5_s0 * price_range * (1 - pr_own_brand_5_s0)
df_price_elasticities['Brand 5 S0'] = own_pe_brand_5_s0
#pd.options.display.max_rows = None
print(df_price_elasticities)

## Cross Brand Price Elasticity for Brand 5 terhadap brand 4 by Segment 0
df_brand5_cross_brand4_s0 = pd.DataFrame(index= np.arange(price_range.size))
df_brand5_cross_brand4_s0['Price_1'] = brand_choice_s0['Price_1'].mean()
df_brand5_cross_brand4_s0['Price_2'] = brand_choice_s0['Price_2'].mean()
df_brand5_cross_brand4_s0['Price_3'] = brand_choice_s0['Price_3'].mean()
df_brand5_cross_brand4_s0['Price_4'] = price_range
df_brand5_cross_brand4_s0['Price_5'] = brand_choice_s0['Price_5'].mean()
print(df_brand5_cross_brand4_s0)

# predict probability brand choice for brand 4
predict_brand5_cross_brand4_s0 = model_brand_choice_s0.predict_proba(df_brand5_cross_brand4_s0)

# extract only probability brand choice for brand 4
pr_cross_brand_5_s0 = predict_brand5_cross_brand4_s0[:][:, 3]

# finding the elasticity of probability brand choice for brand 5 cross brand 4
print(beta5_s0) ## koeff dari price brand 5 dan y brand 5 by segment 0

brand5_cross_brand4_price_elasticity_s0 = -beta5_s0 * price_range * pr_cross_brand_5_s0
df_price_elasticities['Brand_5_Cross_Brand_4_s0'] = brand5_cross_brand4_price_elasticity_s0
#pd.options.display.max_rows = None
print(df_price_elasticities)


## SEGMENT 1: CAREER FOCUSED ##

# data preparation
brand_choice_s1 = df_pa[df_pa['Incidence'] == 1]
brand_choice_s1 = brand_choice_s1[brand_choice_s1['Segment'] == 1]
print(brand_choice_s1)

# model estimation
Y = brand_choice_s1['Brand']
brand_choice_s1 = pd.get_dummies(brand_choice_s1, columns=['Brand'], prefix='Brand', prefix_sep= '_')
X = brand_choice_s1[features]

model_brand_choice_s1 = LogisticRegression(solver='sag', multi_class='multinomial', max_iter= 300)
model_brand_choice_s1.fit(X, Y)

# coefficient table for Brand 5 by segment 1
bc_coef = pd.DataFrame(np.transpose(model_brand_choice_s1.coef_))
bc_coef.columns = [coefficients]
bc_coef.index = [prices]
bc_coef = bc_coef.round(2)
print(bc_coef)

## Own Brand Price Elasticity for Brand 5 by Segment 1
# membuat dataframe baru untuk predict probability brand choice for brand 5
df_own_brand_5_s1 = pd.DataFrame(index= np.arange(price_range.size))
df_own_brand_5_s1['Price_1'] = brand_choice_s1['Price_1'].mean()
df_own_brand_5_s1['Price_2'] = brand_choice_s1['Price_2'].mean()
df_own_brand_5_s1['Price_3'] = brand_choice_s1['Price_3'].mean()
df_own_brand_5_s1['Price_4'] = brand_choice_s1['Price_4'].mean()
df_own_brand_5_s1['Price_5'] = price_range
print(df_own_brand_5_s1)

# predict probability brand choice for brand 5
predict_own_brand_5_s1 = model_brand_choice_s1.predict_proba(df_own_brand_5_s1)

# extract only probability brand choice for brand 5
pr_own_brand_5_s1 = predict_own_brand_5_s1[:][:, 4]

beta5_s1 = bc_coef.iloc[4,4]
print(beta5_s1)

# finding the elasticity of probability brand choice for brand 5
own_pe_brand_5_s1 = beta5_s1 * price_range * (1 - pr_own_brand_5_s1)
df_price_elasticities['Brand 5 S1'] = own_pe_brand_5_s1
#pd.options.display.max_rows = None
print(df_price_elasticities)

## Cross Brand Price Elasticity for Brand 5 terhadap brand 4 by Segment 1
df_brand5_cross_brand4_s1 = pd.DataFrame(index= np.arange(price_range.size))
df_brand5_cross_brand4_s1['Price_1'] = brand_choice_s1['Price_1'].mean()
df_brand5_cross_brand4_s1['Price_2'] = brand_choice_s1['Price_2'].mean()
df_brand5_cross_brand4_s1['Price_3'] = brand_choice_s1['Price_3'].mean()
df_brand5_cross_brand4_s1['Price_4'] = price_range
df_brand5_cross_brand4_s1['Price_5'] = brand_choice_s1['Price_5'].mean()
print(df_brand5_cross_brand4_s1)

# predict probability brand choice for brand 4
predict_brand5_cross_brand4_s1 = model_brand_choice_s1.predict_proba(df_brand5_cross_brand4_s1)

# extract only probability brand choice for brand 4
pr_cross_brand_5_s1 = predict_brand5_cross_brand4_s1[:][:, 3]

# finding the elasticity of probability brand choice for brand 5 cross brand 4
print(beta5_s1) ## koeff dari price brand 5 dan y brand 5 by segment 1

brand5_cross_brand4_price_elasticity_s1 = -beta5_s1 * price_range * pr_cross_brand_5_s1
df_price_elasticities['Brand_5_Cross_Brand_4_s1'] = brand5_cross_brand4_price_elasticity_s1
#pd.options.display.max_rows = None
print(df_price_elasticities)


## SEGMENT 2: FEWER OPPORTUNITIES ##

# data preparation
brand_choice_s2 = df_pa[df_pa['Incidence'] == 1]
brand_choice_s2 = brand_choice_s2[brand_choice_s2['Segment'] == 2]
print(brand_choice_s2)

# model estimation
Y = brand_choice_s2['Brand']
brand_choice_s2 = pd.get_dummies(brand_choice_s2, columns=['Brand'], prefix='Brand', prefix_sep= '_')
X = brand_choice_s2[features]

model_brand_choice_s2 = LogisticRegression(solver='sag', multi_class='multinomial', max_iter= 300)
model_brand_choice_s2.fit(X, Y)

# coefficient table for Brand 5 by segment 2
bc_coef = pd.DataFrame(np.transpose(model_brand_choice_s2.coef_))
bc_coef.columns = [coefficients]
bc_coef.index = [prices]
bc_coef = bc_coef.round(2)
print(bc_coef)

## Own Brand Price Elasticity for Brand 5 by Segment 2
# membuat dataframe baru untuk predict probability brand choice for brand 5
df_own_brand_5_s2 = pd.DataFrame(index= np.arange(price_range.size))
df_own_brand_5_s2['Price_1'] = brand_choice_s2['Price_1'].mean()
df_own_brand_5_s2['Price_2'] = brand_choice_s2['Price_2'].mean()
df_own_brand_5_s2['Price_3'] = brand_choice_s2['Price_3'].mean()
df_own_brand_5_s2['Price_4'] = brand_choice_s2['Price_4'].mean()
df_own_brand_5_s2['Price_5'] = price_range
print(df_own_brand_5_s2)

# predict probability brand choice for brand 5
predict_own_brand_5_s2 = model_brand_choice_s2.predict_proba(df_own_brand_5_s2)

# extract only probability brand choice for brand 5
pr_own_brand_5_s2 = predict_own_brand_5_s2[:][:, 4]

beta5_s2 = bc_coef.iloc[4,4]
print(beta5_s2)

# finding the elasticity of probability brand choice for brand 5
own_pe_brand_5_s2 = beta5_s2 * price_range * (1 - pr_own_brand_5_s2)
df_price_elasticities['Brand 5 S2'] = own_pe_brand_5_s2
#pd.options.display.max_rows = None
print(df_price_elasticities)

## Cross Brand Price Elasticity for Brand 5 terhadap brand 4 by Segment 2
df_brand5_cross_brand4_s2 = pd.DataFrame(index= np.arange(price_range.size))
df_brand5_cross_brand4_s2['Price_1'] = brand_choice_s2['Price_1'].mean()
df_brand5_cross_brand4_s2['Price_2'] = brand_choice_s2['Price_2'].mean()
df_brand5_cross_brand4_s2['Price_3'] = brand_choice_s2['Price_3'].mean()
df_brand5_cross_brand4_s2['Price_4'] = price_range
df_brand5_cross_brand4_s2['Price_5'] = brand_choice_s2['Price_5'].mean()
print(df_brand5_cross_brand4_s2)

# predict probability brand choice for brand 4
predict_brand5_cross_brand4_s2 = model_brand_choice_s2.predict_proba(df_brand5_cross_brand4_s2)

# extract only probability brand choice for brand 4
pr_cross_brand_5_s2 = predict_brand5_cross_brand4_s2[:][:, 3]

# finding the elasticity of probability brand choice for brand 5 cross brand 4
print(beta5_s2) ## koeff dari price brand 5 dan y brand 5 by segment 2

brand5_cross_brand4_price_elasticity_s2 = -beta5_s2 * price_range * pr_cross_brand_5_s2
df_price_elasticities['Brand_5_Cross_Brand_4_s2'] = brand5_cross_brand4_price_elasticity_s2
#pd.options.display.max_rows = None
print(df_price_elasticities)


### membuat plot own and cross price elasticity brand 5 by segment 0, 1, 2, 3, & 4
fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 2, figsize=(11, 9), sharex= True)
ax1[0].plot(price_range, own_pe_brand_5, 'tab:grey')
ax1[0].set_title('Brand 5 Average Customer')
ax1[0].set_ylabel('Elasticity')
ax1[1].plot(price_range, brand5_cross_brand4_price_elasticity, 'tab:grey')
ax1[1].set_title('Cross Brand 4 Average Customer')

ax2[0].plot(price_range, own_pe_brand_5_s0, 'tab:blue')
ax2[0].set_title('Brand 5 Segment Standard')
ax2[0].set_ylabel('Elasticity')
ax2[1].plot(price_range, brand5_cross_brand4_price_elasticity_s0, 'tab:blue')
ax2[1].set_title('Cross Brand 4 Segment Standard')

ax3[0].plot(price_range, own_pe_brand_5_s1, 'tab:green')
ax3[0].set_title('Brand 5 Segment Career-Focused')
ax3[0].set_ylabel('Elasticity')
ax3[1].plot(price_range, brand5_cross_brand4_price_elasticity_s1, 'tab:green')
ax3[1].set_title('Cross Brand 4 Segment Career-Focused')

ax4[0].plot(price_range, own_pe_brand_5_s2, 'tab:red')
ax4[0].set_title('Brand 5 Segment Fewer-Opportunities')
ax4[0].set_ylabel('Elasticity')
ax4[1].plot(price_range, brand5_cross_brand4_price_elasticity_s2, 'tab:red')
ax4[1].set_title('Cross Brand 4 Segment Fewer-Opportunities')

ax5[0].plot(price_range, own_pe_brand_5_s3, 'tab:orange')
ax5[0].set_title('Brand 5 Segment Well-Off')
ax5[0].set_xlabel('Price 5')
ax5[0].set_ylabel('Elasticity')
ax5[1].plot(price_range, brand5_cross_brand4_price_elasticity_s3, 'tab:orange')
ax5[1].set_title('Cross Brand 4 Segment Well-Off')
ax5[1].set_xlabel('Price 4')

plt.show()

## Hasil: We can spot that the standard customer is more elastic when compared to the average. The difference becomes even more pronounced when we compared the standards to
#  the career focused and well off segment in the price range two point one to two point eight (2.1-2.8). The elasticity of the standard segment is between minus one point for two and
#  minus two point seven (-1.42- -2.7). Therefore, its purchase probability for the own brand is elastic for the entire observed price range of the brand. If we were to win some of the
#  standard segment market, our marketing strategy would be to lower prices in this price range to increase the purchase probability for this segment. However, remember that
#  this segment isn't homogenous and a marketing strategy based on only this segment might not be in our best interest. Let's see how the next segment will fare in this respect.
#  It's the career focused straight away. We can as certain that they are the least elastic among the rest. They seem to be inelastic throughout the whole price range.
#  This is great news for the marketing team as it means that this segment is not really affected by the increase in the price of the own brand. In addition, there are cross
#  price elasticity also has extremely low values. This shows that they are unlikely to switch to the competitor brand. Such segments are called loyal to the brand, and it
#  may sound ruthless, but the marketing team could increase prices of our own brand without fear of losing too much market share. All right, let's continue with the fewer
#  opportunities segment. Their elasticity curve seems to differ when compared to the rest of the segments. The own price elasticity has a more pronounced shape. This segment
#  seems to be inelastic at lower price points, and then they rapidly become the most elastic customers at higher prices. In fact, for the whole natural domain of the brand
#  five prices, they are rather inelastic in terms of cross price elasticity is in the same range. As for the career focused, looks as if they're somewhat loyal to Brand five
#  when compared to Brand 4 what could be the reason. When we consult our descriptive analysis table, we concur that this segment almost never buys brand five or indeed
#  brand 4 less than one percent of their customers have purchased one of these brands. Therefore, we don't have enough observations to obtain an accurate model.
#  And that is the reason why both curves look so out of character. We can conclude that in order for marketing to target this segment in particular Brand five, we need
#  to obtain more data of purchases from this segment. Sometimes, like here, a product may be too pricey for a segment, so we may never obtain more data about their behavior.
#  These people are simply not the target group. Based on this observation, it makes sense to actually focus on the descriptives for a bit more. It appears that the career
#  focused and well-off segments require the most attention as they are actually the people that purchase Brand five. We've already gained some insight about the well-off
#  segment in our previous lecture. In fact, it seems that the well-off segment is much more elastic than the career focused. Therefore, if we were to increase our prices,
#  this would barely affect the career focused segment, but would seriously damage our well-off segment sales. Now, what if Brand four were to decrease their price, as we
#  hypothesized in our previous lecture? Well, that would affect the well-off segment, as discussed before, but not the career focused one. Therefore, a tiny decrease in our
#  pricing would compensate such a competitive move. This is extremely important to know because if prices of chocolate candy bars were to drop, we would have space to
#  decrease our price offering while gaining solid market share from the well-off segment and practically retaining our career focused customer base. We've completed our
#  brand choice analysis for Brand five, we saw how to interpret Owen Price and cross price elasticities and use segmentation information to devise a marketing strategy.
#  Of course, you can conduct the same analysis for any of the remaining brands. For instance, it could be interesting to observe what insight can be gathered by comparing
#  brands one and two. A quick hint. It's where the standard and fewer opportunities dominate. Well, you already have the know how to tackle this task all on your own.
#  We'll continue with modeling purchase probability based on purchase quantity. This will be the next chapter in the course. See you there.
