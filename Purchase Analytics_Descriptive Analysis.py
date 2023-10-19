import numpy as np
import pandas as pd 
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import pickle
import matplotlib.pyplot as plt 
import seaborn as sns 
sns.set()

### Data Import ###
df_purchase = pd.read_csv('D:/DATA ANALYST/belajar_python/CUSTOMER ANALYTICS/purchase data.csv')

### Data Exploration ###
print(df_purchase.head())
print(df_purchase.describe())

## the descriptive statistics here wouldn't be very useful.
# The reason for that is that for each customer, we have a road denoting each time they visited the store and each customer may have visited the store a different
# number of times throughout the two years that data was collected.
# All of these imply we don't have an equal number of records per customer and we don't have an equal number of records per day. So descriptive statistics would
# neither be useful nor appropriate.

print(df_purchase.isnull().sum())

### Import Segmentation Model ###
scaler = pickle.load(open('scaler_purchase.pickle', 'rb'))
pca = pickle.load(open('pca_purchase.pickle', 'rb'))
kmeans_pca = pickle.load(open('kmeans_pca_purchase.pickle', 'rb'))

### Standardization ###
features = df_purchase[['Sex', 'Marital status', 'Age', 'Education', 'Income', 'Occupation', 'Settlement size']]
print(features)

df_purchase_segm_std = scaler.transform(features)
print(df_purchase_segm_std)

### PCA ###
df_purchase_segm_pca = pca.transform(df_purchase_segm_std)
## df_purchase_segm_pca will contain duplicates beacuse 1 person can do more than one transaction

### K-Means PCA ###
purchase_segm_kmeans_pca = kmeans_pca.predict(df_purchase_segm_pca)

df_purchase_predictors = df_purchase.copy()
df_purchase_predictors['Segment'] = purchase_segm_kmeans_pca
print(df_purchase_predictors)

### ----- DESCRIPTIVE ANALYSIS BY SEGMENTS ----- ###
### -- Data Analysis by Customer -- ###
temp1 = df_purchase_predictors[['ID', 'Incidence']].groupby(['ID'], as_index=False).count()
temp1 = temp1.set_index('ID')
temp1 = temp1.rename(columns = {'Incidence': 'N_Visits'})
print(temp1.head())

temp2 = df_purchase_predictors[['ID', 'Incidence']].groupby(['ID'], as_index=False).sum()
temp2 = temp2.set_index('ID')
temp2 = temp2.rename(columns = {'Incidence': 'N_Purchases'})

temp3 = temp1.join(temp2)
print(temp3.head())

temp3['Average_N_Purchases'] = temp3['N_Purchases'] / temp3['N_Visits']
print(temp3.head())

temp4 = df_purchase_predictors[['ID', 'Segment']].groupby(['ID'], as_index=False).mean()
temp4 = temp4.set_index('ID')

df_purchase_descr = temp3.join(temp4)
print(df_purchase_descr)

## hasil: dengan tabel diatas kita dapat menganalisis purchase behavior per konsumen
# tapi karena ada 500 konsumen, maka akan sulit dianalisis
## so instead, what we can do is analyze the behavior of the four segments

### -- Data Analysis by Segments -- ###
### Segment Proportions ###
segm_prop = df_purchase_descr[['N_Purchases', 'Segment']].groupby(['Segment']).count() / df_purchase_descr.shape[0] 
segm_prop = segm_prop.rename(columns= {'N_Purchases': 'Segment Proportions'})
print(segm_prop.head())

plt.figure(figsize=(9,6))
plt.pie(segm_prop['Segment Proportions'], 
        labels=['Standard', 'Career-Focused', 'Fewer-Opportunities', 'Well-Off'],
        autopct='%1.1f%%',
        colors=('b', 'g', 'r', 'orange'))
plt.title('Segment Proportions')
plt.show()

## hasil pie chart: the Fewer Opportunities segment is the largest, with thirty seven point nine percent, followed by the career focused one with 22 percent,
#  the segments of the well off people on the standard are almost equally distributed with around 20 percent each.

### Purchase Occasion and Purchase Incidence ###
segments_mean = df_purchase_descr.groupby(['Segment']).mean()
print(segments_mean)

segments_std = df_purchase_descr.groupby(['Segment']).std()
print(segments_std)

## How often to people from different segments visit the store? (Purchase Occasion)
plt.figure(figsize=(9,6))
plt.bar(x=(0,1,2,3), 
        tick_label= ('Standard', 'Career-Focused', 'Fewer-Opportunities', 'Well-Off'),
        height= segments_mean['N_Visits'],
        yerr= segments_std['N_Visits'],
        color= ('b', 'g', 'r', 'orange'))
plt.xlabel('Segment')
plt.ylabel('Number of Store Visits')
plt.title('Average Number of Store Visits by Segment')
plt.show()

## hasil: The height of each bar represents the means store visits, the vertical line, on the other hand, indicates the dispersion of the data points or how big
#  the standard deviation is. With that in mind, we can see that the fewer opportunities segment visits the store, least often while the career focused visits at most.
# However, the standard deviation amongst customers from the second segment is quite high (career focused). This implies that the customers in this segment are at
#  least homogenous. That is least alike when it comes to how often they visit the grocery store.
# All nit picking this aside, it seems to me that the standard, fewer opportunities, and well-off clusters are very similar in terms of their average store visit.
# This is welcome information because it would make them more comparable with respect to our future analysis.

## How often the client buys a product? (Purchase Incidence)
plt.figure(figsize=(9,6))
plt.bar(x=(0,1,2,3), 
        tick_label= ('Standard', 'Career-Focused', 'Fewer-Opportunities', 'Well-Off'),
        height= segments_mean['N_Purchases'],
        yerr= segments_std['N_Purchases'],
        color= ('b', 'g', 'r', 'orange'))
plt.xlabel('Segment')
plt.ylabel('Purchase Incidences')
plt.title('Average Number of Purchases by Segment')
plt.show()

## hasil: We observe that the career focused segment buys products more often, however, once again we see that it's standard deviation is the highest.
# It might be that a part of the segment buys products very frequently and another part less so. Although consumers in the segment have a somewhat similar income, the way that
#  they might want to spend their money might differ. The most homogenous segment appears to be that of the fewer opportunities. This is signified by the segment having the
#  lowest standard deviation or shortest vertical line. The first segment (standard) seems consistent as well, with about 25 average purchases and a standard deviation of 30.

### Brand Choice ###
df_purchase_incidence = df_purchase_predictors[df_purchase_predictors['Incidence'] == 1]
print(df_purchase_incidence)

brand_dummies = pd.get_dummies(df_purchase_incidence['Brand'], prefix= 'Brand', prefix_sep= '_')
brand_dummies['Segment'], brand_dummies['ID'] = df_purchase_incidence['Segment'], df_purchase_incidence['ID']
print(brand_dummies)

## Average brand choice per customer
temp = brand_dummies.groupby(['ID'], as_index= True).mean()
print(temp)

## Average brand choice by segment
mean_brand_choice = temp.groupby(['Segment'], as_index= True).mean()
print(mean_brand_choice)

sns.heatmap(mean_brand_choice,
            vmin=0,
            vmax=1,
            cmap='PuBu',
            annot=True)
plt.yticks([0,1,2,3], ['Standard', 'Career-Focused', 'Fewer Opportunities', 'Well-Off'], rotation=45, fontsize=9)
plt.title('Average Brand Choice by Segment')
plt.show()

### Our minimum value here is zero, while our maximum value is one since we are representing a proportion.

## hasil: Each of the numbers here shows the average proportion of brand choices for each segment, a very important note is that the five brands are arranged in ascending order
#  of price. In other words, Brand one is the cheapest brand, while brand five is the most expensive one. This is how the data set itself is ordered. However, it's crucial to
#  take into account when conducting brand analysis.
## OK, let's start with the Fewer Opportunities segment. It shows an extremely strong preference for brand to almost 70 percent of the segment chooses this brand of chocolate (brand 2).
# It certainly isn't the cheapest one. So we can say the price of chocolate bars isn't what matters most to our customers.
# Surely we can't help but notice that 63 percent of the career focused segment buys Brand five, which is the most expensive brand, it seems that this cluster of young, ambitious,
#  career focused individuals enjoys this fancy candy bar with no additional information. We can speculate that the career focused segment is looking for some kind of luxury status, and this
# alone may be an opportunity to raise the price of Brand five even further. Interestingly enough, the well-off segment enjoys one of the most luxurious brands, but not the most
# expensive one, brand four is by far the most widely bought brand, followed by Brand five.
## Finally, let's go back to the top of the heat map. This is definitely the most heterogeneous segment. We observed that people from the standard segment, as we call them,
#  have a preference for brand, two, and a weaker preference for brands one and three. It's more than obvious they don't like buying brand four, nevertheless, their preference
#  is scattered all around. Bearing that in mind, if we're looking for actionable insight, one idea is to try to influence them to try out different brands.
## Overall, this is one of the most high level brand choice summaries we can get, but even though it's Eye-Opening, it doesn't provide any information about how these
#  preferences affect our bottom line. So in the next lecture, we will explore the revenue by segment.

### Revenue ###
## Revenue per brand and per segment
temp = df_purchase_predictors[df_purchase_predictors['Brand'] == 1]
print(temp)
temp.loc[:, 'Revenue Brand 1'] = temp['Price_1'] * temp['Quantity']
print(temp)

segments_brand_revenue = pd.DataFrame()
segments_brand_revenue[['Segment', 'Revenue Brand 1']] = temp[['Segment', 'Revenue Brand 1']].groupby(['Segment'], as_index = False).sum()
print(segments_brand_revenue)


temp = df_purchase_predictors[df_purchase_predictors['Brand'] == 2]
temp.loc[:, 'Revenue Brand 2'] = temp['Price_2'] * temp['Quantity']
segments_brand_revenue[['Segment', 'Revenue Brand 2']] = temp[['Segment', 'Revenue Brand 2']].groupby(['Segment'], as_index = False).sum()

temp = df_purchase_predictors[df_purchase_predictors['Brand'] == 3]
temp.loc[:, 'Revenue Brand 3'] = temp['Price_3'] * temp['Quantity']
segments_brand_revenue[['Segment', 'Revenue Brand 3']] = temp[['Segment', 'Revenue Brand 3']].groupby(['Segment'], as_index = False).sum()

temp = df_purchase_predictors[df_purchase_predictors['Brand'] == 4]
temp.loc[:, 'Revenue Brand 4'] = temp['Price_4'] * temp['Quantity']
segments_brand_revenue[['Segment', 'Revenue Brand 4']] = temp[['Segment', 'Revenue Brand 4']].groupby(['Segment'], as_index = False).sum()

temp = df_purchase_predictors[df_purchase_predictors['Brand'] == 5]
temp.loc[:, 'Revenue Brand 5'] = temp['Price_5'] * temp['Quantity']
segments_brand_revenue[['Segment', 'Revenue Brand 5']] = temp[['Segment', 'Revenue Brand 5']].groupby(['Segment'], as_index = False).sum()

segments_brand_revenue['Total Revenue'] = (segments_brand_revenue['Revenue Brand 1'] +
                                           segments_brand_revenue['Revenue Brand 2'] +
                                           segments_brand_revenue['Revenue Brand 3'] +
                                           segments_brand_revenue['Revenue Brand 4'] +
                                           segments_brand_revenue['Revenue Brand 5'] )

segments_brand_revenue['Segment Proportions'] = segm_prop['Segment Proportions']
segments_brand_revenue['Segment'] = segments_brand_revenue['Segment'].map({0:'Standard',
                                                                           1:'Career-Focused',
                                                                           2:'Fewer-Opportunities',
                                                                           3:'Well-Off'})
segments_brand_revenue = segments_brand_revenue.set_index(['Segment'])
print(segments_brand_revenue)

## hasil: career focused brings the most revenue, followed by fewer opportunities and well off while the standard segment accounts for the smallest part. Now we've also got the
# segment sizes next to the revenue. Apparently, even though segment two Career Focused is the second largest, it brings the highest revenue, this is an extremely interesting 
# finding. The only hint about it we had so far was that career focused was buying the most expensive brand. However, it seems that they are by far the most prominent segment
# for the store with regard to chocolate candy bars. What about the other segments? The standard is almost equal in size with career focused, but it brings less than half that
# revenue. In fact, this segment contributes the least revenue of all segments. In comparison, the well-off and the fewer opportunity segments spend around the same amount of
# money on chocolate candy bars with the note that the fewer opportunities is twice the size of the well off. Once again, this shows us that we should examine different measures
# together rather than independently.


## Now, let's change the point of view and explore the revenue table from a brand's perspective. What if we were marketers for one of these brands and we're given that table will
#  probably be interested in knowing that. Brand three has the lowest revenue compared to the other four. It is the middle brand in terms of price. Its highest contributor is
#  the standard segment. As we've established, they'd like the first three brands, so they can be influenced to buy more of the third brand, maybe if Brand three reduces its
#  price, it is likely that the standard segment would pivot towards it. Of course, nothing is certain, but it might be worth testing.
# And what about Brand 4? For its customers seem to come almost exclusively from the well-off segment. The customers in this segment who didn't choose this brand bought an
#  even more expensive alternative, the fifth one. Therefore, they seem to be loyal and not really affected by price, therefore brand for Catterick cautiously increasing its
#  price. The hypothesis here is they'd likely retain most of its customers and increase the revenue per sale.

## We've gained some insight into our customers, but for the time being, these are only assumptions based on what we've observed in our dataset. In other words, we're conducting
#  exploratory analysis on our data. This is going to change in the next section where we'll start with predictive analytics, will create machine learning models and use them
#  to estimate price elasticities and predict their effect on revenue.
