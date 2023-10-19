import numpy as np
import pandas as pd 
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram , linkage
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import pickle

df_segmentation = pd.read_csv('D:/DATA ANALYST/belajar_python/CUSTOMER ANALYTICS/segmentation data.csv', index_col=0)
print(df_segmentation.head())
print(df_segmentation.describe())

### Correlation Estimate ###
print(df_segmentation.corr())

plt.figure(figsize=(12,9))
s = sns.heatmap(df_segmentation.corr(), annot=True, cmap='RdBu', vmin=-1, vmax=1)
s.set_yticklabels(s.get_yticklabels(), rotation=0, fontsize=12)
s.set_xticklabels(s.get_xticklabels(), rotation=90, fontsize=12)
plt.title('Correlation Heatmap')
plt.show()

### Visualize Raw Data ###
plt.figure(figsize=(12,9))
plt.scatter(df_segmentation.iloc[:,2], df_segmentation.iloc[:,4])
plt.xlabel('Age')
plt.ylabel('Income')
plt.title('Vsualization of raw data')
plt.show()                                                                              

### Standardization ###
scaler = StandardScaler()
segmentation_std = scaler.fit_transform(df_segmentation)

### Hierarchical Clustering ###
hier_clust = linkage(segmentation_std, method='ward')

plt.figure(figsize=(12,9))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Observations')
plt.ylabel('Distance')
dendrogram(hier_clust, truncate_mode='level', p = 5,show_leaf_counts=False, no_labels=True) #color_threshold=0 (untuk warna, jika 0 artinya tidak ada warna)
plt.show()
## hasil: ada 4 cluster

### K-Means Clustering ###
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i, init= 'k-means++', random_state = 42)
    kmeans.fit(segmentation_std)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(10,9))
plt.plot(range(1,11), wcss, marker='o', linestyle= '--')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.title('K-Means Clustering')
plt.show()

## hasil: cluter 4 sepertinya best solution
kmeans = KMeans(n_clusters = 4, init= 'k-means++', random_state = 42)
kmeans.fit(segmentation_std)

### Results K-Means Clustering ###
df_segm_kmeans = df_segmentation.copy()
df_segm_kmeans['Segment K-Means'] = kmeans.labels_
print(df_segm_kmeans)

## Means setiap cluster
df_segm_analysis = df_segm_kmeans.groupby(['Segment K-Means']).mean()
print(df_segm_analysis)

## Proporsi
df_segm_analysis['N Obs'] = df_segm_kmeans[['Segment K-Means', 'Sex']].groupby(['Segment K-Means']).count()
df_segm_analysis['Prop Obs'] = df_segm_analysis['N Obs'] / df_segm_analysis['N Obs'].sum()
print(df_segm_analysis)

df_segm_analysis = df_segm_analysis.rename({0:'well-off',
                         1: 'fewer-opportunities',
                         2: 'standard',
                         3: 'career focused'})

print(df_segm_analysis)

## plot hasil K-Means
df_segm_kmeans['Labels'] = df_segm_kmeans['Segment K-Means'].map({0:'well-off',
                                                                  1: 'fewer-opportunities',
                                                                  2: 'standard',
                                                                  3: 'career focused'})

print(df_segm_kmeans)

x_axis = df_segm_kmeans['Age']
y_axis = df_segm_kmeans['Income']
plt.figure(figsize=(10,8))
sns.scatterplot(x_axis, y_axis, hue= df_segm_kmeans['Labels'], palette=['g', 'r', 'c', 'm'])
plt.title('Segmentation K-Means')
plt.show()

## ---- Analisis mean per cluster  ---- ##
# We start with the first segment. It is composed of men and women almost equally, with an average age of 56, comparing this mean age with the other clusters.
# We realize that this is the oldest segment. More than two thirds are in relationships, and they also have the highest level of education as well as the highest
#  income we could call this segment well-off people

# What about the second segment? Two thirds are male and almost all are single, their average age is 36 years, their education level is low on average compared
#  to other segments in terms of salary and jobs.
# This segment has the lowest values under a hundred thousand for annual salary. Also, they live almost exclusively in small cities, so these are people in their
#  30s with a relatively low income living in small cities, it seems that this is a segment of people with fewer opportunities.

# Let's carry on with the third segment. These are people in relationships with an average age of 29. This is the youngest segment. They have a medium level of education,
#  average income and middle management jobs. They seem equally distributed between small, mid-sized and big cities, so they seem average in just about every parameter
#  we can label the segment average or standard.

# Finally, we come to the fourth segment, it is comprised almost entirely of men, less than 20 percent of whom are in relationships. Looking at the numbers, we observe
#  relatively low values for education, paired with high values for income and occupation. The majority of this segment lives in big or middle sized cities.
#  It appears people in this segment are career focused.

## ---- Hasil Proporsi sampel per segment ---- ##
# We can see that there are 263 individuals or 13 percent of the entire data in the well off segment, so this is the smallest segment. The largest segment is the third one
#  standard comprised of 35 percent of all individuals in between. We find the career focused on fewer opportunities containing 570 and 462 persons, respectively.

## ---- Kesimpulan ----- ##
# We can see the green segment well off is clearly separated as it is highest in both age and income. Unfortunately, the other three are grouped together, so it's harder
#  to get more insight just by looking at the plot.

## We can conclude that kmeans did a decent job at separating our data into clusters, however, the result is far from perfect, so we're interested to see how
#  we can get even more out of it. So we'll combine k means with principal component analysis and try to get a better result.


### Principal Components Analysis ###
pca = PCA()
pca.fit(segmentation_std)

print(pca.explained_variance_ratio_)

plt.figure(figsize=(12,9))
plt.plot(range(1,8), pca.explained_variance_ratio_.cumsum(), marker='o', linestyle='--')
plt.title('Explained Variance by Components')
plt.xlabel('Number of Components')
plt.ylabel('Cuulative Explained Variance')
plt.show()

## hasil: pilih jumlah variabel yang menyimpan informasi sekitar 70-80% dari keseluruhan
pca = PCA(n_components=3)
pca.fit(segmentation_std)

### PCA Results ###
print(pca.components_)

df_pca_comp = pd.DataFrame(data= pca.components_, columns= df_segmentation.columns.values, index= ['Component 1', 'Component 2', 'Component 3'])
print(df_pca_comp)

sns.heatmap(df_pca_comp, vmin=-1, vmax=1, cmap='RdBu', annot=True)
plt.yticks([0,1,2], ['Component 1', 'Component 2', 'Component 3'], rotation=45, fontsize=9)
plt.show()

## Hasil: There is a positive correlation between component one and age, income, occupation and settlement size, as you can guess, these are relate strictly to the career
#  of a person. So this component shows the career focus of the individual.

## And how about the second component? It seems to be quite different. Sex, marital status and education are by far the most prominent determinants.
# You can also see that all career related features are almost uncorrelated with it. Therefore, this component doesn't refer to the career,
#  but rather to an individual's education and lifestyle

## regarding the final component, We realize that age, marital status and occupation are the most important determinants here. We observed that marital status and occupation
#  load negatively but are still important. That's because we determine the importance of the individual loadings with respect to their absolute values.
# So the three important aspects of the third component indicate the experienced person has, no matter if work experience or life experience.

print(pca.transform(segmentation_std))
scores_pca = pca.transform(segmentation_std)

### K-Means Clustering With PCA ###

wcss = []
for i in range(1,11):
    kmeans_pca = KMeans(n_clusters = i, init= 'k-means++', random_state = 42)
    kmeans_pca.fit(scores_pca)
    wcss.append(kmeans_pca.inertia_)

plt.figure(figsize=(10,9))
plt.plot(range(1,11), wcss, marker='o', linestyle= '--')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.title('K-Means  With PCA Clustering')
plt.show()

kmeans_pca = KMeans(n_clusters=4, init='k-means++', random_state=42 )
kmeans_pca.fit(scores_pca)

### K-Means Clustering With PCA Results ###

df_segm_pca_kmeans = pd.concat([df_segmentation.reset_index(drop=True), pd.DataFrame(scores_pca)], axis=1)
df_segm_pca_kmeans.columns.values[-3:] = ['Component 1', 'Component 2', 'Component 3']
df_segm_pca_kmeans['Segment K-Means PCA'] = kmeans_pca.labels_
print(df_segm_pca_kmeans)

### Analisis Mean
df_segm_pca_kmeans_freq = df_segm_pca_kmeans.groupby(['Segment K-Means PCA']).mean()
print(df_segm_pca_kmeans_freq)

### Unfortunately, they don't say much. Still, so far we provided some qualitative interpretation. Maybe it will help us interpret the segments found by key means PCA
#  to some extent.
## We said the first component was related to career, the second to education and lifestyle, while the third to their life and or work experience.
# We can see that the first segment is low on the first component and high on the second component. Second shows high values for career but low for education and lifestyle.
# The third segment is low on both, and the fourth segment is the highest on both.
# As you can probably recall, our previous four clusters were standard career focused, fewer opportunities and well off.
# Let's see if we can spot the same clusters this time.
## Segment three is highest on all three components career, education and lifestyle and experience. We can easily establish that this is the well-off segment.
# OK, segment two, on the other hand, shows the lowest average PC scores for a career in education and lifestyle, but is high on experience. It seems that this is the
#  fewer opportunities cluster.
# Great segment, one shows high values for career, but low for education and lifestyle and is somewhat independent from experience. This sounds like a career focused segment.
# Finally, we are left with the zero segment, it has low career and experience values while normal to high education and lifestyle values.
# This is our youngest cluster, which we labeled standard in our previous analysis.


## Proporsi
df_segm_pca_kmeans_freq['N Obs'] = df_segm_pca_kmeans[['Segment K-Means PCA', 'Sex']].groupby(['Segment K-Means PCA']).count()
df_segm_pca_kmeans_freq['Prop Obs'] = df_segm_pca_kmeans_freq['N Obs'] / df_segm_pca_kmeans_freq['N Obs'].sum()

df_segm_pca_kmeans_freq = df_segm_pca_kmeans_freq.rename({0:'Standard',
                         1: 'Career Focused',
                         2: 'Fewer Opportunities',
                         3: 'Well-off'})

print(df_segm_pca_kmeans_freq)

# The largest segment with 692 individuals is the standard segment, it's followed by the career focus group with 583 people, or 29 percent of the whole.
# Next comes the Fewer Opportunities Group with 460 people, followed by the smallest segment well off. It contains only 265 or 13 percent of the total number of individuals.

df_segm_pca_kmeans['Legend'] = df_segm_pca_kmeans['Segment K-Means PCA'].map({0:'Standard', 1:'Career Focused', 2:'Fewer Opportunities', 3:'Well-off'})

x_axis = df_segm_pca_kmeans['Component 2']
y_axis = df_segm_pca_kmeans['Component 1']
plt.figure(figsize=(10,8))
sns.scatterplot(x_axis, y_axis, hue=df_segm_pca_kmeans['Legend'], palette=['g', 'r', 'c', 'm'])
plt.title('Clusters by PCA Components')
plt.show()

### Data Export ###
pickle.dump(scaler, open('scaler_purchase.pickle', 'wb'))
pickle.dump(pca, open('pca_purchase.pickle', 'wb'))
pickle.dump(kmeans_pca, open('kmeans_pca_purchase.pickle', 'wb'))
