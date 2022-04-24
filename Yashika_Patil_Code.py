#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[2]:


# Loading the data
df = pd.read_csv('dataset.csv')
df


# # 1. Exploratory Data Analysis (EDA)

# In[3]:


df = df.drop(['Latitude', 'Longitude'], axis = 1) #redundant feature, already used in location


# In[4]:


# converting dates to pandas datetime format
df.Date = pd.to_datetime(df.Date, format='%m/%d/%Y %I:%M:%S %p')

# setting the index to be the date
df.index = pd.DatetimeIndex(df.Date)
df


# In[5]:


# Plot: Heat Map

correlation = df.corr()
sns.set(rc = {'figure.figsize':(15,8)})
sns.heatmap(correlation, xticklabels = correlation.columns, yticklabels = correlation.columns, annot = True)


# In[6]:


# Plot: Number of crimes per month

plt.figure(figsize=(11,5))
df.resample('M').size().plot(legend=False)
plt.title('Number of crimes per month')
plt.xlabel('Months')
plt.ylabel('Number of crimes')
plt.show()


# In[7]:


# Plot: Number of crimes by type

plt.figure(figsize=(8,10))
df.groupby([df['Primary Type']]).size().sort_values(ascending=True).plot(kind='barh')
plt.title('Number of crimes by type')
plt.ylabel('Crime Type')
plt.xlabel('Number of crimes')
plt.show()


# In[8]:


# Plot: Primary Types Vs. Year

df_count_date = df.pivot_table('ID', aggfunc=np.size, columns='Primary Type', index=df.index.date, fill_value=0)
df_count_date.index = pd.DatetimeIndex(df_count_date.index)
plot = df_count_date.rolling(365).sum().plot(figsize=(30, 30), subplots=True, layout=(-1, 3), sharex=False, sharey=False)


# In[9]:


# Plot: Crimes Occuring by the days

days = ['Monday','Tuesday','Wednesday',  'Thursday', 'Friday', 'Saturday', 'Sunday']
df.groupby([df.index.dayofweek]).size().plot(kind='barh')
plt.ylabel('Days of the week')
plt.yticks(np.arange(7), days)
plt.xlabel('Number of crimes')
plt.title('Number of crimes by days')
plt.show()


# In[10]:


# Plot: Crimes Occuring by Months

df.groupby([df.index.month]).size().plot(kind='barh')
plt.ylabel('Months of the year')
plt.xlabel('Number of crimes')
plt.title('Number of crimes by months')
plt.show()


# In[11]:


# Plot: Crimes Occuring by Location

plt.figure(figsize=(10, 30))
df.groupby([df['Location Description']]).size().sort_values(ascending=True).plot(kind='barh')
plt.title('Number of crimes by Location')
plt.ylabel('Crime Location')
plt.xlabel('Number of crimes')
plt.show()


# # 2. Preprocessing Data

# In[12]:


# resetting the index

df.reset_index(drop = True)


# In[13]:


df.info()


# In[14]:


df = df.astype({"Unnamed: 0": object})
df = df.astype({"ID": object})
df = df.astype({"Beat": object})
df = df.astype({"District": object})
df = df.astype({"Ward": object})
df = df.astype({"Community Area": object})
df = df.astype({"Year": object})
df = df.astype({"X Coordinate": object})
df = df.astype({"Y Coordinate": object})

df.info()


# In[15]:


df.isnull().sum()


# In[16]:


df.nunique()


# # 3.1. Feature Selection

# ## 3.1.1 Entropy

# In[17]:


def entropy(y):
    probs = [] # Probabilities of each class label
    for c in set(y): # Set gets a unique set of values. We're iterating over each value
        num_same_class = sum(y == c)  # Remember that true == 1, so we can sum.
        p = num_same_class / len(y) # Probability of this class label
        probs.append(p)
    return sum(-p * np.log2(p) for p in probs)


# In[18]:


columns = ['Unnamed: 0', 'ID', 'Case Number', 'Date', 'Block', 'IUCR',
       'Primary Type', 'Description', 'Location Description', 'Arrest',
       'Domestic', 'Beat', 'District', 'Ward', 'Community Area', 'FBI Code',
       'X Coordinate', 'Y Coordinate', 'Year', 'Updated On', 'Location']

"""This code takes too long to execute. So, the result has been appended as a column "Entropy" in df_ent

#Code:
for column in columns:
    print(entropy(df[column]))
    df_ent.append(entropy(df[column]))

"""


# In[19]:


df_ent = pd.DataFrame()
df_ent["Features"] = columns
df_ent["Entropy"] = [16.60964047441802, 16.60964047441802, 16.60964047441802, 16.50325437354276, 
                     13.842338152132987, 5.6494058088293935, 3.482725829513007, 5.518046958190912,4.0556159613837135,
                     0.839130421235637,0.5773624295265078,8.0915039193288,4.403738721836829,5.475453895970417,
                     5.826020205215316,3.5574293236120447,14.760131667723728,15.11747552923421,4.239196753831807,
                     2.8553923756999566,15.74133241017893]

df_ent


# In[20]:


#Dropping high entropy features(Entropy > 10)

df_drop = df.drop(columns =['Unnamed: 0', 'ID', 'Case Number', 'Date', 
                        'Block', 'X Coordinate', 'Y Coordinate', 'Location'])

df_drop.reset_index(drop = True)


# # Part 1 - 100,000 Rows

# ## 4. Unsupervised Learning

# ### 4.1. K Modes Clustering

# In[21]:


from kmodes.kmodes import KModes


# In[22]:


#Elbow Plot

""""
cost =[]
K = range(0,31)
for num_clusters in list(K):
    kmode = KModes(n_clusters = num_clusters, init = "random", n_init = 2, verbose = 1)
    kmode.fit_predict(df1)
    cost.append(kmode.cost_)
    
plt.plot(K, cost, 'bx-')
plt.xlabel('No. of clusters')
plt.ylabel('Cost')
plt.title('Elbow Method For Optimal k')
plt.show()

"""


# In[23]:


kmodes = KModes(n_clusters = 31, init = "random", n_init = 5, verbose = 1) #given primary type as the class label which 31
kmodes_clusters = kmodes.fit_predict(df_drop)

kmodes_clusters


# In[24]:


set(kmodes_clusters)


# In[25]:


print(kmodes_clusters)


# In[26]:


data_kmodes= pd.read_csv('dataset.csv')
data_kmodes.insert(0, "KMode_Cluster", kmodes_clusters, True)

data_kmodes


# ### 4.2. Hierarchial Clustering

# In[27]:


from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from scipy.cluster import  hierarchy

#Vectorizing
X = CountVectorizer().fit_transform(df_drop)
X = TfidfTransformer().fit_transform(X)

#Clustering
X = X.todense()
threshold = 0.1
Z = hierarchy.linkage(X,"average", metric="cosine")
hierarchial_clusters = hierarchy.fcluster(Z, threshold, criterion="distance")

hierarchial_clusters


# In[28]:


set(hierarchial_clusters)


# In[29]:


print(hierarchial_clusters)


# In[30]:


#data_hierarchial= pd.read_csv('dataset.csv')
#data_hierarchial.insert(0, "Hierarchial_Cluster", hierarchial_clusters, True)

#data_hierarchial


# # Part 2 - 25,000 Rows

# We are considering 25k data points due to technical limitations

# ## 3.2. Feature Engineering

# In[31]:


#These data points have unique primary crime type, later we will need to drop it for splitting

df_drop.drop([df_drop.index[64744], df_drop.index[94033]], axis=0, inplace=True)


# ### 3.2.1. Splitting Data

# In[32]:


# splitting data into 25k

from sklearn.model_selection import train_test_split

X = df_drop[['IUCR','Description','Location Description','Arrest','Domestic','Beat','District',
          'Ward','Community Area','FBI Code','Year','Updated On']]
Y = df_drop['Primary Type']

data25k, data75k, test25k, test75k= train_test_split(X, Y, test_size = 0.75, random_state = 50)

data25 = pd.concat([data25k, test25k], axis=1)
data25 = data25.reset_index(drop= True)

data25.to_csv("data25.csv")


# ### 3.2.2. One Hot Encoding

# In[33]:


column_names = []
for row in df_drop:
  column_names.append(row)

column_names


# In[34]:


# generate binary values using get_dummies

df_dum = pd.get_dummies(data25, columns = column_names)
df_dum = df_dum.reset_index(drop = True)


# ## 4. Unsupervised Learning

# ### 4.3. Spectral Clustering

# In[35]:


# finding cosine similarity maxtrix to be passed as a argument to spectral clustering algorithm

from sklearn.metrics.pairwise import cosine_similarity
cosine = cosine_similarity(df_dum, df_dum)

#Takes too long to execute
#print(cosine)


# In[36]:


from sklearn.cluster import SpectralClustering

# Building the clustering model
spectral_model = SpectralClustering(eigen_solver=None, n_components=None, 
                                          random_state=None, n_init=10, gamma=1.0, affinity='precomputed', 
                                          n_neighbors=10, eigen_tol=0.0, assign_labels='kmeans', degree=3, 
                                          coef0=1, kernel_params=None, n_jobs=3, verbose=False)
  
# Training the model and Storing the predicted cluster labels
spectral_clusters = spectral_model.fit_predict(cosine)


# In[37]:


set(spectral_clusters)


# In[38]:


print(spectral_clusters)


# In[39]:


data_spectral= pd.read_csv('data25.csv')
data_spectral.insert(0, "Spectral_Cluster", spectral_clusters, True)

data_spectral


# # 5. Visualising Results

# ## 5.1. K Modes

# ### 5.1.1. NMI

# In[40]:


from sklearn.metrics.cluster import normalized_mutual_info_score

NMI_KModes= pd.DataFrame()

Feature = []
NMI_Score = []

for feature in data25.columns:
    score = normalized_mutual_info_score(data_kmodes[feature], kmodes_clusters)
    Feature.append(feature)
    NMI_Score.append(score)

NMI_KModes["Feature"] = Feature
NMI_KModes["NMI_Score"] = NMI_Score

NMI_KModes


# ### 5.1.2. Plot

# In[41]:


# Plot btw KModes Clusters VS Relevant Features

high_NMI = ['IUCR', 'Description','FBI Code','Primary Type' ]
for cluster in set(kmodes_clusters):
    print(f'Plot for Cluster:', cluster)
    cls = data_kmodes[cluster == kmodes_clusters]
    for feature in high_NMI:
        cls[feature].value_counts().plot(kind='pie')
        plt.show()


# ## 5.2. Spectral Clustering

# ### 5.2.1. NMI

# In[42]:


from sklearn.metrics.cluster import normalized_mutual_info_score

NMI_Spectral= pd.DataFrame()

Feature = []
NMI_Score = []

for feature in data25.columns:
    score = normalized_mutual_info_score(data25[feature], spectral_clusters)
    Feature.append(feature)
    NMI_Score.append(score)

NMI_Spectral["Feature"] = Feature
NMI_Spectral["NMI_Score"] = NMI_Score

NMI_Spectral


# ### 5.2.2. Plot 

# In[43]:


# Plot btw Spectral Clusters VS Relevant Features

high_NMI = ['IUCR', 'Description','FBI Code','Primary Type' ]
for cluster in set(spectral_clusters):
    print(f'Plot for Cluster:', cluster)
    cls = data25[cluster == spectral_clusters]
    for feature in high_NMI:
        cls[feature].value_counts().plot(kind='pie')
        plt.show()

