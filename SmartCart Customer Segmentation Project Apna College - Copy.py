#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler


# In[2]:


df=pd.read_csv('smartcart_customers.csv')


# In[3]:


df.head()


# In[4]:


df.shape


# In[5]:


df.isnull().sum()


# # Data Preprocessing

# # Handle missing value

# In[6]:


df['Income']=df['Income'].fillna(df['Income'].median())


# In[7]:


df.head()


# In[8]:


df.isnull().sum()


# # Featuring Engineering

# In[9]:


df.columns


# In[10]:


# age

df['Age']=2026-df['Year_Birth']


# In[11]:


df.head()


# In[12]:


# customer joining date

df['Dt_Customer']=pd.to_datetime(df['Dt_Customer'],dayfirst=True)

reference_date=df['Dt_Customer'].max()

df['Customer_Tenure_Days']=(reference_date-df['Dt_Customer']).dt.days


# In[13]:


df.head()


# In[14]:


df.columns


# In[15]:


# spending

df['Total_Spending'] = df['MntWines'] + df['MntFruits'] + df['MntMeatProducts'] + df['MntFishProducts'] + df['MntSweetProducts'] + df['MntGoldProds']


# In[16]:


df.head()


# In[17]:


df['Total_Children']=df['Kidhome']+df['Teenhome']


# In[18]:


# Education

df['Education'].value_counts()

# ug pg graduate

df['Education']=df['Education'].replace({
    'Basic':'Undergraduate','2n Cycle':'Undergraduate',
    'Graduation':'Graduate',
    'Master':'Postgraduate','PhD':'Postgraduate'
})


# In[19]:


df['Education'].value_counts()


# In[20]:


# marital status

df['Marital_Status']


# In[21]:


df['Living_With']=df['Marital_Status'].replace({
    'Married':'Partner','Together':'Partner',
    'Single':'Alone',
    'Divorced':'Alone','Widow':'Alone',
    'Absurd':'Alone','YOLO':'Alone'
})


# In[22]:


df['Living_With'].value_counts()


# # Drop columns

# In[23]:


df.columns


# In[24]:


cols=['ID','Year_Birth','Marital_Status','Kidhome','Teenhome','Dt_Customer']
spending_cols=['MntWines','MntFruits','MntMeatProducts','MntFishProducts','MntSweetProducts','MntGoldProds']

cols_to_drop=cols+spending_cols

df_cleaned=df.drop(columns=cols_to_drop)


# In[25]:


df_cleaned.shape


# In[26]:


df.shape


# In[27]:


df_cleaned.shape


# In[28]:


df.head()


# # Outliers

# In[29]:


cols=['Income','Recency','Response','Age','Total_Spending','Total_Children']

#pairplot

sns.pairplot(df_cleaned[cols])


# In[30]:


#remove outlier

print('data size with outliers:',len(df_cleaned))

df_cleaned=df_cleaned[ (df_cleaned['Age'] < 90) ]
df_cleaned=df_cleaned[ (df_cleaned['Income'] < 600_000) ]

print('data size without outliers:',len(df_cleaned))


# In[31]:


corr=df_cleaned.corr(numeric_only=True)


# In[32]:


plt.figure(figsize=(8,6))

sns.heatmap(
    corr,annot=True,
    cmap='coolwarm',
    annot_kws={'size':6}
)


# In[33]:


df_cleaned.shape


# In[34]:


df_cleaned.head()


# # Encoding

# In[35]:


# feature encoding

from sklearn.preprocessing import OneHotEncoder


# In[36]:


one=OneHotEncoder()

cat_cols=['Education','Living_With']

enc_cols=one.fit_transform(df_cleaned[cat_cols])


# In[37]:


enc_df=pd.DataFrame(enc_cols.toarray(),columns=one.get_feature_names_out(cat_cols),index=df_cleaned.index)


# In[38]:


enc_df.head()


# In[39]:


df_encoded=pd.concat([df_cleaned.drop(columns=cat_cols),enc_df],axis=1)


# In[40]:


df_encoded.head()


# # Scaling

# In[41]:


from sklearn.preprocessing import StandardScaler


# In[42]:


selected_cols = [
    'Income','Recency','Age','Customer_Tenure_Days',
    'Total_Spending','Total_Children',
    'Education_Graduate','Education_Postgraduate','Education_Undergraduate',
    'Living_With_Alone','Living_With_Partner'
]

x = df_encoded[selected_cols]


# In[43]:


scaler=StandardScaler()
x_scaled=scaler.fit_transform(x)


# In[44]:


x_scaled


# # Visualize

# In[45]:


x_scaled.shape


# In[46]:


# 2d
from sklearn.decomposition import PCA


# In[47]:


pca=PCA(n_components=3)
x_pca=pca.fit_transform(x_scaled)


# In[48]:


pca.explained_variance_ratio_


# In[49]:


fig=plt.figure(figsize=(8,6))


ax=fig.add_subplot(111,projection='3d')

ax.scatter(x_pca[:,0],x_pca[:,1],x_pca[:,2])
ax.set_xlabel('pca1')
ax.set_ylabel('pca2')
ax.set_zlabel('pca3')
ax.set_title('3d Projection')


# In[50]:


pca.explained_variance_ratio_


# # Analyze K Value

# In[51]:


from sklearn.cluster import KMeans
from kneed import KneeLocator


# In[52]:


wcss=[]
for k in range(1,11):
    kmeans=KMeans(n_clusters=k,random_state=42)
    labels=kmeans.fit_predict(x_pca)
    wcss.append(kmeans.inertia_)


# In[53]:


knee=KneeLocator(range(1,11),wcss,curve='convex',direction='decreasing')


# In[54]:


knee.elbow


# In[55]:


# plot elbow

plt.plot(range(1,11),wcss,marker='o')
plt.xlabel('k')
plt.ylabel('wcss')


# # silhouette score

# In[56]:


from sklearn.metrics import silhouette_score


# In[57]:


scores=[]

for k in range(2,11):
    kmeans=KMeans(n_clusters=k,random_state=42)
    labels=kmeans.fit_predict(x_pca)
    score=silhouette_score(x_pca,labels)
    scores.append(score)


# In[58]:


plt.plot(range(2,11),scores,marker='o')
plt.xlabel('k')
plt.ylabel('silhouette score')


# In[59]:


# combined plot
k_range=range(2,11)
fig,ax1=plt.subplots(figsize=(8,6))

ax1.plot(k_range,wcss[:len(k_range)],marker='o',color='blue')
ax1.set_xlabel('k')
ax1.set_ylabel('wcss')


ax2=ax1.twinx()
ax2.plot(k_range,scores[:len(k_range)],marker='x',color='red',linestyle='--')
ax2.set_ylabel('SS')


# # Clustering

# In[60]:


# k means

kmeans=KMeans(n_clusters=4,random_state=42)
labels_kmeans=kmeans.fit_predict(x_pca)


# In[61]:


fig=plt.figure(figsize=(8,6))
ax=fig.add_subplot(111,projection='3d')
ax.scatter(x_pca[:,0],x_pca[:,1],x_pca[:,2],c=labels_kmeans)


# In[62]:


# Agglomerative clustering

from sklearn.cluster import AgglomerativeClustering


# In[63]:


agg_clf=AgglomerativeClustering(n_clusters=4,linkage='ward')
labels_agg=agg_clf.fit_predict(x_pca)


# In[64]:


fig=plt.figure(figsize=(8,6))
ax=fig.add_subplot(111,projection='3d')
ax.scatter(x_pca[:,0],x_pca[:,1],x_pca[:,2],c=labels_agg)


# # characterization of clusters

# In[65]:


x['cluster']=labels_agg


# In[66]:


x.head()


# In[67]:


pal=['red','blue','yellow','green']


# In[68]:


sns.countplot(x=x['cluster'],palette=pal,hue=x['cluster'])


# In[69]:


# income and spending

sns.scatterplot(x=x['Total_Spending'],y=x['Income'],hue=x['cluster'],palette=pal)


# In[70]:


# cluster_summary
cluster_summary=x.groupby('cluster').mean()
print(cluster_summary)


# In[71]:


import joblib

joblib.dump(scaler, "scaler.pkl")
joblib.dump(pca, "pca.pkl")
joblib.dump(kmeans, "cluster_model.pkl")

print("✅ Saved scaler.pkl, pca.pkl, cluster_model.pkl")

