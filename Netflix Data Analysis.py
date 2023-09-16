#!/usr/bin/env python
# coding: utf-8

# # Netflix Data Analysis
# ##### Netflix is an American subscription video on-demand over-the-top streaming service owned and operated by Netflix, Inc. The service primarily distributes films and television series produced by the media company of the same name from various genres, and it is available internationally in multiple languages

# Importing libraries pandas, numpy and matplotlib

# In[112]:


#importing libraries pandas, numpy and matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


# ## Loading CSV Data 

# In[84]:


#reading netflix data and assigning it to variable called df
df = pd.read_csv("Netflix Data.csv")


# ## Reading Data 

# In[85]:


#getting top 10 rows from the database
df.head(3)


# In[86]:


df.tail(3)


# In[87]:


#Data type of all attributes
print(df.dtypes)


# In[106]:


df.release_year = df.release_year.apply(pd.to_numeric)
print(df.dtypes)


# In[90]:


df.describe()


# In[91]:


df.nunique()


# In[92]:


#getting shape of the data meand number of rows and columns
df.shape


# In[93]:


#getting all the column names
df.columns


# ## Handling Null Values

# In[94]:


#getting all the null value count in all columns
df.isnull().sum()


# In[95]:


df['director'].fillna('No Director', inplace=True)
df['cast'].fillna('No Cast', inplace=True)
df['country'].fillna('Country Unavailable', inplace=True)
df.head(5)


# In[96]:


#getting all the null value count in all columns
df.isnull().sum()


# In[97]:


df.dropna(subset=['date_added'],inplace=True)
df.dropna(subset=['rating'],inplace=True)
df.dropna(subset=['duration'],inplace=True)


# In[98]:


df.isnull().sum()


# ## Data Cleansing

# In[99]:


#checking value count in Type column 
df["type"].value_counts()


# In[100]:


#checking index number of incorrect data in column 
df[df["type"]== "William Wyler"]


# In[101]:


#dropping row where incorrect data existed
df.drop(8421,axis = 0, inplace = True)
df.head()


# In[102]:


#checking shape of data after dropping a row where incorrect data existed
df.shape


# In[103]:


#checking updated data in column type after dropping incorrect value
cat_counts = df["type"].value_counts()
cat_counts


# In[104]:


#checking rating column with number of value counts 
df["rating"].value_counts()


# In[105]:


df.head(2)


# In[24]:


df["cast"] = df["cast"].str.split(",")
df["country"] = df["country"].str.split(",")
df["listed_in"] = df["listed_in"].str.split(",")


# In[25]:


df.head(2)


# In[26]:


df_cast = df.explode("cast",ignore_index = True)


# In[27]:


df_cast.head()


# In[28]:


df_country = df.explode("country",ignore_index = True)


# In[29]:


df_country.head(3)


# ## Data Visualization

# In[30]:


index = cat_counts.index
value = cat_counts.values
index
value
plt.figure(figsize = (4,2))
plt.title("Shows Count")
plt.bar(index,value,color = ["Red", "Black"])
plt.xlabel("Show Type",fontsize = 8)
plt.ylabel("Count of Show",fontsize = 8)
plt.show()


# In[63]:


labels = df.type.value_counts().index
plt.figure(figsize=(12,6))
plt.title("Percentation of Netflix Titles that are either Movies or TV Shows")
plt.pie(df.type.value_counts(),labels = labels, colors=["red","black"],autopct="%1.1f%%")
plt.show()


# In[64]:


country_count = df_country.country.value_counts()[:10]
country_count


# In[65]:


index = country_count.index
value = country_count.values
plt.figure(figsize=(10,2))
plt.xticks(rotation = 30)
plt.title("Top 10 Countries with max number of shows")
plt.plot(index,value,marker = "*",color = "Red")
plt.xlabel("Country Name",fontsize = 8)
plt.ylabel("Count of Show",fontsize = 8)
plt.show()


# In[34]:


df_listed_in = df.explode("listed_in",ignore_index = True)


# In[35]:


df_listed_in.head()


# In[36]:


listed_in_count = df_listed_in.listed_in.value_counts()
listed_in_count


# In[66]:


plt.title("Top 10 Genres available on Netflix for Movies")
plt.ylabel("Genre",fontsize = 8)
plt.xlabel("Count of Show",fontsize = 8)
sns.boxplot(data = df_listed_in, 
            x = df_listed_in[df_listed_in["type"] == "Movie"]["listed_in"].value_counts().values[:10], 
            y = df_listed_in[df_listed_in["type"] == "Movie"]["listed_in"].value_counts().index[:10],
            color = "Red")
plt.show()


# In[38]:


plt.title("Top 10 Genres available on Netflix for TV Shows")
plt.ylabel("Genre",fontsize = 8)
plt.xlabel("Count of Show",fontsize = 8)
sns.boxplot(data = df_listed_in, 
            x = df_listed_in[df_listed_in["type"] == "TV Show"]["listed_in"].value_counts().values[:10], 
            y = df_listed_in[df_listed_in["type"] == "TV Show"]["listed_in"].value_counts().index[:10],
            color = "Red")
plt.show()


# In[67]:


plt.figure(figsize = (12,2))  
df[df["type"] == "Movie"]["release_year"].value_counts()[:20].plot(kind = "bar",color = "Red")  
plt.title("Top 20 year in which maximum moview released")  
plt.show()


# In[70]:


# Distplot for 'release_year'
plt.figure(figsize=(15, 5))
ax = sns.histplot(data =df, x = "release_year", bins = 12, kde=True, color="g")
plt.title('Distribution of Release Year')
plt.xlabel('Release Year')
plt.ylabel('Content Produced per Year')
plt.legend(labels=['KDE'])
plt.show()
# Countplot for 'release_year'
plt.figure(figsize=(10, 10))
ax = sns.countplot(y='release_year', data=df, order=df['release_year'].value_counts().index, palette="viridis")
plt.title('Content Produced per Year')
plt.xlabel('Count')
plt.ylabel('Release Year')
# Adding labels to the bars
for p in ax.patches:
 width = p.get_width()
 plt.text(width + 10, p.get_y() + p.get_height()/2, int(width), va='center')
plt.show()


# In[71]:


plt.figure(figsize = (12,2))  
df[df["type"] == "TV Show"]["release_year"].value_counts()[:20].plot(kind = "bar",color = "Red")  
plt.title("Top 20 year in which maximum TV Show released")  
plt.show()


# In[73]:


sns.boxplot(x='type', y='release_year', data=df)
plt.title('Boxplot of Release Year by Type')
plt.xlabel('Type')
plt.ylabel('Realease Year')
plt.show()


# In[72]:


plt.figure(figsize=(12, 6))
sns.boxplot(x='rating', y='release_year', data=df)
plt.title('Boxplot of Release Year by Rating')
plt.xlabel('Rating')
plt.ylabel('Release Year')
plt.xticks(rotation=45)
plt.show()


# In[403]:


df.rating.value_counts()


# In[404]:


filtered_directors = df[df.director != 'No Director'].director.str.split(', ', expand=True).stack()
sns.countplot(y = filtered_directors, order=filtered_directors.value_counts().index[:10], palette='dark:salmon_r')
plt.title("Top 10 Directors")
plt.show()


# In[405]:


plt.figure(figsize=(15,4))
sns.countplot(x='rating',hue='type',data=df, palette='dark:salmon_r' )
plt.title('Rating (Movies / TV Shows)')
plt.show()


# In[406]:


#Which individual country has the highest no: of TV Shows?
df[df['type']=='TV Show']['country'].value_counts().head(1)


# In[77]:


plt.figure(figsize=(8, 5))
ax = sns.countplot(data=df, x='type')
plt.title('Distribution by Type')
plt.xlabel('Type')
plt.ylabel('Count')
for p in ax.patches:
 ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
 ha='center', va='center', xytext=(0, 10), textcoords='offset points')
plt.show()

#Distribution of TV Show and Movie releases over the Years

plt.figure(figsize=(12, 6))
sns.countplot(data=df, x='release_year', hue='type')
plt.title('Distribution of Movies and TV Shows Released Over the Years')
plt.xlabel('Release Year')
plt.ylabel('Frequency')
plt.legend(title='Type')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
plt.figure(figsize=(12, 6))

#Distribution of TV Show and Movie releases over the Years with labels

plt.figure(figsize=(12, 6))
ax = sns.countplot(data=df, x='release_year', hue='type')
plt.title('Distribution of Movies and TV Shows Released Over the Years')
plt.xlabel('Release Year')
plt.ylabel('Frequency')
plt.legend(title='Type')
plt.xticks(rotation=90)
for p in ax.patches:
 ax.annotate(f'{int(p.get_height())}',
 (p.get_x() + p.get_width() / 2., p.get_height()),
 ha='center',
 va='center',
 xytext=(0, 10),
 textcoords='offset points',
 rotation=90)
plt.tight_layout()
plt.show()

# Distribution of content ratings

ax = sns.countplot(y='rating', data=df, order=df['rating'].value_counts().index)
plt.xlabel('Count')
plt.ylabel('Rating')
for p in ax.patches:
 width = p.get_width()
 plt.text(width + 10, p.get_y() + p.get_height() / 2, int(width), ha="center")
plt.show()


# Trend of content production over the years

yearly_production = df['release_year'].value_counts().sort_index()
plt.plot(yearly_production.index, yearly_production.values)
plt.xlabel('Year')
plt.ylabel('Number of Shows')
plt.show()

# Distribution of content across different genres

plt.figure(figsize=(10, 12))
plt.title("Top 10 Genres available on Netflix for Movies")
plt.ylabel("Genre",fontsize = 8)
plt.xlabel("Count of Show",fontsize = 8)
sns.boxplot(data = df_listed_in, 
            x = df_listed_in[df_listed_in["type"] == "Movie"]["listed_in"].value_counts().values[:10], 
            y = df_listed_in[df_listed_in["type"] == "Movie"]["listed_in"].value_counts().index[:10],
            color = "Red")
plt.show()

# DIstribution of content type

df_count = df.groupby(['release_year', 'type']).size().reset_index(name='count')
plt.figure(figsize=(10, 6))
ax = sns.lineplot(x='release_year', y='count', hue='type', data=df_count)
plt.title('Content Produced per Year')
plt.xlabel('Release Year')
plt.ylabel('Count')
plt.xticks(rotation=45)
ax.legend(title='Type')
plt.show()

# Heatmap for 'type' and 'rating'

cross_tab = pd.crosstab(df['type'], df['rating'])
plt.figure(figsize=(10, 5))
sns.heatmap(cross_tab, annot=True, cmap='viridis', fmt='d')
plt.title('Heatmap of Type vs Rating')
plt.xlabel('Rating')
plt.ylabel('Type')
plt.show()


# In[78]:


plt.figure(figsize=(10, 6))
sns.violinplot(x='type', y='release_year', data=df)
plt.title('Content Type vs Release Year')
plt.xlabel('Content Type')
plt.ylabel('Release Year')
plt.show()


# In[107]:


# Trend of Individual Genre over Time
df_exploded = df.assign(listed_in=df['listed_in'].str.split(', ')).explode('listed_in')
df_grouped = df_exploded.groupby(['release_year', 'listed_in', 'type']).size().reset_index(name='count')
genres = df_exploded['listed_in'].unique()
for genre in genres:
 plt.figure(figsize=(10, 5))
 
 data = df_grouped[df_grouped['listed_in'] == genre]
 sns.lineplot(x='release_year', y='count', hue='type', data=data)
 
 plt.title(f'{genre} Trend Over Time')
 plt.xlabel('Release Year')
 plt.ylabel('Count')
 
 plt.xticks(rotation=45)
 plt.tight_layout()
 plt.show()


# In[109]:


# Genre Trend Over Time
df_exploded = df.assign(listed_in=df['listed_in'].str.split(', ')).explode('listed_in')
df_grouped = df_exploded.groupby(['release_year', 'listed_in']).size().reset_index(name='count')
plt.figure(figsize=(15, 8))
sns.lineplot(x='release_year', y='count', hue='listed_in', data=df_grouped)
plt.title('Genre Trend Over Time')
plt.xlabel('Release Year')
plt.ylabel('Count')
plt.legend(title='Genre', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[110]:


#Outliers
plt.figure(figsize=(10, 6))
sns.boxplot(x='release_year', data=df)
plt.title('Boxplot of Release Year')
plt.show()


# In[113]:


#Outliers
#The Z-score is the number of standard deviations a data point is from the mean. 
#A Z-score higher than 3 or lower than -3 usually indicates an outlier.

z_scores = np.abs(stats.zscore(df['release_year']))
outliers = z_scores > 3
total_outliers = np.sum(outliers)
print(f'Total outliers: {total_outliers}')


# In[115]:


#Treatment of Outliers
#Method 1 - By removing the outliers
df_filtered = df[~outliers]


# In[116]:


#Treatment of Outliers
#Method 2 - By replacing with mean or median
median = df['release_year'].median()
df['release_year'] = np.where(outliers, median, df['release_year'])


# In[117]:


#Treatment of Outliers
#Method 3 - By capping the outliers
upper_bound = df['release_year'].quantile(0.99) # 99th percentile
df['release_year'] = np.where(df['release_year'] > upper_bound, upper_bound, df['release_year'])


# In[119]:


#Exploding the cast column to facilitate filtration on the basis of actors
df['cast'] = df['cast'].astype(str)
df_exploded = df.assign(cast=df['cast'].str.split(', ')).explode('cast')
df_exploded.reset_index(drop=True, inplace=True)
df_exploded['cast'].replace('nan', np.nan, inplace=True)
df_exploded


# In[120]:


# Getting the number of Movies and TV Show for each Cast
cast_type_count = df_exploded.groupby(['cast', 'type']).size().unstack(fill_value=0)
cast_type_count.columns = [f'Number of {col}' for col in cast_type_count.columns]
cast_type_count.sort_values(by=['Number of Movie', 'Number of TV Show'], ascending=[False, False], inplace=True)
cast_type_count.reset_index(inplace=True)
cast_type_count


# In[122]:


# Filtering for Movies
cast_movies_count = cast_type_count[['cast', 'Number of Movie']].sort_values(by='Number of Movie', ascending=False)
# Plotting the top 20 cast members by movie count
plt.figure(figsize=(12, 8))
cast_movies_count.head(20).plot(x='cast', kind='bar')
plt.title('Top 20 Cast Members by Movie Count')
plt.xlabel('Cast')
plt.ylabel('Number of Movies')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()


# In[123]:


cast_movies_count = cast_type_count[['cast', 'Number of Movie']].sort_values(by='Number of Movie', ascending=False)
print(cast_movies_count.head(20))


# In[124]:


cast_tv_shows_count = cast_type_count[['cast', 'Number of TV Show']].sort_values(by='Number of TV Show', ascending=False)
plt.figure(figsize=(12, 8))
cast_tv_shows_count.head(20).plot(x='cast', kind='bar', color='orange')
plt.title('Top 20 Cast Members by TV Show Count')
plt.xlabel('Cast')
plt.ylabel('Number of TV Shows')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()


# In[125]:


cast_tv_shows_count = cast_type_count[['cast', 'Number of TV Show']].sort_values(by='Number of TV Show', ascending=False)
print(cast_tv_shows_count.head(20))


# In[126]:


plt.figure(figsize=(15, 7))
ax = sns.countplot(data=df, x='rating', order=df['rating'].value_counts().index)
plt.title('Distribution by Rating')
plt.xlabel('Rating')
plt.ylabel('Count')
plt.xticks(rotation=45)
for p in ax.patches:
 ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
 ha='center', va='center', xytext=(0, 10), textcoords='offset points')
plt.show()


# In[133]:


tv_shows_df = df[df['type'] == 'TV Show']
movies_df = df[df['type'] == 'Movie']
# Combining the two dataframes
combined_df = pd.concat([movies_df, tv_shows_df])
# Plotting TV shows and movies based on ratings with bin labels at the top
plt.figure(figsize=(12, 6))
ax = sns.countplot(data=combined_df, x='rating', hue='type', order=combined_df['rating'].value_counts().index)
plt.title('Movies and TV Shows by Rating')
plt.xlabel('Rating')
plt.ylabel('Count')
plt.xticks(rotation=45, ha='right')
# Adding bin labels at the top of the bins at an angle of 45 degrees (as integers)
for p in ax.patches:
    height = p.get_height()
    if height > 0:
        ax.annotate(str(int(height)), (p.get_x() + p.get_width() / 2., height), ha='center', va='bottom', fontsize=10, rotation=4)
plt.legend(title='Type')
plt.tight_layout()
plt.show()


# In[137]:


country_type_counts = df.groupby(['country', 'type']).size().unstack().fillna(0)
country_type_counts['Movie'] = country_type_counts['Movie'].astype(int)
country_type_counts['TV Show'] = country_type_counts['TV Show'].astype(int)
country_type_counts['total'] = country_type_counts.sum(axis=1)
sorted_countries = country_type_counts.sort_values(by='total', ascending=False)
sorted_countries['total'] = sorted_countries['total'].astype(int)
top_10 = sorted_countries.head(10)
plt.figure(figsize=(16, 9))
ax = top_10[['Movie', 'TV Show']].plot(kind='bar', stacked=False, figsize=(16, 9))
plt.title('Top 10 Countries by Number of Movies & TV Shows')
plt.xlabel('Country')
plt.ylabel('Count')
plt.legend(title='Type')
plt.xticks(rotation=45)
for p in ax.patches:
 width, height = p.get_width(), p.get_height()
 x, y = p.get_xy()
 ax.text(x + width / 2,
 y + height + 50,
 '{:.0f}'.format(height),
 horizontalalignment='center',
 verticalalignment='center')
plt.tight_layout()
plt.show()


# In[138]:


us_india_df = df[df['country'].isin(['United States', 'India'])]
us_india_df = us_india_df.assign(genre=us_india_df['listed_in'].str.split(', ')).explode('genre')
plt.figure(figsize=(15, 8))
sns.countplot(data=us_india_df, y='genre', hue='country', order=us_india_df['genre'].value_counts().index)
plt.title('Distribution of Genres for US and India')
plt.xlabel('Count')
plt.ylabel('Genre')
plt.legend(title='Country')
plt.tight_layout()
plt.show()


# In[141]:


us_df = df[df['country'] == 'United States']
us_df['month_added'] = pd.to_datetime(us_df['date_added']).dt.month_name()
us_df = us_df.assign(genre=us_df['listed_in'].str.split(', ')).explode('genre')
plt.figure(figsize=(18, 10))
sns.countplot(data=us_df, x='month_added', hue='genre', order=["January", "February", "March", "April", "May", "June", "July", "August","September", "October", "November", "December"])
plt.title('Monthly Distribution of Genres in US')
plt.xlabel('Month')
plt.ylabel('Count')
plt.legend(title='Genre', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[142]:


india_df = df[df['country'] == 'India']
india_df['month_added'] = pd.to_datetime(india_df['date_added']).dt.month_name()
india_df = india_df.assign(genre=india_df['listed_in'].str.split(', ')).explode('genre')
plt.figure(figsize=(18, 10))
sns.countplot(data=india_df, x='month_added', hue='genre', order=["January", "February", "March", "April", "May", "June", "July", "August","September", "October", "November", "December"]) 
plt.title('Monthly Distribution of Genres in India')
plt.xlabel('Month')
plt.ylabel('Count')
plt.legend(title='Genre', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[149]:


india_df = df[df['country'] == 'India']
india_df['month_added'] = pd.to_datetime(india_df['date_added']).dt.month_name()
plt.figure(figsize=(18, 8))
ax_india = sns.countplot(data=india_df, x='month_added', hue='type', order=["January", "February", "March", "April", "May", "June", "July", "August","September", "October", "November", "December"]) 
plt.title('Monthly Distribution of TV Shows & Movies in India')
plt.xlabel('Month')
plt.ylabel('Count')
plt.legend(title='Type')
plt.xticks(rotation=90)
plt.tight_layout()
for p in ax_india.patches:
 ax_india.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center',fontsize=18, rotation=4)
plt.show()


# ## Data Analysis

# ## Movies Vs. TV Shows
# ##### Netflix contains more movies than TV shows.
# ##### It indicates that Movies has more audience than TV Shows.
# ## Releasing Year Analysis
# ##### Most no:of releases are during the period 2000-2020
# ##### Highest no: of releases are in the year 2018
# ## Released country Analysis
# ##### Netflix has a greater no: of contents from United States.
# ##### India stands out in the second position.
# ## Top Geners of TV Shows / Movies
# ##### Netflix has most number of TV Shows and Movies under International Gener followed by Dramas
# ## Top Ratings Movies / TV Shows
# ##### TV-MA has most number of Movies and TV Shows followed by TV-14
# ## Top Directors on Netflix in terms of releasing number of Movies / TV Shows
# ##### Rajiv Chilaka released most number of Movies and TV Shows followed by Jan Suter

# # Netflix age rating for U. S.
# ## For Kids
# ### TV-Y | This category is appropriate for all kids.
# ### TV-Y7 | This category is appropriate for all kids above the age of 7.
# ### G | It means it is suitable for all general audience
# ### TV-G | It is suitable for the general audience
# ### PG | It means the movie\series under this category requires parental guidance
# ### TV-PG | Again, this means the movie\ series requires parental guidance.
# ## For Teenagers
# ### PG-13 | It means the series\ movie may not be suitable for teens below 12
# ### TV-14 | It means the series may not be suitable for teens under the age of 14. 
# ## For Adults
# ### R | R stands for restricted. It may not be suitable for people under 17.
# ### TV-MA | Suitable for a mature audience, not suitable for people under 17
# ### NC-17 | Not suitable for ages under 17.
