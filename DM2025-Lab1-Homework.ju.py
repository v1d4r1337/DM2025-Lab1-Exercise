# %% [markdown]
"""
### Student Information
Name: Vidar Nykvist

Student ID: X1140051

GitHub ID: v1d4r1337
"""

# %% [markdown]
"""
---
"""

# %% [markdown]
"""
### Instructions
"""

# %% [markdown]
"""
### First Phase Submission
"""

# %% [markdown]
"""
1. First: do the **take home** exercises in the [DM2025-Lab1-Master](https://github.com/leoson-wu/DM2025-Lab1-Exercise/blob/main/DM2025-Lab1-Master.ipynb) that considered as **phase 1 (from exercise 1 to exercise 15)**. You can answer in the master file. __This part is worth 10% of your grade.__


2. Second: follow the same process from the [DM2025-Lab1-Master](https://github.com/leoson-wu/DM2025-Lab1-Exercise/blob/main/DM2025-Lab1-Master.ipynb) on **the new dataset** up **until phase 1**. You can skip some exercises if you think some steps are not necessary. However main exercises should be completed. You don't need to explain all details as we did (some **minimal comments** explaining your code are useful though).  __This part is worth 15% of your grade.__
    -  Use [the new dataset](https://github.com/leoson-wu/DM2025-Lab1-Exercise/blob/main/newdataset/Reddit-stock-sentiment.csv). The dataset contains a 16 columns including 'text' and 'label', with the sentiment labels being: 1.0 is positive, 0.0 is neutral and -1.0 is negative. You can simplify the dataset and use only the columns that you think are necessary. 
    
    - You are allowed to use and modify the `helper` functions in the folder of the first lab session (notice they may need modification) or create your own.
    - Use this file to complete the homework from the second part. Make sure the code can be run from the beginning till the end and has all the needed output.


3. Third: please attempt the following tasks on **the new dataset**. __This part is worth 10% of your grade.__
    - Generate meaningful **new data visualizations**. Refer to online resources and the Data Mining textbook for inspiration and ideas. 
    


4. Fourth: It's hard for us to follow if your code is messy, so please **tidy up your notebook** and **add minimal comments where needed**. __This part is worth 5% of your grade.__

You can submit your homework following these guidelines: [DM2025-Lab1-announcement](https://github.com/leoson-wu/DM2025-Lab1-Announcement/blob/main/README.md). Make sure to commit and save your changes to your repository __BEFORE the deadline (September 28th 11:59 pm, Sunday)__. 
"""

# %% [markdown]
"""
### Second Phase Submission 
"""

# %% [markdown]
"""
**You can keep the answer for phase 1 for easier running and update the phase 2 on the same page.**

1. First: Continue doing the **take home** exercises in the [DM2025-Lab1-Master](https://github.com/leoson-wu/DM2025-Lab1-Exercise/blob/main/DM2025-Lab1-Master.ipynb) for **phase 2, starting from Finding frequent patterns**. Use the same master(.ipynb) file. Answer from phase 1 will not be considered at this stage. You can answer in the master file. __This part is worth 10% of your grade.__


2. Second: Continue from first phase and do the same process from the [DM2025-Lab1-Master](https://github.com/leoson-wu/DM2025-Lab1-Exercise/blob/main/DM2025-Lab1-Master.ipynb) on **the new dataset** for phase 2, starting from Finding frequent pattern. You can skip some exercises if you think some steps are not necessary. However main exercises should be completed. You don't need to explain all details as we did (some **minimal comments** explaining your code are useful though).  __This part is worth 15% of your grade.__
    - Continue using this file to complete the homework from the second part. Make sure the code can be run from the beginning till the end and has all the needed output. Use the same new dataset as in phase 1.
    
    - You are allowed to use and modify the `helper` functions in the folder of the first lab session (notice they may need modification) or create your own.

3. Third: please attempt the following tasks on **the new dataset**. __This part is worth 20% of your grade.__
    - Use this file to answer.
    - Generate **TF-IDF features** from the tokens of each text. This will generating a document matrix, however, the weights will be computed differently (using the TF-IDF value of each word per document as opposed to the word frequency).  Refer to this Scikit-learn [guide](http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html) .
    - Implement a simple **Naive Bayes classifier** that automatically classifies the records into their categories. Use both the TF-IDF features and word frequency features to build two seperate classifiers. Note that for the TF-IDF features you might need to use other type of NB classifier different than the one in the Master Notebook. Comment on the differences and when using augmentation with feature pattern.  Refer to this [article](https://hub.packtpub.com/implementing-3-naive-bayes-classifiers-in-scikit-learn/).


4. Fourth: In the lab, we applied each step really quickly just to illustrate how to work with your dataset. There are somethings that are not ideal or the most efficient/meaningful. Each dataset can be handled differently as well. What are those inefficent parts you noticed? How can you improve the Data preprocessing for these specific datasets? __This part is worth 10% of your grade.__


5. Fifth: It's hard for us to follow if your code is messy, so please **tidy up your notebook** and **add minimal comments where needed**. __This part is worth 5% of your grade.__


You can submit your homework following these guidelines: [DM2025-Lab1-announcement](https://github.com/leoson-wu/DM2025-Lab1-Announcement/blob/main/README.md). Make sure to commit and save your changes to your repository __BEFORE the deadline (October 19th 11:59 pm, Sunday)__. 
"""

# %% [markdown]
"""
# Phase 1
"""

# %% [markdown]
"""
# Load data
"""

# %%
import pandas as pd
import numpy as np
import nltk
nltk.download('punkt') # download the NLTK datasets
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
import plotly as py
import math
import helpers.data_mining_helpers as dmh
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


# %matplotlib inline


# load data
new_data_set_file = "./newdataset/Reddit-stock-sentiment.csv"
df = pd.read_csv(new_data_set_file)
print(df.columns)
# explore columns and keep the ones we are interested in
df = df[["text", "label", "subreddit"]]

# %% [markdown]
"""
# Data exploration
"""

# %% [markdown]
"""
### Exercise 1 
"""

# %%
print(df.sample(n=10))
print("\n")

i = 1
linebreaker = "="*30
for text in df.head(n=5)["text"]:
    print(f"{linebreaker} Begin number: {i} {linebreaker}")
    print(f"{text}")
    print(f"{linebreaker} End number: {i} {linebreaker} \n")
    i=i+1


# %% [markdown]
"""
### Exercise 2
"""

# %%
print(type(df))
print(df.columns)
print(df.head(n=10)["text"])
print(df.head(n=10)["label"])
print(df.tail(n=10)["subreddit"])


# %% [markdown]
"""
### Exercise 3
"""

# %%
# Lets look at wallstreetbets
df[df["subreddit"]=="wallstreetbets"][::10].head(n=5)


# %% [markdown]
"""
# Data mining
"""

# %%
# Check for empty elements

dups_df = df[df.duplicated()==True]
print(dups_df)
print(len(dups_df))

df.isnull().apply(lambda x: dmh.check_missing_values(x))

# %% [markdown]
"""
### Exercise 4
"""

# %%
df.isnull().apply(lambda x: dmh.check_missing_values(x), axis=1)

# %% [markdown]
"""
### Exercise 5
"""

# %% [markdown]
"""
Should be the same as in the master file
"""

# %% [markdown]
"""
# Data cleaning
"""

# %%
# Seems like we do not have missing value in our columns but we have duplicates
# lets drop those

clean_df = df.drop_duplicates()
print(clean_df)
df = clean_df

# %% [markdown]
"""
# Sampling
"""

# %% [markdown]
"""
### Exercise 6 
"""

# %%
df_sample = df.sample(n=250)
print(len(df))
print(len(df_sample))
print(df)
print(f"\n {df_sample}")

# %% [markdown]
"""
### Exercise 7
"""

# %%
# Since this is a smaller dataset, maybe we dont need sampling
# But we can sample for fun and see if it has the same label distribution
# Also we can check both subreddit and label distribution

df["label"].value_counts().plot(kind = 'bar',
                                           title = 'Label distribution',
                                           ylim = [0, 800], 
                                           rot = 0, fontsize = 12, figsize = (8,3))



# %%
df["subreddit"].value_counts().plot(kind = 'bar',
                                           title = 'Subreddit distribution',
                                           ylim = [0, 800], 
                                           rot = 0, fontsize = 12, figsize = (8,3))

# %%
df_sample = df.sample(n=250)

df_sample["label"].value_counts().plot(kind = 'bar',
                                           title = 'Label distribution',
                                           ylim = [0, 250], 
                                           rot = 0, fontsize = 12, figsize = (8,3))



# %%
df_sample["subreddit"].value_counts().plot(kind = 'bar',
                                           title = 'Subreddit distribution',
                                           ylim = [0, 250], 
                                           rot = 0, fontsize = 12, figsize = (8,3))

# %% [markdown]
"""
### Exercise 8
"""

# %%
max_count = df["subreddit"].value_counts().max()
pd.concat([df["subreddit"].value_counts(), df_sample["subreddit"].value_counts()], axis=1).plot(kind = 'bar',
                                           title = 'Subreddit distribution',
                                           ylim = [0, max_count+max_count*0.1], 
                                           rot = 0, fontsize = 12, figsize = (8,3))

# %% [markdown]
"""
# Feature creation 
"""

# %%
# Download stop words
from nltk.corpus import stopwords
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

# %% [markdown]
"""
### Create document term matrix
"""

# %%
count_vect = CountVectorizer()
df_counts = count_vect.fit_transform(clean_df["text"])
clean_count_vect = CountVectorizer(stop_words='english')
print(type(df_counts))


# %%
dt_df = pd.DataFrame(df_counts.toarray(), index=df.index) 
# we want to preserve index even after dropping duplicates
stop_words = set(stopwords.words("english"))
feature_names = count_vect.get_feature_names_out()
droplist = [i for i in dt_df.columns if feature_names[i] in stop_words]
cleaned_dt_df = dt_df.drop(droplist, axis=1)
cleaned_df_counts = clean_count_vect.fit_transform(df["text"])

print(cleaned_dt_df)



# %% [markdown]
"""
### Clean document term matrix
"""

# %% [markdown]
"""
### Exercise 9
"""

# %%
analyze = count_vect.build_analyzer()
analyze(df["text"][0])

# %% [markdown]
"""
# Aggregation
"""

# %% [markdown]
"""
### Exercise 12 and 13
"""

# %%
# we use plotly instead here

term_frequencies = np.asarray(cleaned_df_counts.sum(axis=0))[0]
tmp_df = pd.DataFrame({"term": clean_count_vect.get_feature_names_out()[:300],"frequency" :term_frequencies[:300] })
smaller_df = tmp_df[tmp_df["frequency"]>2]
# print(smaller_df)
fig = px.bar(smaller_df, x="term", y="frequency")
fig.show()

# %% [markdown]
"""
# Longtail 
"""

# %% [markdown]
"""
### Exercise 14
"""

# %%
sorted_df = smaller_df.sort_values(by="frequency", ascending=False)
fig = px.bar(sorted_df, x="term", y="frequency")
fig.show()

# %% [markdown]
"""
### Exercise 15
"""

# %%
term_frequencies_log = [math.log(i) for i in term_frequencies]
tmp_df = pd.DataFrame({"term": clean_count_vect.get_feature_names_out()[:300],"frequency" :term_frequencies_log[:300] })
df_smaller = tmp_df[tmp_df["frequency"] > 1] # change threshhold when using log
df_sorted = df_smaller.sort_values(by="frequency", ascending=False)
fig = px.bar(df_sorted, x="term", y="frequency")
fig.show()

# %% [markdown]
"""
# Meaningful visualisations 
"""

# %% [markdown]
"""
## With stop words vs without term document matrix (heatmap and bar plot) (exercise 11)
"""

# %% [markdown]
"""
#### With stop words
"""

# %%
top_20_doc = dt_df.astype(bool).sum(axis=1).nlargest(n=20)
top_20_word = dt_df.sum(axis=0).nlargest(n=20)

plot_x = ["term_" + str(i) for i in count_vect.get_feature_names_out()[top_20_word.index]]
plot_y = ["doc_" + str(i) for i in top_20_doc.index]
plot_z = df_counts[top_20_doc.index][:, top_20_word.index].toarray()

df_todraw = pd.DataFrame(plot_z, columns = plot_x, index = plot_y)
plt.subplots(figsize=(20, 10))
ax = sns.heatmap(df_todraw,
                 cmap="PuRd",
                 fmt="g",
                 vmin=0, vmax=20, annot=True)

# %% [markdown]
"""
#### Without stop words 
"""

# %%
top_20_doc = cleaned_dt_df.astype(bool).sum(axis=1).nlargest(n=20)
top_20_word = cleaned_dt_df.sum(axis=0).nlargest(n=20)

plot_x = ["term_" + str(i) for i in count_vect.get_feature_names_out()[top_20_word.index]]
plot_y = ["doc_" + str(i) for i in top_20_doc.index]
plot_z = df_counts[top_20_doc.index][:, top_20_word.index].toarray() # use old index df_counts since
# cleaned_dt_df kept the same index from previous.

df_todraw = pd.DataFrame(plot_z, columns = plot_x, index = plot_y)
plt.subplots(figsize=(20, 10))
ax = sns.heatmap(df_todraw,
                 cmap="PuRd",
                 fmt="g",
                 vmin=0, vmax=5, annot=True)


# %% [markdown]
"""
## Top 20 most negative and positive associated terms (without stop words, bar plot and word cloud)
"""

# %% [markdown]
"""
#### Most positive 
"""

# %%
# not interested in stop words
labels = clean_df["label"]

# regardless how many times the term occurs in the document
# if the document has score -1 then that adds -1 to that term
binary_cleaned_dt_df = cleaned_dt_df.astype(bool)

term_score = binary_cleaned_dt_df.multiply(labels, axis=0).sum(axis=0)


top_20 = term_score.sort_values().tail(n=20)
top_20_df = pd.DataFrame({"term": count_vect.get_feature_names_out()[top_20.index],"term_score" :top_20 })
fig = px.bar(top_20_df, x="term", y="term_score")
fig.show()
# print(term_score)

# %% [markdown]
"""
#### Most negative 
"""

# %%
last_20 = term_score.sort_values().head(n=20)

last_20_df = pd.DataFrame({"term": count_vect.get_feature_names_out()[last_20.index],"term_score" :last_20 })
fig = px.bar(last_20_df, x="term", y="term_score")
fig.show()

# %% [markdown]
"""
# Phase 2
"""

# %% [markdown]
"""
### Exercise 16 
"""



# %%
### Begin Assignment Here

