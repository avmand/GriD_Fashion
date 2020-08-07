#!/usr/bin/env python
# coding: utf-8

# In[1]:


from bs4 import BeautifulSoup as bs
import pandas as pd
import requests
import re
import numpy as np


# In[15]:


#The url for flipkart>tshirts. Can replace with any other product category
url="https://www.flipkart.com/clothing-and-accessories/bottomwear/skirts/pr?sid=clo,vua,iku&otracker=categorytree"
x= requests.get(url)
soup= bs(x.content, "html.parser")


# In[16]:


page=soup.findAll("a", {"class": "_2Xp0TH"})
page


# In[17]:


#Getting links for all pages. Each page has 40 items
page_links=[]
for p in page:
    page_links.append(p.get('href'))
page_links=['http://flipkart.com'+p for p in page_links ]


# In[18]:


#Getting individual product links of all t shirts listed (400)
linkss=[]
for p in page_links:
    url=p
    x= requests.get(url)
    soup= bs(x.content, "html.parser")
    links=soup.findAll("a", {"class": "_3dqZjq"})
    for link in links:
        linkss.append('http://flipkart.com'+link.get('href'))
        
len(linkss)


# In[19]:


linkss[0]


# In[20]:


#Extracts the following features from each item and stores them in a list, to be stored as dataframe later on
urls=[]
brands=[]
item_names=[]
disc_prices=[]
mrp_prices=[]
stars=[]
ratings=[]
reviews=[]
text_reviews=[]
type_=[]
sleeve_=[]
fit_=[]
fabric_=[]
neck_=[]
pattern_=[]
brand_fit_=[]
brand_color_=[]

shirts=linkss



for i in range(len(shirts)): 
    
    try:
        print(i)
        url=shirts[i]
        x= requests.get(url)
        soup= bs(x.content, "html.parser")
        
        brand=soup.findAll("span", {"class": "_2J4LW6"})[0].text
        
        item=soup.findAll("span", {"class": "_35KyD6"})[0].text
        
        disc_price=soup.findAll("div", {"class": "_1vC4OE _3qQ9m1"})[0].text
        
        mrp=soup.findAll("div", {"class": "_3auQ3N _1POkHg"})[0].text
        
        star=soup.findAll("div", {"class": "hGSR34 bqXGTW"})[0].text
        
        rating_number=soup.findAll("span", {"class": "_38sUEc"})[0].text    
        ratings_num=(rating_number[0:rating_number.find('ratings')-1])
        
        reviews_num=rating_number[rating_number.find('and')+4:rating_number.find('reviews')-1]
        
        review=[]
        x=soup.findAll("div", {"class": "_2t8wE0"})
        for i in range(len(x)):
            review.append(x[i].text)
        
        
        feat=soup.findAll("div", {"class": "col col-3-12 _1kyh2f"})
        feat_ans=soup.findAll("div", {"class": "col col-9-12 _1BMpvA"})
        features={}
        for x in range(len(feat)):
            features[feat[x].text]=feat_ans[x].text
        urls.append(url)
        brands.append(brand)
        item_names.append(item)
        disc_prices.append(disc_price)
        mrp_prices.append(mrp)
        stars.append(star)
        ratings.append(ratings_num)
        reviews.append(reviews_num)
        text_reviews.append(review)
        type_.append(features['Type'])
        sleeve_.append(features['Sleeve'])
        #fit_.append(features['Fit'])
        fabric_.append(features['Fabric'])
        #neck_.append(features['Neck Type'])
        pattern_.append(features['Pattern'])
        #brand_fit_.append(features['Brand Fit'])
        #brand_color_.append(features['Brand Color'])
    except:
        print("Incomplete details for item"+str(i)+url)
    
    
        
        
    
    


# In[21]:


#convert ratings,reviews, stars to int and floats for easy calculation later
y=[int(r.replace(',','')) for r in ratings]
ratings=y
reviews=[(r.replace(',','')) for r in reviews]
stars=[float(r) for r in stars]

#creating a dataframe and saving it to csv
table={'URL':urls,'BRAND':brands,'ITEM':item_names,'DISCOUNTED PRICE':disc_prices,'MRP':mrp_prices,'STARS':stars,'NUMBER OF RATINGS':ratings,'NUMBER OF REVIEWS':reviews,'LIST OF REVIEWS':text_reviews}
df=pd.DataFrame(table)
df.to_csv(r'skirt-flipkart-final-final.csv')


# In[317]:




