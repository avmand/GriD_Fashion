# Flipkart Grid Software Development Challenge 2.0
## Fashion Intelligence System
Looking for a solution to predict Fashion Trends and know the Current Trends too? Presenting Llama Fashions Systems, a one-stop-shop solution for all the designers out there. You can see numerous current trends and even predict the upcoming trends. No need to browse through multiple fashion sites, e-commerce websites, inspiration boards to know what people are liking. Let Llama Fashion do it for you!
## Use Cases
<complete later>
  
## Identifying Current Trends
### To extract current trends, we make use of e-commerce websites. But why?
E-commerce websites like Flipkart, Myntra, Nordstorm etc. give a good representation of what people are liking and buying online. The reviews, rating, number of rating, number of reviews give a good impression of what today's trends are. It gives an indirect representation of how a product is doing in the market. A good product would have more positive reviews and higher ratings. This would show that the particular product has features that are being liked by the audience. We can use the images provided by the e-commerce websites to come up with a mood board which showcases what is trending and what is not!
### The Process
#### The Dataset
The dataset consists of data scraped from various e-commerce websites. The details scraped are shown here. <image>  
  

#### Judging the Sentiment Attached with each Product
Here, we make use of the rating, number of people who rated, reviews and number of people who reviewed to understand whether the product is doing well in the market. The Vader Polarity Score <MIT link> is a measure of how postive or negative a certain piece of text is. The relation between these aspects would be: <image>
We decided upon this equation because we feel that the positivity score is directly proportional to the above features. We pick the top 5 and bottom 5 products based on their final score. These products are displayed in the current trends section.
  
#### The Leaderboard
The leaderboard displays the features of the clothing item that are trending. For example, the neckline, prints, colours, fit of tshirts, dresses and skirts. This can be extended to any article of clothing since it is extracted from the product description obtained from the e-commerce website.
Mostly, each site follows its own naming convention and the features can be extracted using this convention. If not, we use bigram analysis to figure out these trends after removing stop-words and punctuations. <image>
 
<flowchart>
  
#### Concepts and Tech Stack Used
The concepts used here are: 
- Web scraping (Python)
- Natural Language Processing (Python)<br>
  a. Sentiment Analysis<br>
  b. Bigram Analysis
- Data Visualizations (Flask APIs, d3.js)

#### The Modules
<complete later>


## Predicting Future Trends
#### The Dataset





It uses Artificial Intelligence and models like GANs, CNNs. 

