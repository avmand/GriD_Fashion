# Flipkart Grid Software Development Challenge 2.0
## Fashion Intelligence Systems
Looking for a solution to predict Fashion Trends and know the Current Trends too? Presenting Unicorn Fashions Systems, a one-stop-shop solution for all the designers out there. You can see numerous current trends and even predict the upcoming trends. No need to browse through multiple fashion sites, e-commerce websites, inspiration boards to know what people are liking. Let Unicorn Fashion do it for you!
## Use Cases
Show Fashion Retailers current trends and failures and provide them with an analysis of what is working in the market product wise and reduce manual effort. Help fashion retailers know about possible future trends, and give them more insight into how they are derived product wise.
  
## Identifying Current Trends
### To extract current trends, we make use of e-commerce websites. But why?
E-commerce websites like Flipkart, Myntra, Nordstorm etc. give a good representation of what people are liking and buying online. The reviews, rating, number of rating, number of reviews give a good impression of what today's trends are. It gives an indirect representation of how a product is doing in the market. A good product would have more positive reviews and higher ratings. This would show that the particular product has features that are being liked by the audience. We can use the images provided by the e-commerce websites to come up with a mood board which showcases what is trending and what is not!

### The Dataset
The dataset consists of data scraped from various e-commerce websites. The details scraped are shown here. <image>  
  

### Judging the Sentiment Attached with Each Product
Here, we make use of the rating, number of people who rated, reviews and number of people who reviewed to understand whether the product is doing well in the market. The Vader Polarity Score <MIT link> is a measure of how postive or negative a certain piece of text is. The relation between these aspects would be: <image>
We decided upon this equation because we feel that the positivity score is directly proportional to the above features. We pick the top 5 and bottom 5 products based on their final score. These products are displayed in the current trends section.
  
### The Leaderboard
The leaderboard displays the features of the clothing item that are trending. For example, the neckline, prints, colours, fit of tshirts, dresses and skirts. This can be extended to any article of clothing since it is extracted from the product description obtained from the e-commerce website.
Mostly, each site follows its own naming convention and the features can be extracted using this convention. If not, we use bigram analysis to figure out these trends after removing stop-words and punctuations. <image>
 
<flowchart>
  
### Concepts and Tech Stack Used
The concepts and tech stack used here are: 
- Web scraping (Python)
- Natural Language Processing (Python)<br>
  a. Sentiment Analysis<br>
  b. Bigram Analysis
- Data Visualizations (Flask APIs, d3.js)

### The Modules
<complete later>


## Predicting Future Trends
### The Dataset
Future trends are guided by what celebrities wear, what is shown in high-end magazines like Vogue and what is being shown in fashion blogs like Fashion Vignette. The trends guided by celebrities can be obtained using their Instagram<image> handles and similarly, scraping blogs and online magazines. We believe that celebrities have the power to influence the fashion trends of the future. <example> We scrape only images of these sources and use them to predict the future trends.

### Deep Learning
We make use of the Convolutional Neural Networks (CNNs) to extract the different clothing items worn by the person in the image. We then split it into its categories accordingly. After that, we use Generative Adversial Networks (GANs) to envision the future trends in that category. We also use the Deep Dream Model to make inspiration boards for future trends. 

### Concepts and Tech Stack Used
The concepts and tech stack used here are: 
- Web scraping (Python)
- Convolutional Neural Networks (CNNs)
- Generative Adversial Networks (GANs)
- Deep Dream Model

### The Modules
<complete later>
