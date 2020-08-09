# Flipkart Grid Software Development Challenge 2.0
Submitted by "Team IRG4" (<a href="https://github.com/avmand">Asrita V Mandalam,</a><a href="https://github.com/sehgal-simran"> Simran Sehgal,</a><a href="https://github.com/milonimittal"> Miloni Mittal</a>)
## Fashion Intelligence Systems :tshirt:
Looking for a solution to predict Fashion Trends and know the Current Trends too? Presenting Unicorn Fashion Systems :unicorn:, a one-stop-shop solution for all the designers out there. You can see numerous current trends and even predict the upcoming trends. No need to browse through multiple fashion sites, e-commerce websites, inspiration boards to know what people are liking. Let Unicorn Fashion Systems :unicorn: do it for you!
<br>
  <p align="center"><img src="https://github.com/avmand/GriD_Fashion/blob/master/images/logo2.png" width=12.5%></p><br>
  
## Use Cases
Show Fashion Retailers current trends and failures and provide them with an analysis of what is working in the market product wise and reduce manual effort. Help fashion retailers know about possible future trends, and give them more insight into how they are derived product wise.
  
## Identifying Current Trends
### To extract current trends, we make use of e-commerce websites. But why? :iphone:
E-commerce websites like Flipkart, Myntra, Nordstorm etc. give a good representation of what people are liking and buying online. The reviews, rating, number of rating, number of reviews give a good impression of what today's trends are. It gives an indirect representation of how a product is doing in the market. A good product would have more positive reviews and higher ratings. This would show that the particular product has features that are being liked by the audience. We can use the images provided by the e-commerce websites to come up with a mood board which showcases what is trending and what is not!

### The Dataset  :page_facing_up:
The dataset consists of data scraped from various e-commerce websites. The details scraped are shown here. 
<br>
  <p align="center"><img src="https://github.com/avmand/GriD_Fashion/blob/master/images/dataset.PNG"></p><br>
  

### Judging the Sentiment Attached with Each Product :smile:/:neutral_face:/ :disappointed:
Here, we make use of the rating, number of people who rated, reviews and number of people who reviewed to understand whether the product is doing well in the market. The Vader Polarity Score <MIT link> is a measure of how postive or negative a certain piece of text is. The relation between these aspects would be:<br>
  <p align="center"><img src="https://github.com/avmand/GriD_Fashion/blob/master/images/3.png" width=37.5%></p><br>
We decided upon this equation because we feel that the positivity score is directly proportional to the above features. We pick the top 5 and bottom 5 products based on their final score. These products are displayed in the current trends section.
  
### The Leaderboard :bar_chart:
The leaderboard displays the features of the clothing item that are trending. For example, the neckline, prints, colours, fit of tshirts, dresses and skirts. This can be extended to any article of clothing since it is extracted from the product description obtained from the e-commerce website.
Mostly, each site follows its own naming convention and the features can be extracted using this convention. If not, we use bigram analysis to figure out these trends after removing stop-words and punctuations. 
<br><p align="center"><img src="https://github.com/avmand/GriD_Fashion/blob/master/images/4.png" width=37.5%></p><br>
 

  
### Concepts and Tech Stack Used :computer:
The concepts and tech stack used here are: 
- Web scraping (Python)
- Natural Language Processing (Python)<br>
  a. Sentiment Analysis<br>
  b. Bigram Analysis
- Data Visualizations (Flask APIs, d3.js)
<br><p align="center"><img src="https://github.com/avmand/GriD_Fashion/blob/master/images/1.png" width=37.5%></p><br>

### The Modules :beginner:
Identifying current trends is based on the following pipeline:<br>
- <b>Scraping Reviews, Ratings and Images</b><br>
We used BeautifulSoup to extract links from the specific category's page. After that, features such as brands, item descriptions, ratings and reviews were extracted from each of the extracted URLs.</br>
<b>a. Input: </b> Input: URL for the required category from any e-commerce website.</br>
<b>b. Output: </b> A csv file with the extracted details. </br>
	 -tshirt-flipkart-final-final.csv </br>
	 -dress-flipkart-final-final.csv </br>
	 -skirt-flipkart-final-final.csv </br>
<b>   c. Code: </b> Review-scrape.py 

- <b>Scoring Products</b><br>
The Vader Polarity Score was used to measure the sentiment of buyers towards the given product. After that, the final score was calculated using the Vader Score and the average rating of a particular product.</br>
<b> a. Input: </b> The CSV files from the previous module</br>
	-tshirt-flipkart-final-final.csv </br>
	-dress-flipkart-final-final.csv </br>
	-skirt-flipkart-final-final.csv </br>
<b> b. Output: </b> CSV files with the each product and its final score.</br>
	 -tshirts_csv_final.csv </br>
	 -dress_csv_final.csv </br>
	 -skirt_csv_final.csv  </br>
<b> c. Code: </b> 
	 -tshirts_current_trends.ipynb </br>
	 -dresses_current_trends.ipynb </br>
	 -skirts_current_trends.ipynb </br>

- <b>Ranking Products</b><br>
This module utilizes Bigram Analysis to rank products. From the description of the products, we removed stopwords (such as Tshirt, Men's, etc). After creating a list of bigrams, they are ranked according to their frequency. After that, we collect the top five and bottom five bigrams.
<br/> For tshirts, we took into account the neck pattern, colour and print.</br>
 <b> a. Input: </b> The CSV files with the each product and its final score.</br>
	-tshirts_csv_final.csv </br>
	 -dress_csv_final.csv </br>
	 -skirt_csv_final.csv </br>
<b> b. Output: </b> CSV files with the top and bottom five products and their IDs.</br>
	 -tshirts/colour_top_bottom.csv </br>
	 -tshirts/neck_top_bottom.csv </br>
	 -tshirts/print_top_bottom.csv </br>
	 -dresses/top_bottom.csv </br>
	 -skirts/top_bottom.csv 	  </br>
<b> c. Code: </b>
	 -tshirts_current_trends.ipynb </br>
	 -dresses_current_trends.ipynb </br>
	 -skirts_current_trends.ipynb </br>

- <b>Currently Trending Leaderboard</b><br>
After extracting the rows in which the top and bottom bigrams appear in, we create a CSV file with the bigram, the ratings of all the products in which the bigram occurs, the bigram itself and the count of each of these.</br>
<b> a. Input: </b> CSV files with the top and bottom five products and their IDs.</br>
	 -tshirts/colour_top_bottom.csv </br>
	 -tshirts/neck_top_bottom.csv </br>
	 -tshirts/print_top_bottom.csv </br>
	 -dresses/top_bottom.csv </br>
	 -skirts/top_bottom.csv </br>
<b> b. Output: </b> The data required for the leaderboard.</br>
	 -tshirt_colour_top_bottom.csv </br>
	 -tshirt_neck_top_bottom.csv </br>
	 -tshirt_print_top_bottom.csv </br>
	 -dress_top_bottom.csv </br>
	 -skirt_top_bottom.csv </br>
<b> c. Code: </b> 
	 -tshirts_graph.py </br>
	 -dress_graph.py </br>
	 -skirts_graph.py 

- <b>Current Trends UI</b><br>
The top and bottom five products are displayed. An interactive bar graph representing the top and bottom five bigrams and their rating is there as well.</br>
<b> a. Input: </b> The data required for the leaderboard.</br>
	 -tshirt_colour_top_bottom.csv </br>
	 -tshirt_neck_top_bottom.csv </br>
	 -tshirt_print_top_bottom.csv </br>
	 -dress_top_bottom.csv </br>
	 -skirt_top_bottom.csv </br>
<b> b. Output: </b> HTML site using a local host </br>
<b> c. Code: </b> MainUI.py </br>
 	 To run it, follow the instructions given <a href="https://timmyreilly.azurewebsites.net/python-flask-windows-development-environment-setup/">here</a> to set up a virtual environment for Flask. Then execute your code using 'python3 MainUI.py' and navigate to your local host address (127.0.0.1:5000). Also, please ensure that you have a stable internet connection as the data visualizations require the source code from the D3.js website.


## Predicting Future Trends
### The Dataset :page_facing_up:
Future trends are guided by what celebrities wear, what is shown in high-end magazines like Vogue and what is being shown in fashion blogs like Fashion Vignette. The trends guided by celebrities can be obtained using their Instagram<image> handles and similarly, scraping blogs and online magazines. We believe that celebrities have the power to influence the fashion trends of the future. <example> We scrape only images of these sources and use them to predict the future trends.

### Deep Learning :snake:
We make use of the Convolutional Neural Networks (CNNs) to extract the different clothing items worn by the person in the image. We then split it into its categories accordingly. After that, we use Generative Adversial Networks (GANs) to envision the future trends in that category. We also use the Deep Dream Model to make inspiration boards for future trends. 

### Concepts and Tech Stack Used :computer:
The concepts and tech stack used here are: 
- Web scraping (Python)
- Convolutional Neural Networks (CNNs)
- Generative Adversial Networks (GANs)
- Trendifying Model (based on DeepDream)
<br><p align="center"><img src="https://github.com/avmand/GriD_Fashion/blob/master/images/Future trends.png" width=37.5% align="center"></p><br>
### The Modules :beginner:
Predicting future trends is based on the following pipeline:
- <b> Webscraper-1 </b>: 
A huge majority of future trends are adopted from what clothing items celebrities endorse. We implemented an instagram scraping module in python to get the latest brand endorsements by instagram influencers to use as input for our Trendifying Model stage later on.<br>
 <b>  a. Input: </b>   List of instagram profiles to scrape from <br>
 <b>  b. Output: </b>  Recent fashion endorsment images from user profiles <br>
 <b>  c. Code: </b> <br> https://github.com/milonimittal/Instagram-Scraper
 
- <b> Webscraper-2 </b>: 
We also implemented a google images scraper to scrape images from the internet according to a given query to generate a dataset to train our CNN Classifier. <br>
 <b> a. Input: </b> list of queris of images to scrape <br>
 <b> b. Output: </b> Images <br>
 <b> c. Code:</b> GriD_Fashion/FutureTrends/GoogleImagesScraper/scrapeImages.py <br>
  
- <b> CNNs:</b><br>
A Convolutional Neural Network is used here to identify the category to which the given article of clothing belongs to. To train this, we used the Myntra Dataset which consisted of nearly 44,500 images to train, validate and test our model. After generating features for the dataset, we saved the data to a NumPy binary file (the .npy files can be downloaded from our google drive link). We used a VGG16 model for transfer learning and obtained an accuracy of 86% on our test dataset. <br>
<b> a. Input: </b>  Myntra Dataset (<a href= "https://drive.google.com/drive/folders/1xqIcVBhV1elZghY7vkzTGh1wh9Gr2erq?usp=sharing">Google Drive link</a>), numpy files and test images from the user. A feature has been included to aid in adding any dataset of clothing.  <br>
<b>  b. Output: </b> Category of the test images. <br>
<b>  c. Code: </b> CNN_clasifier/cnnClassifier.py <br>

- <b> GANS: </b><br>
Generative Adversial Networks consist of 2 neural network models: the generator and the discriminator. The Fashion GAN we implemented learns from a dataset of trendy images (tshirts, skirts and dresses) and generates its own image with a minimum discriminator loss of 0.084. The Generator and the discriminator is made of 7 layers and 6 layers with a leaky RELU activation to avoid the vanishing gradient descent problem. <br>
 <b> a. Input: </b>  Subset of Myntra Dataset: Tshirts (500), Dresses (434), Skirts (128) to train <br>
<b>  b. Output: </b> Images generated by the model as a prediction of future trends (Present in static/img/{dress_images, skirt_images, tshirt_images} <br>
<b>  c. Code: </b>   FutureTrends/fashion_gan.py which uses numpy files for training dataset (resized from original dataset). Numpy Files are present in FutureTrends folder <br>

- <b>Trendifying Model</b>:
The Trendifying Model creates the inspiration board.GANs, though quite accurate and state of the art, are constrained due to the heavy computation power required and large dataset requirements. To get a closer look into how a neural network looks at a dataset of trendy images, we adopted the DeepDream model to trendify certain images. The DeepDream concept increases activations of certain layers so that it exemplifies the features that the certain layer uses to calculate outputs. We adopted the pretrained VGG16 image classifier for this purpose and chose the layer that we thought influenced trend the most by experimentation. <br>
 <b> a. Input:</b> An image which depicts a future trend. <br>
 <b> b. Output:</b> A trendified image <br>
 <b> c. Code:</b> FutureTrends/Trendified.py (Pls change the address of the image you want to use in the code before running)<br>
