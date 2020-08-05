from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import json
import os
import urllib3
import argparse
import urllib.request

print("define program variables and open google images...")

#write it in the same format as its appearance in the URL
searchterm = 'calvin_klein_dresses'   #how it's written for query 'calvin_klein_dresses'
#searchterm = 'calvin+klein+dresses'   #how it's written for query 'calvin klein dresses'

filename = 'ckdress'  #name of file will start with this
url = "https://www.google.co.in/search?q="+searchterm+"&source=lnms&tbm=isch"

"""
# Insert path to chromedriver inside parentheses in following line
# if it's not in the same folder as this code
"""
browser = webdriver.Chrome()

browser.get(url)
header={'User-Agent':"Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/84.0.4147.89 Safari/537.36"}
counter = 0
succounter = 0

print("start scrolling to generate more images on the page...")
# 500 time we scroll down by 10000 in order to generate more images on the website
for _ in range(500):
    browser.execute_script("window.scrollBy(0,10000)")

print("start scraping ...")
for x in browser.find_elements_by_xpath('//img[contains(@class,"rg_i Q4LuWd")]'):
    counter = counter + 1
    print("Total Count:", counter)
    print("Succsessful Count:", succounter)
    print("URL:", x.get_attribute('src'))

    img = x.get_attribute('src')
    new_filename = "image"+str(counter)+".jpg"

    try:
        path = r'C:\Users\name\folder_that_exists'+'\\' + filename
        path += new_filename
        urllib.request.urlretrieve(img, path)
        succounter += 1
    except Exception as e:
        print(e)

print(succounter, "pictures succesfully downloaded")
browser.close()
