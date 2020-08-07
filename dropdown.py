import flask
from flask import request, jsonify
from flask import Flask, send_file, render_template
	
from flask import Flask, flash, redirect, render_template, request, session, abort,send_from_directory,send_file,jsonify
import pandas as pd

import json

app = flask.Flask(__name__)
app.config["DEBUG"] = True

#2. Declare data stores
class DataStore():
    Prod= None
data=DataStore() 
    
@app.route('/')
def home(): 
    return render_template("website.html")  



@app.route('/currenttrends', methods=["GET","POST"])
def api_currenttrends():
    if 'tshirts' in request.args:
#        Insert code here/ information to retrieve
        top1_name=[]
        top1_description=[]
        top1_score=[]
        bottom1_name=[]
        bottom1_description=[]
        bottom1_score=[]
        for i in range(0,3):
            top1_name.append("name"+str(i+1))
            top1_description.append("description"+str(i+1))
            top1_score.append("score"+str(i+1))
            bottom1_name.append("name"+str(i+1))
            bottom1_description.append("description"+str(i+1))
            bottom1_score.append("score"+str(i+1))
        
        
        
            #    CountryName = request.form.get('Country_field','India')
    #    Year = request.form.get('Year_field', 2013)
        
    #    data.CountryName=CountryName
    #    data.Year=Year
        
        df = pd.read_csv('currenttrends.csv')
        # dfP=dfP
        
        
        # print(CountryName)
        #Year = data.Year
        #data.Year = Year
    
        # choose columns to keep, in the desired nested json hierarchical order
    #    df = df[df.Country == CountryName]
    #    df = df[df.Year == int(Year)]
        print(df.head())
        # df = df.drop(
        # ['Country', 'Item Code', 'Flag', 'Unit', 'Year Code', 'Element', 'Element Code', 'Code', 'Item'], axis=1)
        df = df[["Category", "Cat", "value"]]
    
        # order in the groupby here matters, it determines the json nesting
        # the groupby call makes a pandas series by grouping 'the_parent' and 'the_child', while summing the numerical column 'child_size'
        df1 = df.groupby(['Category', 'Cat'])['value'].sum()
        df1 = df1.reset_index()
    
        # start a new flare.json document
        flare = dict()
        d = {"name": "flare", "children": []}
    
        for line in df1.values:
            Category = line[0]
            Cat = line[1]
            value = line[2]
    
            # make a list of keys
            keys_list = []
            for item in d['children']:
                keys_list.append(item['name'])
    
            # if 'the_parent' is NOT a key in the flare.json yet, append it
            if not Category in keys_list:
                d['children'].append({"name": Category, "children": [{"name": Cat, "size": value}]})
    
            # if 'the_parent' IS a key in the flare.json, add a new child to it
            else:
                d['children'][keys_list.index(Category)]['children'].append({"name": Cat, "size": value})
    
        flare = d
        e = json.dumps(flare)
        data.Prod = json.loads(e)
        Prod=data.Prod

        
        
        
        
        
            
        return render_template("website_currenttrends.html",Prod=Prod, top1_name=top1_name,top1_description=top1_description,top1_score=top1_score,bottom1_name=bottom1_name,bottom1_description=bottom1_description,bottom1_score=bottom1_score)
    if 'dresses' in request.args:
#        Insert code here/ information to retrieve
        top1_name=[]
        top1_description=[]
        top1_score=[]
        bottom1_name=[]
        bottom1_description=[]
        bottom1_score=[]
        for i in range(0,3):
            top1_name.append("name"+str(i+1))
            top1_description.append("description"+str(i+1))
            top1_score.append("score"+str(i+1))
            bottom1_name.append("name"+str(i+1))
            bottom1_description.append("description"+str(i+1))
            bottom1_score.append("score"+str(i+1))
            
        return render_template("website_currenttrends.html",top1_name=top1_name,top1_description=top1_description,top1_score=top1_score,bottom1_name=bottom1_name,bottom1_description=bottom1_description,bottom1_score=bottom1_score)
    if 'skirts' in request.args:
#        Insert code here/ information to retrieve
        top1_name=[]
        top1_description=[]
        top1_score=[]
        bottom1_name=[]
        bottom1_description=[]
        bottom1_score=[]
        for i in range(0,3):
            top1_name.append("name"+str(i+1))
            top1_description.append("description"+str(i+1))
            top1_score.append("score"+str(i+1))
            bottom1_name.append("name"+str(i+1))
            bottom1_description.append("description"+str(i+1))
            bottom1_score.append("score"+str(i+1))
            
        return render_template("website_currenttrends.html",top1_name=top1_name,top1_description=top1_description,top1_score=top1_score,bottom1_name=bottom1_name,bottom1_description=bottom1_description,bottom1_score=bottom1_score)
    


@app.route("/currenttrendsget-data",methods=["GET","POST"])
def returnProdDatacurrenttrends():
    f=data.Prod

    return jsonify(f)

    
    
@app.route('/futuretrends', methods=['GET'])
def api_futuretrends():
    if 'tshirts' in request.args:
#        Insert code here/ information to retrieve
        top1_name=[]
        top1_description=[]
        for i in range(0,3):
            top1_name.append("name"+str(i+1))
            top1_description.append("description"+str(i+1))
            
        return render_template("website_futuretrends.html",top1_name=top1_name,top1_description=top1_description)
    if 'dresses' in request.args:
#        Insert code here/ information to retrieve
        top1_name=[]
        top1_description=[]
        for i in range(0,3):
            top1_name.append("name"+str(i+1))
            top1_description.append("description"+str(i+1))
            
        return render_template("website_futuretrends.html",top1_name=top1_name,top1_description=top1_description)
    if 'skirts' in request.args:
#        Insert code here/ information to retrieve
        top1_name=[]
        top1_description=[]
        for i in range(0,3):
            top1_name.append("name"+str(i+1))
            top1_description.append("description"+str(i+1))
            
        return render_template("website_futuretrends.html",top1_name=top1_name,top1_description=top1_description)
    
    
    
        
@app.route('/home', methods=['GET'])
def api_homepage():
    return render_template("website.html")



@app.route('/aboutus', methods=['GET'])
def api_aboutus():
    return render_template("website_aboutus.html")



@app.route('/downloadcode', methods=['GET'])
def api_downloadcode():
    return render_template("website_downloadcode.html")



@app.route('/contactus', methods=['GET'])
def api_contactus():
    return render_template("website_contactus.html")



app.run()









