import flask
from flask import request, jsonify
from flask import Flask, send_file, render_template
	
from flask import Flask, flash, redirect, render_template, request, session, abort,send_from_directory,send_file,jsonify
import pandas as pd
import glob
import random
import json

app = flask.Flask(__name__)
app.config["DEBUG"] = True

#2. Declare data stores
class DataStore():
    Prod= None
    Prod2=None
    Prod3=None
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
        
        images_path=[]
        imagestshirts = glob.glob('static/img/current_trends/shirt+' + '*.jpeg')
        for i in range(0,len(imagestshirts)):
            images_path.append(imagestshirts[i])
        imagestshirts = glob.glob('static/img/current_trends/shirt-' + '*.jpeg')
        for i in range(0,len(imagestshirts)):
            images_path.append(imagestshirts[i])
        
        colnames=['sno','URL','id','desc','stars','num_ratings','num_reviews','reviews','vader_score','final_score']
        reqdcolnames=['id','stars','desc','URL','final_score']  
        dataset_csv = pd.read_csv('CurrentTrends/final_csv/tshirts/tshirts_csv_final.csv',names=colnames, delimiter=',', error_bad_lines=False, 
                                  header=None,usecols=reqdcolnames, na_values=" NaN")
        dataset_csv = dataset_csv.dropna()
        dataset_csv2=dataset_csv.sort_values(by='final_score', ascending=False)
        dataset_csv2=dataset_csv2.reset_index()
        #print(dataset_csv2.head())
        for i in range(1, 6):
            top1_name.append(dataset_csv2['desc'][i])
            top1_description.append(dataset_csv2['URL'][i])
            top1_score.append(dataset_csv2['final_score'][i])
        for i in range((len(dataset_csv2)-5),len(dataset_csv2)):
            bottom1_name.append(dataset_csv2['desc'][i])
            bottom1_description.append(dataset_csv2['URL'][i])
            bottom1_score.append(dataset_csv2['final_score'][i])
            
        
        df = pd.read_csv('CurrentTrends/Leaderboard/tshirt_colour_top_bottom.csv')
        #print(df.head())
        df = df[["Bigram", "Rating", "Count"]]
        
        # order in the groupby here matters, it determines the json nesting
        # the groupby call makes a pandas series by grouping 'the_parent' and 'the_child', while summing the numerical column 'child_size'
        df1 = df.groupby(['Bigram', 'Rating'])['Count'].sum()
        df1 = df1.reset_index()
        
        
        # start a new flare.json document
        flare = dict()
        d = {"name": "flare", "children": []}
        i=0
        for line in df1.values:
            Bigram = line[0]
            Rating = line[1]
            Count = line[2]
            # make a list of keys
            keys_list = []
            for item in d['children']:
                keys_list.append(item['name'])
    
            # if 'the_parent' is NOT a key in the flare.json yet, append it
            if not Bigram in keys_list:
                d['children'].append({"name": Bigram, "children": [{"name": Rating, "size": Count}]})
                
    
            # if 'the_parent' IS a key in the flare.json, add a new child to it
            else:
                d['children'][keys_list.index(Bigram)]['children'].append({"name": Rating, "size": Count})
                
    
        flare = d
        e = json.dumps(flare)
        data.Prod = json.loads(e)
        Prod=data.Prod
        
        
        
        df = pd.read_csv('CurrentTrends/Leaderboard/tshirt_neck_top_bottom.csv')
        #print(df.head())
        df = df[["Bigram", "Rating", "Count"]]
    
        # order in the groupby here matters, it determines the json nesting
        # the groupby call makes a pandas series by grouping 'the_parent' and 'the_child', while summing the numerical column 'child_size'
        df1 = df.groupby(['Bigram', 'Rating'])['Count'].sum()
        df1 = df1.reset_index()
    
        # start a new flare.json document
        flare = dict()
        d = {"name": "flare", "children": []}
    
        for line in df1.values:
            Bigram = line[0]
            Rating = line[1]
            Count = line[2]
    
            # make a list of keys
            keys_list = []
            for item in d['children']:
                keys_list.append(item['name'])
    
            # if 'the_parent' is NOT a key in the flare.json yet, append it
            if not Bigram in keys_list:
                d['children'].append({"name": Bigram, "children": [{"name": Rating, "size": Count}]})
    
            # if 'the_parent' IS a key in the flare.json, add a new child to it
            else:
                d['children'][keys_list.index(Bigram)]['children'].append({"name": Rating, "size": Count})
    
        flare = d
        e = json.dumps(flare)
        data.Prod2 = json.loads(e)
        Prod2=data.Prod2
        
        
        
        df = pd.read_csv('CurrentTrends/Leaderboard/tshirt_print_top_bottom.csv')
        #print(df.head())
        df = df[["Bigram", "Rating", "Count"]]
    
        # order in the groupby here matters, it determines the json nesting
        # the groupby call makes a pandas series by grouping 'the_parent' and 'the_child', while summing the numerical column 'child_size'
        df1 = df.groupby(['Bigram', 'Rating'])['Count'].sum()
        df1 = df1.reset_index()
    
        # start a new flare.json document
        flare = dict()
        d = {"name": "flare", "children": []}
    
        for line in df1.values:
            Bigram = line[0]
            Rating = line[1]
            Count = line[2]
    
            # make a list of keys
            keys_list = []
            for item in d['children']:
                keys_list.append(item['name'])
    
            # if 'the_parent' is NOT a key in the flare.json yet, append it
            if not Bigram in keys_list:
                d['children'].append({"name": Bigram, "children": [{"name": Rating, "size": Count}]})
    
            # if 'the_parent' IS a key in the flare.json, add a new child to it
            else:
                d['children'][keys_list.index(Bigram)]['children'].append({"name": Rating, "size": Count})
    
        flare = d
        e = json.dumps(flare)
        data.Prod3 = json.loads(e)
        Prod3=data.Prod3
        
        
        
        
        
        return render_template("website_currenttrends_tshirts.html",images_path=images_path,Prod=Prod, Prod2=Prod2,Prod3=Prod3, top1_name=top1_name,top1_description=top1_description,top1_score=top1_score,bottom1_name=bottom1_name,bottom1_description=bottom1_description,bottom1_score=bottom1_score)
    if 'dresses' in request.args:
#        Insert code here/ information to retrieve
        top1_name=[]
        top1_description=[]
        top1_score=[]
        bottom1_name=[]
        bottom1_description=[]
        bottom1_score=[]
        
        images_path=[]
        imagestshirts = glob.glob('static/img/current_trends/dress+' + '*.jpeg')
        for i in range(0,len(imagestshirts)):
            images_path.append(imagestshirts[i])
        imagestshirts = glob.glob('static/img/current_trends/dress-' + '*.jpeg')
        for i in range(0,len(imagestshirts)):
            images_path.append(imagestshirts[i])
        
        colnames=['sno','URL','id','desc','stars','num_ratings','num_reviews','reviews','vader_score','final_score']
        reqdcolnames=['id','stars','desc','URL','final_score']  
        dataset_csv = pd.read_csv('CurrentTrends/final_csv/dresses/dresses_csv_final.csv',names=colnames, delimiter=',', error_bad_lines=False, 
                                  header=None,usecols=reqdcolnames, na_values=" NaN")
        dataset_csv = dataset_csv.dropna()
        dataset_csv2=dataset_csv.sort_values(by='final_score', ascending=False)
        dataset_csv2=dataset_csv2.reset_index()
        #print(dataset_csv2.head())
        for i in range(1, 6):
            top1_name.append(dataset_csv2['desc'][i])
            top1_description.append(dataset_csv2['URL'][i])
            top1_score.append(dataset_csv2['final_score'][i])
        for i in range((len(dataset_csv2)-5),len(dataset_csv2)):
            bottom1_name.append(dataset_csv2['desc'][i])
            bottom1_description.append(dataset_csv2['URL'][i])
            bottom1_score.append(dataset_csv2['final_score'][i])
        
        df = pd.read_csv('CurrentTrends/Leaderboard/dress_top_bottom.csv')
        #print(df.head())
        df = df[["Bigram", "Rating", "Count"]]
    
        # order in the groupby here matters, it determines the json nesting
        # the groupby call makes a pandas series by grouping 'the_parent' and 'the_child', while summing the numerical column 'child_size'
        df1 = df.groupby(['Bigram', 'Rating'])['Count'].sum()
        df1 = df1.reset_index()
    
        # start a new flare.json document
        flare = dict()
        d = {"name": "flare", "children": []}
    
        for line in df1.values:
            Bigram = line[0]
            Rating = line[1]
            Count = line[2]
    
            # make a list of keys
            keys_list = []
            for item in d['children']:
                keys_list.append(item['name'])
    
            # if 'the_parent' is NOT a key in the flare.json yet, append it
            if not Bigram in keys_list:
                d['children'].append({"name": Bigram, "children": [{"name": Rating, "size": Count}]})
    
            # if 'the_parent' IS a key in the flare.json, add a new child to it
            else:
                d['children'][keys_list.index(Bigram)]['children'].append({"name": Rating, "size": Count})
    
        flare = d
        e = json.dumps(flare)
        data.Prod = json.loads(e)
        Prod=data.Prod
        return render_template("website_currenttrends.html",images_path=images_path,Prod=Prod, top1_name=top1_name,top1_description=top1_description,top1_score=top1_score,bottom1_name=bottom1_name,bottom1_description=bottom1_description,bottom1_score=bottom1_score)
    if 'skirts' in request.args:
#        Insert code here/ information to retrieve
        top1_name=[]
        top1_description=[]
        top1_score=[]
        bottom1_name=[]
        bottom1_description=[]
        bottom1_score=[]
        
        images_path=[]
        imagestshirts = glob.glob('static/img/current_trends/skirt+' + '*.jpeg')
        for i in range(0,len(imagestshirts)):
            images_path.append(imagestshirts[i])
        imagestshirts = glob.glob('static/img/current_trends/skirt-' + '*.jpeg')
        for i in range(0,len(imagestshirts)):
            images_path.append(imagestshirts[i])
        
        colnames=['sno','URL','id','desc','stars','num_ratings','num_reviews','reviews','vader_score','final_score']
        reqdcolnames=['id','stars','desc','URL','final_score']  
        dataset_csv = pd.read_csv('CurrentTrends/final_csv/skirts/skirts_csv_final.csv',names=colnames, delimiter=',', error_bad_lines=False, 
                                  header=None,usecols=reqdcolnames, na_values=" NaN")
        dataset_csv = dataset_csv.dropna()
        dataset_csv2=dataset_csv.sort_values(by='final_score', ascending=False)
        dataset_csv2=dataset_csv2.reset_index()
        #print(dataset_csv2.head())
        for i in range(1, 6):
            top1_name.append(dataset_csv2['desc'][i])
            top1_description.append(dataset_csv2['URL'][i])
            top1_score.append(dataset_csv2['final_score'][i])
        for i in range((len(dataset_csv2)-5),len(dataset_csv2)):
            bottom1_name.append(dataset_csv2['desc'][i])
            bottom1_description.append(dataset_csv2['URL'][i])
            bottom1_score.append(dataset_csv2['final_score'][i])
        
        df = pd.read_csv('CurrentTrends/Leaderboard/skirt_top_bottom.csv')
        #print(df.head())
        df = df[["Bigram", "Rating", "Count"]]
    
        # order in the groupby here matters, it determines the json nesting
        # the groupby call makes a pandas series by grouping 'the_parent' and 'the_child', while summing the numerical column 'child_size'
        df1 = df.groupby(['Bigram', 'Rating'])['Count'].sum()
        df1 = df1.reset_index()
    
        # start a new flare.json document
        flare = dict()
        d = {"name": "flare", "children": []}
    
        for line in df1.values:
            Bigram = line[0]
            Rating = line[1]
            Count = line[2]
    
            # make a list of keys
            keys_list = []
            for item in d['children']:
                keys_list.append(item['name'])
    
            # if 'the_parent' is NOT a key in the flare.json yet, append it
            if not Bigram in keys_list:
                d['children'].append({"name": Bigram, "children": [{"name": Rating, "size": Count}]})
    
            # if 'the_parent' IS a key in the flare.json, add a new child to it
            else:
                d['children'][keys_list.index(Bigram)]['children'].append({"name": Rating, "size": Count})
    
        flare = d
        e = json.dumps(flare)
        data.Prod = json.loads(e)
        Prod=data.Prod
        return render_template("website_currenttrends.html",images_path=images_path,Prod=Prod, top1_name=top1_name,top1_description=top1_description,top1_score=top1_score,bottom1_name=bottom1_name,bottom1_description=bottom1_description,bottom1_score=bottom1_score)
    


@app.route("/currenttrendsget-data",methods=["GET","POST"])
def returnProdDatacurrenttrends():
    f=data.Prod

    return jsonify(f)


@app.route("/currenttrendsgetdata2",methods=["GET","POST"])
def returnProdDatacurrenttrends2():
    f=data.Prod2

    return jsonify(f)


@app.route("/currenttrendsgetdata3",methods=["GET","POST"])
def returnProdDatacurrenttrends3():
    f=data.Prod3

    return jsonify(f)
    
    
@app.route('/futuretrends', methods=['GET'])
def api_futuretrends():
    
    if 'tshirts' in request.args:
#        Insert code here/ information to retrieve
        top1_name=[]
        top1_description=[]
        for i in range(0,10):
            top1_name.append("name"+str(i+1))
            top1_description.append("description"+str(i+1))
        imagestshirts = glob.glob('static/img/tshirt_images/generated' + '*.png')
        imgs_id=random.sample(range(0, len(imagestshirts)), 10)
        images_path=[]
        for i in range(0,len(imgs_id)):
            images_path.append(imagestshirts[imgs_id[i]])
            

        imagestshirts2 = glob.glob('static/img/content/outputs/output_shirt-' + '*.*')
        images_path_insp=[]
        for i in range(0,len(imagestshirts2)):
            images_path_insp.append(imagestshirts2[i])
#        print(images_path)
        return render_template("website_futuretrends.html",images_path_insp=images_path_insp,images_path=images_path,top1_name=top1_name,top1_description=top1_description)
    
    if 'dresses' in request.args:
#        Insert code here/ information to retrieve
        top1_name=[]
        top1_description=[]
        for i in range(0,10):
            top1_name.append("name"+str(i+1))
            top1_description.append("description"+str(i+1))
        imagesdresses = glob.glob('static/img/dress_images/generated' + '*.png')
        imgs_id=random.sample(range(0, len(imagesdresses)), 10)
        images_path=[]
        for i in range(0,len(imgs_id)):
            images_path.append(imagesdresses[imgs_id[i]])
            
        imagestshirts2 = glob.glob('static/img/content/outputs/output_dress-' + '*.*')
        images_path_insp=[]
        for i in range(0,len(imagestshirts2)):
            images_path_insp.append(imagestshirts2[i])
#        print(images_path)
        return render_template("website_futuretrends.html",images_path_insp=images_path_insp,images_path=images_path,top1_name=top1_name,top1_description=top1_description)
    
    if 'skirts' in request.args:
#        Insert code here/ information to retrieve
        top1_name=[]
        top1_description=[]
        for i in range(0,10):
            top1_name.append("name"+str(i+1))
            top1_description.append("description"+str(i+1))
        imagesskirts = glob.glob('static/img/skirt_images/generated' + '*.png')
        imgs_id=random.sample(range(0, len(imagesskirts)), 10)
        images_path=[]
        for i in range(0,len(imgs_id)):
            images_path.append(imagesskirts[imgs_id[i]])
            
        imagestshirts2 = glob.glob('static/img/content/outputs/output_skirt-' + '*.*')
        images_path_insp=[]
        for i in range(0,len(imagestshirts2)):
            images_path_insp.append(imagestshirts2[i])
#        print(images_path)
        return render_template("website_futuretrends.html",images_path_insp=images_path_insp,images_path=images_path,top1_name=top1_name,top1_description=top1_description)
    
    
    
        
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









