import flask
from flask import request, jsonify
from flask import Flask, send_file, render_template
	
from flask import Flask, flash, redirect, render_template, request, session, abort,send_from_directory,send_file,jsonify
import pandas as pd

import json

app = flask.Flask(__name__)
app.config["DEBUG"] = True

    
    
@app.route('/')
def home(): 
    return render_template("website.html")  



@app.route('/currenttrends', methods=['GET'])
def api_currenttrends():
    if 'tshirts' in request.args:
#        Insert code here/ information to retrieve
        return render_template("website_currenttrends.html")
    if 'dresses' in request.args:
#        Insert code here/ information to retrieve
        return render_template("website_currenttrends.html")
    if 'skirts' in request.args:
#        Insert code here/ information to retrieve
        return render_template("website_currenttrends.html")
    
    
    
@app.route('/futuretrends', methods=['GET'])
def api_futuretrends():
    if 'tshirts' in request.args:
#        Insert code here/ information to retrieve
        return render_template("website_futuretrends.html")
    if 'dresses' in request.args:
#        Insert code here/ information to retrieve
        return render_template("website_futuretrends.html")
    if 'skirts' in request.args:
#        Insert code here/ information to retrieve
        return render_template("website_futuretrends.html")
    
    
    
        
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









