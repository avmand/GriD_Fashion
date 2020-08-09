from csv import reader
import csv
import pandas as pd
import math

read_objfiles=['CurrentTrends/final_csv/skirts/skirt_top_bottom.csv']
dataset_csv_files='CurrentTrends/final_csv/skirts/skirts_csv_final.csv'
colnames=['sno','URL','id','desc','stars','num_ratings','num_reviews','reviews','vader_score','final_score']
reqdcolnames=['id','stars','num_ratings']
write_objfiles=['skirt_top_bottom.csv']




for ite in range(0, len(write_objfiles)):
    with open(read_objfiles[ite], 'r') as read_obj:
        dataset_csv = pd.read_csv(dataset_csv_files,names=colnames, delimiter=',', error_bad_lines=False, header=None,usecols=reqdcolnames, na_values=" NaN")
        csv_reader = reader(read_obj)
        newfile=[]
        newfile.append(('Rating','Bigram','Count'))
        for row in csv_reader:
            for i in range(1, len(row)):
                for x in range(1,len(dataset_csv)):
                    if row[i] == dataset_csv['id'][x] :
                        stars=math.floor(float(dataset_csv['stars'][x]))
                        if stars !=5:
                            starsw='Rating '+str(stars)+' to '+str(stars+1)
                        else:
                            starsw='Rating '+str(stars-1)+' to '+str(stars)
                        newfile.append((starsw,row[0],"1"))
                        break
    with open(write_objfiles[ite], 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(newfile)
 
