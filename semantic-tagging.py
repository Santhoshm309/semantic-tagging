import pandas as pd
import nltk
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.externals import joblib
import argparse
import sys
import numpy as np
import json
from sklearn.pipeline import Pipeline
stopwords = nltk.corpus.stopwords.words('english')
stemmer = nltk.stem.snowball.SnowballStemmer('english')
base_uri = 'https://pixabay.com/api/?key=8583084-7c7aca42f03797bf824268233&image_type=photo&q='
from PIL import Image
import requests
from io import BytesIO



def parse_arguments():
	
	ap = argparse.ArgumentParser()

	ap.add_argument("-t","--train", required='--cluster' in sys.argv, help="Input file to train the model..... \n", nargs='?')
	ap.add_argument("-f", "--file", required='--train' not in sys.argv, help="Input file for testing.... \n")
	ap.add_argument("-n","--cluster", required=False, default=5, help="Number of clusters the model needs to train with....\n")


	args = vars(ap.parse_args())	
	return args


def load_data(fileName):
	data = pd.read_csv(fileName)
	'''
		Check to see if there are null values in the dataframe columns....

	'''
	# print(data.isnull().sum())

	''' Summary column has 27 null values ........................................'''


	# Handle NaN values ??????



	return data['Summary'].tolist(), data['Text'].tolist()



def tokenize(text):

	# Basic punctuators ..........................

	stopwords.extend([',','.','?','!',';','-','_','\'', 'br','=','\\','href','\'s','n\'t'])




	# Tokenize without any filter.................
	
	tokens = [ word	for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent) \
	
	# Stop word filter

	if word.lower() not in stopwords ] 

	tokens = [ token for token in tokens if re.search('[a-zA-Z]+',token)]

	#Remove anything but alphabets ...............................................

	

	return tokens




def build_dictionary(review_list, cluster_count):
	print("Extracting tokens \n")
	all_tokens = []
	for review in review_list:
		tokens = tokenize(review)
		all_tokens.extend(tokens)


	#	Create a pandas dataframe from the tokens ...................................

	df = pd.DataFrame({'words': all_tokens}, index=all_tokens)
	
	# Remove duplicate values .......................................................
	print("Droping duplicate values \n")
	df = df.drop_duplicates(subset='words')

	joblib.dump(df,'df.pkl')
	# Create a tf-idf vector ........................................................
	print("Creating TF-IDF vector \n")
	tfidf_vectorizer = TfidfVectorizer( max_features=3000,
                                 stop_words='english',
                                 use_idf=True, tokenizer=tokenize, ngram_range=(1,3))
	
	tfidf_matrix = tfidf_vectorizer.fit_transform(review_list)

	# A local dictionary that contains the terms that were converted to vector ..........................


	terms = tfidf_vectorizer.get_feature_names()

	# Store the dictionary in byte-format (pickle)..................................................


	joblib.dump(terms,'terms.pkl')

	
	# Start clustering ..........................................................................
	print("Creating our model \n")
	km = KMeans(n_clusters=cluster_count)

	# Fit our model for our tf-idf vector........................................................

	km.fit(tfidf_matrix)

	# Pickle the model file for training purpose .................................................
	tfidf_km = Pipeline([('tfidf', tfidf_vectorizer), ('km', km)])
	tfidf_km.fit(review_list)

	joblib.dump(tfidf_km,'model.pkl')
	joblib.dump(km,'km.pkl')


	order_centroids = km.cluster_centers_.argsort()[:, ::-1]

	for i in range(cluster_count):
	    print("Cluster %d words:" % i, end='')
	    
	    for ind in order_centroids[i, :6]: #replace 6 with n words per cluster
	        print(' %s' % df.ix[terms[ind].split(' ')].values.tolist()[0][0], end=',')
	    print() 
	    print() 
	    
	return


def get_model_dict():
	try:
		print("Loading model files \n")
		model = joblib.load('model.pkl')
		terms = joblib.load('terms.pkl')
		km = joblib.load('km.pkl')
		df = joblib.load('df.pkl')
	except Exception as e:
		print(e + " train model before testing \n")

	return model,km, terms, df



def get_images(tags):
	for tag in tags:
		url = base_uri+tag
		response = requests.get(url)
		hit = json.loads(str(response.content, encoding='UTF-8'))['hits'][0]
		res = requests.get(hit['webformatURL'])
		img = Image.open(BytesIO(res.content))
		img.show()


	
def predict(inputText):
	i = []
	tags = []
	i.append(inputText)
	model,km, terms, df = get_model_dict()

	# Predict cluster number .........................................


	associated_clusters = model.predict(i)
	

	order_centroids = km.cluster_centers_.argsort()[:, ::-1] 
	
	for a in associated_clusters:
		for ind in order_centroids[a, :6]: #replace 6 with n words per cluster
			tags.append(df.ix[terms[ind].split(' ')].values.tolist()[0][0])
	
	print("Suggested tags are : ")
	print(tags)

	print("Getting images for the tags \n")
	get_images(tags)






if __name__ == '__main__':
	
	# df = build_dictionary(review[0:4000])

	args = parse_arguments();
	if args["train"]:
		summary, review =load_data(args["train"])
		cluser_count = args["cluster"]
		build_dictionary(review[0:5000],int(cluser_count))	
	
	else:
		 predict(args["file"])
	
