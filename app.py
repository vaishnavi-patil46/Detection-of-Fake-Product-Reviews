from flask import Flask, render_template, request, url_for, Markup, jsonify
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)
#loadpickle = open('fake_review_detection_GaussianNBmodel.pkl','rb')
pickle_in = open('review.pkl','rb')
pac = pickle.load(pickle_in)
tfid = open('tfid_review.pkl','rb')
tfidf_vectorizer = pickle.load(tfid)

@app.route('/')
@app.route('/first')
def first():
    return render_template('first.html')

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/upload')
def upload():
    return render_template('upload.html')
@app.route('/preview',methods=["POST"])
def preview():
    if request.method == 'POST':
        dataset = request.files['datasetfile']
        df = pd.read_csv(dataset,encoding = 'unicode_escape')
        #df.set_index('Id', inplace=True)
        return render_template("preview.html",df_view = df)

@app.route('/index')
def index():
 	return render_template("index.html")

@app.route('/predict')
def predict():
	abc = request.args.get('news')
	input_data = [abc.rstrip()]
	# transforming input
	tfidf_test = tfidf_vectorizer.transform(input_data)
	 #predicting the input
	y_pred = pac.predict(tfidf_test)
    #output=y_pred[0]
	return render_template('index.html', prediction_text='Review is {}'.format(y_pred[0]))

@app.route('/chart')
def chart():
		return render_template("chart.html")
    #abc = request.args.get('news')
	#input_data = [abc.rstrip()]
	# transforming input
	#tfidf_test = tfidf_vectorizer.transform(input_data)
	# predicting the input
	#y_pred = pac.predict(tfidf_test)
	#if y_pred[0] == 1:
	#   label=1
	#elif y_pred[0] == 0:
	 #   label=0
	#return render_template('chart.html', prediction_text=label)


if __name__=='__main__':
    app.run(debug=True)
