from flask import Flask,render_template,request,redirect,url_for,session
import pickle
import numpy as np

from processing import dataPreprocessing

app = Flask(__name__)
app.secret_key = 'test2020205434'
app.config['PERMANENT_SESSION_LIFETIME'] = 10
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
@app.route('/', methods= ['POST','GET'])
def index():  # put application's code here
    if request.method == 'GET':
        return render_template("index.html")
    else:
        message = request.form['message']
        X = dataPreprocessing(message)
        print(X)
        result =np.array(model.predict(X))
        session['message'] = message
        if result.argmax() == 1:
            session['result'] = "We suspect that the present communication may be a spam."
            session['style']  = "red"
        else:
            session['result'] = "We suspect that the present communiation may be not a spam"
            session['style'] = "green"

        return render_template("index.html")

if __name__ == '__main__':
    app.run(debug=True)
