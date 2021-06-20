from flask import Flask,render_template,request
import joblib
import numpy as np

#app= Flask(__name__)
app = Flask(__name__, template_folder='templates')  # still relative to module

#load model
model=joblib.load('hiring_model.pkl')

@app.route('/')
def hello_world():
    return render_template('base.html')

@app.route('/predict',methods=['POST'])
def predict():
    exp=request.form.get('experience')
    score=request.form.get('test_score')
    interview_score=request.form.get('interview_score')
    prediction=model.predict([[int(exp),int(score),int(interview_score)]])
    print(prediction)
    output=round(prediction[0],2)

    return render_template('base.html',prediction_txt=f"employee salary will be $ {output}")




@app.route('/feedback')
def feedback():
    return 'welcome to feedback page'

@app.route('/help')
def help():
    return 'welcome to help page'

@app.route('/datascience')
def datascience():
    return 'welcome to data science page'


app.run(debug=True)