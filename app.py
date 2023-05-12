from flask import Flask,request,render_template,jsonify
from src.pipeline.prediction_pipeline import CustomData,PredictPipeline
from src.logger import logging


application=Flask(__name__)

app=application



@app.route('/')
def home_page():
    return render_template('index.html')

@app.route('/predict1',methods=['GET','POST'])

def predict_datapoint():
    if request.method=='GET':
        return render_template('form.html')
    
    else:
        data=CustomData(
                Customer_ID= int(request.form.get('Customer_ID')),
                X1=int(request.form.get('X1')), X2=int(request.form.get('X2')), X3=int(request.form.get('X3')), 
                X4=int(request.form.get('X4')), X5=int(request.form.get('X5')), X6=int(request.form.get('X6')), 
                X7=int(request.form.get('X7')),X8=int(request.form.get('X8')), X9=int(request.form.get('X9')),
                X10=int(request.form.get('X10')), X11=int(request.form.get('X11')),X12=int(request.form.get('X12')), 
                X13=int(request.form.get('X13')), X14=int(request.form.get('X14')), X15=int(request.form.get('X15')), 
                X16=int(request.form.get('X16')),X17=int(request.form.get('X17')) , X18=int(request.form.get('X18')), 
                X19=int(request.form.get('X19')),X20=int(request.form.get('X20')), X21=int(request.form.get('X21')), 
                X22=int(request.form.get('X22')),X23=int(request.form.get('X23'))

        )
        final_new_data=data.get_data_as_dataframe()
        predict_pipeline=PredictPipeline()
        logging.info(f'test data:\n{final_new_data.head()}')
        pred=predict_pipeline.predict(final_new_data)
        
        results=round(pred[0],2)

        return render_template('result.html',final_result=results)






if __name__=="__main__":
    app.run(host='0.0.0.0',debug=True)

