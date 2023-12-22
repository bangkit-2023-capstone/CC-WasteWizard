from flask import Flask,jsonify,request
from tensorflow import keras
import numpy as np
import tensorflow 
import base64
from PIL import Image
from io import BytesIO
import os

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
app = Flask(__name__)

# Load Model dengan flask
label = ['Organic','Recyclable']

model = keras.models.load_model('model_v1.h5')

def prediksiii(gambar): 
    Loaded=np.asarray(gambar)/255.0
    Loaded = Loaded.reshape(1, 150,150,3)
    preddd= model.predict(Loaded)
    prediksi= label[np.argmax(preddd)]
    return prediksi
        
@app.route('/Prediksi', methods=['POST'])
def prediksi():
    if request.method == 'POST':
        data = request.files.get('file')
        if data is None or data.filename == "":
               return jsonify({
                     "error":"Tidak ada file"
               })
        img_binary=data.read()
        image = Image.open(BytesIO(img_binary))
        image = image.convert("RGB")
		       
        image= image.resize((150,150))
		# Panggil method prediksi CNN
        Hasilprediksi = prediksiii(image)
		# Kirim hasil prediksi dalam bentuk JSON
        return jsonify(
		 {
		  'data': 'success',
	      'Klasifikasi':Hasilprediksi
		}
		),200
    else:
        return jsonify(
			    {
				  'error':"Method tidak mendukung"
			     }
			      ),404
@app.route('/')
def rera():
       return jsonify(
			{
			'Api':"Ok"
		  }
		),200              
if __name__ =='__main__':
	#app.debug = True
	app.run(debug=True)