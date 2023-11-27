import os
import base64
import numpy as np
import cv2
from flask import Flask, render_template, request, jsonify

# Load the pre-trained MNIST model
from keras.models import load_model

# load mode mnist_model.h5
model = load_model('TA.model.h5')

app = Flask(__name__)

# Counter for naming images
image_counter = 0

@app.route('/')
def index():
    return render_template('index_v5.html')

@app.route('/predict_digit', methods=['POST'])
def predict_digit():
    global image_counter
    data_url = request.form['data_url']  # Dữ liệu vẽ từ Canvas dưới dạng URL
    image_data = data_url.split(',')[1]  # Lấy dữ liệu hình ảnh từ URL
    binary_data = base64.b64decode(image_data)  # Chuyển đổi dữ liệu thành dạng nhị phân

    # Lưu hình ảnh vào tệp ảnh tạm thời
    temp_image_path = 'img_400x400.png'
    with open(temp_image_path, 'wb') as f:
        f.write(binary_data)


    # Đọc và xử lý hình ảnh sử dụng OpenCV
    img = cv2.imread(temp_image_path, cv2.IMREAD_GRAYSCALE)

    # resize ảnh --> 28x28
    img =cv2.resize(img, (28, 28))

    # # lưu ảnh để kiểm tra 
    # cv2.imwrite(f'data_img/Number_0/img_28x28_1_{100 + image_counter}.jpg', img)

    # reshape ảnh 
    img = img.reshape(1, 28, 28, 1)

    # Normalize ảnh 
    img = img.astype('float32') / 255.0
   

    # Dự đoán số
    prediction = model.predict(img)
    predicted_digit = np.argmax(prediction)

    # Xóa tệp ảnh tạm thời
    os.remove(temp_image_path)

    image_counter += 1
    # Hiển thị 
    return jsonify({"predicted_digit": str(predicted_digit)})

if __name__ == '__main__':
    app.run(debug=True)
