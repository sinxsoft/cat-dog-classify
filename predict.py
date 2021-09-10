from werkzeug.utils import secure_filename
from flask import Flask, request, redirect, url_for,render_template
from io import BytesIO
from io import BufferedReader
import numpy as np
from PIL import Image
from flask import Flask
from skimage import io
import sys, os, json, time
import tensorflow as tf
start = time.time()
from keras.models import model_from_json
print("import keras time = ", time.time()-start)

TEMPLATE = open('./index.html').read()
ALLOWED_EXTENSIONS = set(['jpg'])
app = Flask(__name__)

base_path = ''
abspath = os.getcwd()
model = None
model_path = abspath + "/model"
model_path_json = model_path + '/model.json'
model_path_weights = model_path + '/weights.h5'
graph = None

# with open(model_path_json, 'r') as f:
#     model_content = f.read()
#     model = model_from_json(model_content)
#     # Getting weights
#     model.load_weights(model_path_weights)
#     graph = tf.get_default_graph()
# def allowed_file(filename):
#     return '.' in filename and \
#            filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
# 在下方手动输入上方代码，体验智能的代码补全功能


with open(model_path_json, 'r') as f:
    model_content = f.read()
    model = model_from_json(model_content)
    # Getting weights
    model.load_weights(model_path_weights)
    graph = tf.get_default_graph()


def allowed_file(filename):
    return True

@app.route('/', methods=['GET', 'POST'])
def home():
    return TEMPLATE.replace('{fc-result}', "请输入图片")

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    imgData = request.files["imagefile"]
    if imgData.filename is "":
        return TEMPLATE.replace('{fc-result}', "没有图片，请上传")
    filename= imgData.filename
    pic_suffix_list = [".jpg", ".png", ".jpeg"]
    pic_flag = False
    for sufix  in pic_suffix_list:
        if filename.rfind(sufix) !=-1:
            pic_flag = True
            break
    if pic_flag is False:
        return TEMPLATE.replace('{fc-result}', "请上传jpg，png,jpeg格式的图片")
    img_buff = BufferedReader(imgData)
    img_byte = BufferedReader.read(img_buff)

    final_result = predict(img_byte, "")
    return TEMPLATE.replace('{fc-result}', final_result)

def predict(event, context):
    start = time.time()
    img_size = 64
    img_buffer = np.asarray(bytearray(event), dtype='uint8')
    img = io.imread(BytesIO(img_buffer))

    img = np.array(Image.fromarray(img).convert("RGB").resize((img_size, img_size)))
    X = np.zeros((1, 64, 64, 3), dtype='float64')
    X[0] = img

    global model, graph
    if graph is None or model is None:
        return "请按照README的第三步提示，输入代码，补全模型加载"
    with graph.as_default():
        Y = model.predict(X)
    result = "狗: {:.2}, 猫: {:.2}; ".format(Y[0][1], Y[0][0])
    Y = np.argmax(Y, axis=1)
    Y = '猫' if Y[0] == 0 else '狗'
    print("predict time = {}".format(time.time()-start))
    return result + '\n这张图片是 ' + Y + ' !'

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=8080)
