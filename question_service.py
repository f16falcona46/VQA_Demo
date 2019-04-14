import os, sys, argparse
import cv2, spacy, numpy as np
from keras.models import model_from_json
from keras.optimizers import SGD
from sklearn.externals import joblib
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras import backend as K
K.set_image_data_format('channels_first')
K.set_image_dim_ordering('th')

from openvino.inference_engine import IENetwork, IEPlugin


import wsgiref.simple_server
from urllib.parse import parse_qs

   
# File paths for the model, all of these except the CNN Weights are 
# provided in the repo, See the models/CNN/README.md to download VGG weights
VQA_weights_file_name   = 'models/VQA/VQA_MODEL_WEIGHTS.hdf5'
classes_file_name       = 'classes.txt'
CNN_ov_file_name        = "models/CNN.xml"
device                  = "CPU"
cpu_extensions          = ["/opt/intel/openvino_2019.1.094/deployment_tools/inference_engine/lib/intel64/libcpu_extension_avx2.so",
				"/home/intel/inference_engine_samples_build/intel64/Release/lib/libcpu_extension.so"]

# Chagne the value of verbose to 0 to avoid printing the progress statements
verbose = 1

def get_image_features(image_file_name, CNN_exec_net, input_blob, output_blob):
    ''' Runs the given image_file to VGG 16 model and returns the 
    weights (filters) as a 1, 4096 dimension vector '''
    req = 0

    image_features = np.zeros((1, 4096))
    # Magic_Number = 4096  > Comes from last layer of VGG Model

    # Since VGG was trained as a image of 224x224, every new image
    # is required to go through the same transformation
    im = cv2.resize(cv2.imread(image_file_name), (224, 224))


    # The mean pixel values are taken from the VGG authors, which are the values computed from the training dataset.
    mean_pixel = [103.939, 116.779, 123.68]

    im = im.astype(np.float32, copy=False)
    for c in range(3):
        im[:, :, c] = im[:, :, c] - mean_pixel[c]

    #im = im.transpose((2,0,1)) # convert the image to RGBA

    
    # this axis dimension is required becuase VGG was trained on a dimension
    # of 1, 3, 224, 224 (first axis is for the batch size
    # even though we are using only one image, we have to keep the dimensions consistent
    im = np.expand_dims(im, axis=0).transpose((0, 2, 3, 1))

    CNN_exec_net.start_async(request_id=req, inputs={input_blob: im})
    CNN_exec_net.requests[req].wait(-1)
    a = CNN_exec_net.requests[req].outputs[output_blob]
    image_features[0,:] = a
    return image_features

def get_VQA_model(VQA_weights_file_name):
    ''' Given the VQA model and its weights, compiles and returns the model '''

    from models.VQA.VQA import VQA_MODEL
    vqa_model = VQA_MODEL()
    vqa_model.load_weights(VQA_weights_file_name)

    vqa_model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    return vqa_model

def get_question_features(question):
    ''' For a given question, a unicode string, returns the timeseris vector
    with each word (token) transformed into a 300 dimension representation
    calculated using Glove Vector '''
    # word_embeddings = spacy.load('en', vectors='en_glove_cc_300_1m_vectors')
    word_embeddings = spacy.load('en_vectors_web_lg')
    tokens = word_embeddings(question)
    question_tensor = np.zeros((1, 30, 300))
    for j in range(len(tokens)):
            question_tensor[0,j,:] = tokens[j].vector
    return question_tensor

def init_openvino():
    CNN_xml = CNN_ov_file_name
    CNN_bin = os.path.splitext(CNN_xml)[0] + ".bin"
    plugin = IEPlugin(device=device)
    if plugin.device == "CPU":
        for ext in cpu_extensions:
            plugin.add_cpu_extension(ext)
    CNN_net = IENetwork(model=CNN_xml, weights=CNN_bin)

    if plugin.device == "CPU":
        supported_layers = plugin.get_supported_layers(CNN_net)
        layers = CNN_net.layers.keys()
        not_supported_layers = [l for l in layers if l not in supported_layers]
        if len(not_supported_layers) != 0:
            print("Following layers are not supported by the plugin for specified device {}:\n {}".
                      format(plugin.device, ', '.join(not_supported_layers)), file=sys.stderr)
            print("Please try to specify cpu extensions library path in demo's command line parameters using -l "
                      "or --cpu_extension command line argument", file=sys.stderr)
            sys.exit(1)
    input_CNN = next(iter(CNN_net.inputs))
    output_CNN = next(iter(CNN_net.outputs))
    exec_net = plugin.load(network=CNN_net, num_requests=1)
    del CNN_net
    return exec_net, input_CNN, output_CNN

def main():
    ''' accepts command line arguments for image file and the question and 
    builds the image model (VGG) and the VQA model (LSTM and MLP) 
    prints the top 5 response along with the probability of each '''

    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    exec_net, input_blob, output_blob = init_openvino()

    if verbose : print("Loading VQA Model ...")
    vqa_model = get_VQA_model(VQA_weights_file_name)

    # This task here is represented as a classification into a 1000 top answers
    # this means some of the answers were not part of trainng and thus would 
    # not show up in the result.
    # These 1000 answers are stored in the sklearn Encoder class
    with open(classes_file_name, 'r') as content_file:
        labels = content_file.read().split('\n')
    
    def app(env, start_response):
        start_response("200 OK", [("Content-type", "text/html; charset=utf-8")])
        q = parse_qs(env["QUERY_STRING"])
        print(q['q'][0])
        image_features = get_image_features(q["f"][0], exec_net, input_blob, output_blob)
        question_features = get_question_features(q["q"][0])
        y_output = vqa_model.predict([question_features, image_features])
        y_sort_index = np.argsort(y_output)
        found = False
        for label in reversed(y_sort_index[0,-5:]):
            print(labels[label], y_output[0, label])
            if not found:
                found = True
                best = labels[label]
                if y_output[0, label] < 0.2:
                    best = "Sorry, I'm not sure how to answer."
        return [best.encode('utf-8')]

    httpd = wsgiref.simple_server.make_server("localhost", 8080, app)
    httpd.serve_forever()
    
if __name__ == "__main__":
    main()
