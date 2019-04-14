import onnxmltools
from demo import get_image_model, get_VQA_model

VQA_weights_file_name   = 'models/VQA/VQA_MODEL_WEIGHTS.hdf5'
label_encoder_file_name = 'models/VQA/FULL_labelencoder_trainval.pkl'
CNN_weights_file_name   = 'models/CNN/vgg16_weights.h5'

# Convert it! The target_opset parameter is optional.
#print("First")
#keras_model = get_image_model(CNN_weights_file_name)
#onnx_model = onnxmltools.convert_keras(keras_model, target_opset=8)
#onnxmltools.utils.save_model(onnx_model, 'VQA_CNN.onnx')

print("Second")
keras_model = get_VQA_model(VQA_weights_file_name)
onnx_model = onnxmltools.convert_keras(keras_model, target_opset=8)
onnxmltools.utils.save_model(onnx_model, 'VQA_VQA.onnx')
