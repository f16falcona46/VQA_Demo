import onnxmltools
import tensorflow as tf
from demo import get_image_model, get_VQA_model

VQA_weights_file_name   = 'models/VQA/VQA_MODEL_WEIGHTS.hdf5'
label_encoder_file_name = 'models/VQA/FULL_labelencoder_trainval.pkl'
CNN_weights_file_name   = 'models/CNN/vgg16_weights.h5'

# Convert it! The target_opset parameter is optional.
print("First")
model = get_image_model(CNN_weights_file_name)
model.save("CNN.h5")

print("Second")
model = get_VQA_model(VQA_weights_file_name)
model.save("VQA.h5")
