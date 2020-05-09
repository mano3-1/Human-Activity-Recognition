from abc import ABC, abstractmethod
import tensorflow as tf


class BaseModel(ABC):

    # keys for the output_tensor_names map
    HEATMAP_KEY = "heatmap"
    OFFSETS_KEY = "offsets"
    DISPLACEMENT_FWD_KEY = "displacement_fwd"
    DISPLACEMENT_BWD_KEY = "displacement_bwd"

    def __init__(self, model_function, output_tensor_names, output_stride):
        self.output_stride = output_stride
        self.output_tensor_names = output_tensor_names
        self.model_function = model_function

    def valid_resolution(self, width, height):
        # calculate closest smaller width and height that is divisible by the stride after subtracting 1 (for the bias?)
        target_width = (int(width) // self.output_stride) * self.output_stride + 1
        target_height = (int(height) // self.output_stride) * self.output_stride + 1
        return target_width, target_height

    @abstractmethod
    def preprocess_input(self, image):
        pass

    def predict(self, image):
        input_image, image_scale = self.preprocess_input(image)

        input_image = tf.convert_to_tensor(input_image, dtype=tf.float32)

        result = self.model_function(input_image)

        heatmap_result = result[self.output_tensor_names[self.HEATMAP_KEY]]
        offsets_result = result[self.output_tensor_names[self.OFFSETS_KEY]]
        displacement_fwd_result = result[self.output_tensor_names[self.DISPLACEMENT_FWD_KEY]]
        displacement_bwd_result = result[self.output_tensor_names[self.DISPLACEMENT_BWD_KEY]]

        return tf.sigmoid(heatmap_result), offsets_result, displacement_fwd_result, displacement_bwd_result, image_scale
