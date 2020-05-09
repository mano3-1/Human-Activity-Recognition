from posenet.base_model import BaseModel
import numpy as np
import cv2


class MobileNet(BaseModel):

    def __init__(self, model_function, output_tensor_names, output_stride):
        super().__init__(model_function, output_tensor_names, output_stride)

    def preprocess_input(self, image):
        target_width, target_height = self.valid_resolution(image.shape[1], image.shape[0])
        # the scale that can get us back to the original width and height:
        scale = np.array([image.shape[0] / target_height, image.shape[1] / target_width])
        input_img = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB).astype(np.float32)  # to RGB colors

        input_img = input_img * (2.0 / 255.0) - 1.0  # normalize to [-1,1]
        input_img = input_img.reshape(1, target_height, target_width, 3)  # NHWC
        return input_img, scale
