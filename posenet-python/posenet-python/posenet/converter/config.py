import os

BASE_DIR = os.path.dirname(__file__)
TFJS_MODEL_DIR = './_tfjs_models'
TF_MODEL_DIR = './_tf_models'

MOBILENET_BASE_URL = 'https://storage.googleapis.com/tfjs-models/savedmodel/posenet/mobilenet/'
RESNET50_BASE_URL = 'https://storage.googleapis.com/tfjs-models/savedmodel/posenet/resnet50/'

POSENET_ARCHITECTURE = 'posenet'

RESNET50_MODEL = 'resnet50'
MOBILENET_MODEL = 'mobilenet'


def bodypix_resnet50_config(stride, quant_bytes=4):

    graph_json = 'model-stride' + str(stride) + '.json'

    # quantBytes = 4 corresponding to the non - quantized full - precision checkpoints.
    if quant_bytes == 4:
        base_url = RESNET50_BASE_URL + 'float'
        model_dir = RESNET50_MODEL + '_float'
    else:
        base_url = RESNET50_BASE_URL + 'quant' + str(quant_bytes) + '/'
        model_dir = RESNET50_MODEL + '_quant' + str(quant_bytes)

    stride_dir = 'stride' + str(stride)

    return {
        'base_url': base_url,
        'filename': graph_json,
        'output_stride': stride,
        'data_format': 'NHWC',
        'input_tensors': {
            'image': 'sub_2:0'
        },
        'output_tensors': {
            'heatmap': 'float_heatmaps:0',
            'offsets': 'float_short_offsets:0',
            'displacement_fwd': 'resnet_v1_50/displacement_fwd_2/BiasAdd:0',
            'displacement_bwd': 'resnet_v1_50/displacement_bwd_2/BiasAdd:0'
        },
        'tfjs_dir': os.path.join(TFJS_MODEL_DIR, POSENET_ARCHITECTURE, model_dir, stride_dir),
        'tf_dir': os.path.join(TF_MODEL_DIR, POSENET_ARCHITECTURE, model_dir, stride_dir)
    }


def bodypix_mobilenet_config(stride, quant_bytes=4, multiplier=1.0):

    graph_json = 'model-stride' + str(stride) + '.json'

    multiplier_map = {
        1.0: "100",
        0.75: "075",
        0.5: "050"
    }

    # quantBytes = 4 corresponding to the non - quantized full - precision checkpoints.
    if quant_bytes == 4:
        base_url = MOBILENET_BASE_URL + 'float/' + multiplier_map[multiplier] + '/'
        model_dir = MOBILENET_MODEL + '_float_' + multiplier_map[multiplier]
    else:
        base_url = MOBILENET_BASE_URL + 'quant' + str(quant_bytes) + '/' + multiplier_map[multiplier] + '/'
        model_dir = MOBILENET_MODEL + '_quant' + str(quant_bytes) + '_' + multiplier_map[multiplier]

    stride_dir = 'stride' + str(stride)

    return {
        'base_url': base_url,
        'filename': graph_json,
        'output_stride': stride,
        'data_format': 'NHWC',
        'input_tensors': {
            'image': 'sub_2:0'
        },
        'output_tensors': {
            'heatmap': 'MobilenetV1/heatmap_2/BiasAdd:0',
            'offsets': 'MobilenetV1/offset_2/BiasAdd:0',
            'displacement_fwd': 'MobilenetV1/displacement_fwd_2/BiasAdd:0',
            'displacement_bwd': 'MobilenetV1/displacement_bwd_2/BiasAdd:0'
        },
        'tfjs_dir': os.path.join(TFJS_MODEL_DIR, POSENET_ARCHITECTURE, model_dir, stride_dir),
        'tf_dir': os.path.join(TF_MODEL_DIR, POSENET_ARCHITECTURE, model_dir, stride_dir)
    }
