import os
import tensorflow as tf
import tfjs_graph_converter as tfjs
import posenet.converter.config as config
import posenet.converter.tfjsdownload as tfjsdownload


def __tensor_info_def(sess, tensor_names):
    signatures = {}
    for tensor_name in tensor_names:
        tensor = sess.graph.get_tensor_by_name(tensor_name)
        tensor_info = tf.compat.v1.saved_model.build_tensor_info(tensor)
        signatures[tensor_name] = tensor_info
    return signatures


def convert(model_cfg):
    model_file_path = os.path.join(model_cfg['tfjs_dir'], model_cfg['filename'])
    if not os.path.exists(model_file_path):
        print('Cannot find tfjs model path %s, downloading tfjs model...' % model_file_path)
        tfjsdownload.download_tfjs_model(model_cfg)

    # 'graph_model_to_saved_model' doesn't store the signature for the model!
    #   tfjs.api.graph_model_to_saved_model(model_cfg['tfjs_dir'], model_cfg['tf_dir'], ['serve'])
    # So we do it manually below.
    # This link was a great help to do this:
    # https://www.programcreek.com/python/example/104885/tensorflow.python.saved_model.signature_def_utils.build_signature_def

    graph = tfjs.api.load_graph_model(model_cfg['tfjs_dir'])
    builder = tf.compat.v1.saved_model.Builder(model_cfg['tf_dir'])

    with tf.compat.v1.Session(graph=graph) as sess:
        input_tensor_names = tfjs.util.get_input_tensors(graph)
        output_tensor_names = tfjs.util.get_output_tensors(graph)

        signature_inputs = __tensor_info_def(sess, input_tensor_names)
        signature_outputs = __tensor_info_def(sess, output_tensor_names)

        method_name = tf.compat.v1.saved_model.signature_constants.PREDICT_METHOD_NAME
        signature_def = tf.compat.v1.saved_model.build_signature_def(inputs=signature_inputs,
                                                                     outputs=signature_outputs,
                                                                     method_name=method_name)
        signature_map = {tf.compat.v1.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature_def}
        builder.add_meta_graph_and_variables(sess=sess,
                                             tags=['serve'],
                                             signature_def_map=signature_map)
    return builder.save()
