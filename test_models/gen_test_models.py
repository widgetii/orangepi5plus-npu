#!/usr/bin/env python3
"""Generate minimal INT8 test models for each new Rocket NPU software op."""
import os
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

def representative_dataset(shape):
    for _ in range(100):
        yield [np.random.rand(1, *shape).astype(np.float32)]

def quantize_model(model, input_shape):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: representative_dataset(input_shape)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    return converter.convert()

def gen_concat_model():
    """CONV -> split into 2 convs -> CONCATENATION -> CONV"""
    inp = tf.keras.Input(shape=(8, 8, 3))
    conv1 = tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu')(inp)
    conv2 = tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu')(inp)
    concat = tf.keras.layers.Concatenate(axis=-1)([conv1, conv2])
    out = tf.keras.layers.Conv2D(8, 1, padding='same')(concat)
    model = tf.keras.Model(inp, out)
    return quantize_model(model, (8, 8, 3))

def gen_maxpool_model():
    """CONV -> MAX_POOL_2D -> CONV"""
    inp = tf.keras.Input(shape=(8, 8, 3))
    conv1 = tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu')(inp)
    pool = tf.keras.layers.MaxPooling2D(pool_size=3, strides=1, padding='same')(conv1)
    out = tf.keras.layers.Conv2D(8, 1, padding='same')(pool)
    model = tf.keras.Model(inp, out)
    return quantize_model(model, (8, 8, 3))

def gen_pad_model():
    """PAD -> CONV"""
    inp = tf.keras.Input(shape=(8, 8, 3))
    padded = tf.keras.layers.ZeroPadding2D(padding=((1, 1), (1, 1)))(inp)
    out = tf.keras.layers.Conv2D(16, 3, padding='valid', activation='relu')(padded)
    model = tf.keras.Model(inp, out)
    return quantize_model(model, (8, 8, 3))

def gen_resize_model():
    """CONV -> RESIZE_NEAREST_NEIGHBOR -> CONV"""
    inp = tf.keras.Input(shape=(8, 8, 3))
    conv1 = tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu')(inp)
    resized = tf.keras.layers.Lambda(
        lambda x: tf.image.resize(x, [16, 16], method='nearest'))(conv1)
    out = tf.keras.layers.Conv2D(8, 1, padding='same')(resized)
    model = tf.keras.Model(inp, out)
    return quantize_model(model, (8, 8, 3))

def gen_logistic_model():
    """CONV -> LOGISTIC -> CONV"""
    inp = tf.keras.Input(shape=(8, 8, 3))
    conv1 = tf.keras.layers.Conv2D(16, 3, padding='same')(inp)
    sigmoid = tf.keras.layers.Activation('sigmoid')(conv1)
    out = tf.keras.layers.Conv2D(8, 1, padding='same')(sigmoid)
    model = tf.keras.Model(inp, out)
    return quantize_model(model, (8, 8, 3))

if __name__ == '__main__':
    outdir = os.path.dirname(os.path.abspath(__file__))
    models = {
        'test_concat.tflite': gen_concat_model,
        'test_maxpool.tflite': gen_maxpool_model,
        'test_pad.tflite': gen_pad_model,
        'test_resize.tflite': gen_resize_model,
        'test_logistic.tflite': gen_logistic_model,
    }
    for name, gen_fn in models.items():
        print(f'Generating {name}...')
        tflite_model = gen_fn()
        path = os.path.join(outdir, name)
        with open(path, 'wb') as f:
            f.write(tflite_model)
        print(f'  -> {path} ({len(tflite_model)} bytes)')
    print('Done.')
