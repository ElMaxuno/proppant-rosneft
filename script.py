




# %% [code]
import tensorflow as tf
import numpy as np
import os
import pandas as pd
from RPCC_metric_utils_for_participants import sive_diam_pan

def get_submit(cnt_preds, dist_preds, indices):
    submit = []
    for idx, cnt, dist in zip(indices, cnt_preds, dist_preds):
        cnt = int(cnt)
        sizes = np.random.choice(sive_diam_pan, size=cnt, p=dist / np.sum(dist))
        submit.extend([{
            "ImageId": idx,
            "prop_size": sizes[i]
        } for i in range(cnt)])
    return pd.DataFrame.from_records(submit)

# %% [code]

TARGET_SIZE = [384,384]
AUTO = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 8

columns = ['ImageId', '6', '7', '8', '10', '12', '14', '16', '18', '20', '25',
       '30', '35', '40', '45', '50', '60', '70', '80', '100', 'pan',
       'prop_count', 'fraction']
prediction_columns = ['16', '18', '20', '25', '30', '35', '40', '45', '50']

# %% [code]
def decode_image(filename, label=None, image_size=TARGET_SIZE):
    bits = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(bits, channels=3)
    image = tf.cast(image, tf.float32) / 255.
#     try:
#         if image.shape[0]>image.shape[1]:
#             image = tf.image.transpose(image)
#         image = tf.image.crop_to_bounding_box(image, 0, int((image.shape[1] - image.shape[0])*.75) , int(image.shape[0]), int(image.shape[0]))
#     except TypeError:
#         pass
    image = tf.image.resize(image, image_size)
    
    if label is None:
        return image
    else:
        return image, label

def get_model():
        inp1 = tf.keras.layers.Input(shape = (*TARGET_SIZE, 3), name = 'inp1')
        pretrained_model = tf.keras.applications.MobileNetV2(weights = None, include_top = False)
        pretrained_model.trainable = True

        x = pretrained_model(inp1)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        output = tf.keras.layers.Dense(9, activation = 'softmax')(x)

        model = tf.keras.models.Model(inputs = [inp1], outputs = [output])

        # opt = tfa.optimizers.SWA(opt)

        model.compile(
            optimizer = 'adam',
            loss = 'mse',
            metrics = ['mse']
        )

        return model   

# %% [code]


# %% [code]
model= get_model()

model.load_weights('./models/model_count/model_count_weights')

# %% [code]
model_count = tf.keras.models.load_model('./models/modelcount.pb')


# %% [code]

# %% [code]
valid_paths = os.listdir('./data/test/')
valid_paths = pd.Series(valid_paths).apply(lambda x: './data/test/'+ x  )

# %% [code]
valid_dataset = (
    tf.data.Dataset
    .from_tensor_slices((valid_paths))
    .map(decode_image, num_parallel_calls=AUTO)
    .batch(BATCH_SIZE)
    .cache()
    .prefetch(AUTO)
)

# %% [code]
preds_dist = pd.DataFrame(np.zeros((len(valid_paths),23)))
preds_dist.columns = columns
preds_dist[prediction_columns] = model.predict(valid_dataset)
preds_dist.ImageId = valid_paths.apply(lambda x: int( x[len('./data/test/'): ][:-4] )) 
preds_dist.prop_count = [np.max([int(i[0]),688]) for i in model_count.predict(valid_dataset)]

# %% [code]
get_submit(preds_dist.prop_count,    preds_dist[list(preds_dist.columns[1:21])].fillna(.5).values ,   preds_dist.ImageId ).to_csv("answers.csv", index=False)
