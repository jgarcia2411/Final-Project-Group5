# ------------------------------------------------------------------------------------------------------------------
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import f1_score, cohen_kappa_score, accuracy_score, matthews_corrcoef
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import datasets, layers, models, optimizers
from tf_explain.callbacks.grad_cam import GradCAM

# ------------------------------------------------------------------------------------------------------------------

## Process images in parallel
AUTOTUNE = tf.data.AUTOTUNE

## folder "Data" images
## folder "excel" excel file , whatever is there is the file
## get the classes from the excel file
## folder "Documents" readme file

OR_PATH = os.getcwd()
os.chdir("..")  # Change to the parent directory
PATH = os.getcwd()
DATA_DIR = os.getcwd() + os.path.sep + 'Data' + os.path.sep
sep = os.path.sep
os.chdir(OR_PATH)  # Come back to the folder where the code resides , all files will be left on this directory

n_epoch = 200
BATCH_SIZE = 32

## Image processing
CHANNELS = 3
IMAGE_SIZE = 256

NICKNAME = 'Pokemon'


# ------------------------------------------------------------------------------------------------------------------

def process_target(target_type):
    '''
        1- Multiclass  target = (1...n, text1...textn)
        2- Multilabel target = ( list(Text1, Text2, Text3 ) for each observation, separated by commas )
        3- Binary   target = (1,0)

    :return:
    '''

    class_names = np.sort(xdf_data['target'].unique())

    if target_type == 1:

        x = lambda x: tf.argmax(x == class_names).numpy()

        final_target = xdf_data['target'].apply(x)

        final_target = to_categorical(list(final_target))

        xfinal = []
        for i in range(len(final_target)):
            joined_string = ",".join(str(int(e)) for e in (final_target[i]))
            xfinal.append(joined_string)
        final_target = xfinal

        xdf_data['target_class'] = final_target

    if target_type == 2:
        target = np.array(xdf_data['target'].apply(lambda x: x.split(",")))

        xdepth = len(class_names)

        final_target = tf.one_hot(target, xdepth)

        xfinal = []
        if len(final_target) == 0:
            xerror = 'Could not process Multilabel'
        else:
            for i in range(len(final_target)):
                joined_string = ",".join(str(e) for e in final_target[i])
                xfinal.append(joined_string)
            final_target = xfinal

        xdf_data['target_class'] = final_target

    if target_type == 3:
        # target_class is already done
        pass

    return class_names


# ------------------------------------------------------------------------------------------------------------------

def process_path(feature, target):
    '''
          feature is the path and id of the image
          target is the result
          returns the image and the target as label
    '''

    label = target

    file_path = feature
    img = tf.io.read_file(file_path)
    img = tf.io.decode_jpeg(img, channels=CHANNELS)

    img = tf.image.resize(img, [IMAGE_SIZE, IMAGE_SIZE])

    # augmentation

    return img, label


# ------------------------------------------------------------------------------------------------------------------

def get_target(num_classes):
    '''
    Get the target from the dataset
    1 = multiclass
    2 = multilabel
    3 = binary
    '''

    y_target = np.array(xdf_dset['target_class'].apply(lambda x: ([int(i) for i in str(x).split(",")])))

    end = np.zeros(num_classes)
    for s1 in y_target:
        end = np.vstack([end, s1])

    y_target = np.array(end[1:])

    return y_target


# ------------------------------------------------------------------------------------------------------------------


def read_data(num_classes):
    '''
          reads the dataset and process the target
    '''

    ds_inputs = np.array(DATA_DIR + xdf_dset['id'])
    ds_targets = get_target(num_classes)

    list_ds = tf.data.Dataset.from_tensor_slices(
        (ds_inputs, ds_targets))  # creates a tensor from the image paths and targets

    final_ds = list_ds.map(process_path, num_parallel_calls=AUTOTUNE).batch(BATCH_SIZE)

    return final_ds


# ------------------------------------------------------------------------------------------------------------------

def save_model(model):
    '''
         receives the model and print the summary into a .txt file
    '''
    with open('summary_{}.txt'.format(NICKNAME), 'w') as fh:
        # Pass the file handle in as a lambda function to make it callable
        model.summary(print_fn=lambda x: fh.write(x + '\n'))


# ------------------------------------------------------------------------------------------------------------------

def model_definition():
    # Define a Keras sequential model

    kernel_size = (3, 3)
    pool_size = (2, 2)

    model = tf.keras.Sequential()

    model.add(layers.Conv2D(16, kernel_size, activation='relu', input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)))
    model.add(layers.AvgPool2D(pool_size))
    model.add(layers.Dropout(0.2))

    model.add(layers.Conv2D(32, kernel_size, activation='relu', input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)))
    model.add(layers.AvgPool2D(pool_size))
    model.add(layers.Dropout(0.2))

    model.add(layers.Conv2D(64, kernel_size, activation='relu', input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)))
    model.add(layers.AvgPool2D(pool_size))
    model.add(layers.Dropout(0.2))

    model.add(layers.Conv2D(64, kernel_size, activation='relu', input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)))
    model.add(layers.AvgPool2D(pool_size))

    model.add(layers.Dropout(0.2))
    model.add(layers.Flatten())

    model.add(layers.Dense(OUTPUTS_a, activation='softmax'))  # final layer , outputs_a is the number of targets

    model.compile(optimizers.Adam(0.001), loss='categorical_crossentropy', metrics=['accuracy'])

    save_model(model)  # print Summary
    return model


# ------------------------------------------------------------------------------------------------------------------

def train_func(train_ds, val_ds):
    '''
        train the model
    '''

    early_stop = tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience=40)
    check_point = tf.keras.callbacks.ModelCheckpoint('model_{}.h5'.format(NICKNAME), monitor='accuracy',
                                                     save_best_only=True)
    final_model = model_definition()

    final_model.fit(train_ds, epochs=n_epoch, validation_data=val_ds, callbacks=[early_stop, check_point])


# ------------------------------------------------------------------------------------------------------------------

def predict_func(test_ds):
    '''
        predict fumction
    '''

    final_model = tf.keras.models.load_model('model_{}.h5'.format(NICKNAME))
    res = final_model.predict(test_ds)
    xres = [tf.argmax(f).numpy() for f in res]
    xdf_dset['results'] = xres
    xdf_dset.to_excel('results_{}.xlsx'.format(NICKNAME), index=False)


# ------------------------------------------------------------------------------------------------------------------

def metrics_func(metrics, aggregates=[]):
    '''
    multiple functiosn of metrics to call each function
    f1, cohen, accuracy, mattews correlation
    list of metrics: f1_micro, f1_macro, f1_avg, coh, acc, mat
    list of aggregates : avg, sum
    :return:
    '''

    def f1_score_metric(y_true, y_pred, type):
        '''
            type = micro,macro,weighted,samples
        :param y_true:
        :param y_pred:
        :param average:
        :return: res
        '''
        res = f1_score(y_true, y_pred, average=type)
        print("f1_score {}".format(type), res)
        return res

    def cohen_kappa_metric(y_true, y_pred):
        res = cohen_kappa_score(y_true, y_pred)
        print("cohen_kappa_score", res)
        return res

    def accuracy_metric(y_true, y_pred):
        res = accuracy_score(y_true, y_pred)
        print("accuracy_score", res)
        return res

    def matthews_metric(y_true, y_pred):
        res = matthews_corrcoef(y_true, y_pred)
        print('mattews_coef', res)
        return res

    # For multiclass

    x = lambda x: tf.argmax(x == class_names).numpy()
    y_true = np.array(xdf_dset['target'].apply(x))
    y_pred = np.array(xdf_dset['results'])

    # End of Multiclass

    xcont = 1
    xsum = 0
    xavg = 0

    for xm in metrics:
        if xm == 'f1_micro':
            # f1 score average = micro
            xmet = f1_score_metric(y_true, y_pred, 'micro')
        elif xm == 'f1_macro':
            # f1 score average = macro
            xmet = f1_score_metric(y_true, y_pred, 'macro')
        elif xm == 'f1_weighted':
            # f1 score average =
            xmet = f1_score_metric(y_true, y_pred, 'weighted')
        elif xm == 'coh':
            # Cohen kappa
            xmet = cohen_kappa_metric(y_true, y_pred)
        elif xm == 'acc':
            # Accuracy
            xmet = accuracy_metric(y_true, y_pred)
        elif xm == 'mat':
            # Matthews
            xmet = matthews_metric(y_true, y_pred)
        else:
            xmet = print('Metric does not exist')

        xsum = xsum + xmet
        xcont = xcont + 1

    if 'sum' in aggregates:
        print('Sum of Metrics : ', xsum)
    if 'avg' in aggregates and xcont > 0:
        print('Average of Metrics : ', xsum / xcont)
    # Ask for arguments for each metric


# ------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":

    for file in os.listdir(PATH + os.path.sep + "Excel"):
        if file[-5:] == '.xlsx':
            FILE_NAME = PATH + os.path.sep + "Excel" + os.path.sep + file

    # Reading and filtering Excel file
    xdf_data = pd.read_excel(FILE_NAME)

    class_names = process_target(1)  # 1: Multiclass 2: Multilabel 3:Binary

    # INPUTS_r = IMAGE_SIZE * IMAGE_SIZE * CHANNELS
    OUTPUTS_a = len(class_names)

    ## Processing Train dataset

    xdf_dset = xdf_data[xdf_data["split"] == 'train'].copy()
    train_ds = read_data(OUTPUTS_a)

    xdf_dset = xdf_data[xdf_data["split"] == 'val'].copy()
    val_ds = read_data(OUTPUTS_a)

    train_func(train_ds, val_ds)

    # Preprocessing Test dataset

    xdf_dset = xdf_data[xdf_data["split"] == 'test'].copy()

    test_ds = read_data(OUTPUTS_a)
    predict_func(test_ds)

    ## Metrics Function over the result of the test dataset
    list_of_metrics = ['f1_macro', 'acc']
    metrics_func(list_of_metrics)

    img = tf.keras.preprocessing.image.load_img(DATA_DIR+'1000.jpg', target_size=(256, 256))
    img = tf.keras.preprocessing.image.img_to_array(img)
    data = ([img], None)

    final_model = tf.keras.models.load_model('model_{}.h5'.format(NICKNAME))

    explainer = GradCAM()
    grid = explainer.explain(data, final_model, class_index=2)

    explainer.save(grid, ".", "grad_cam.png")




# ------------------------------------------------------------------------------------------------------------------
