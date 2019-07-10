from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Flatten,Dense
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras.preprocessing import image
from keras.callbacks import ModelCheckpoint
from keras.models import  load_model
from sklearn.metrics import confusion_matrix
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
import matplotlib.pyplot as plt
from keras.callbacks import CSVLogger
import  pandas
img_width , img_height = 48 , 48
import matplotlib.image as mpimg
import xlsxwriter

def cnn_training():

    print("initializing model ...")
    classifier = Sequential()

    print("Convolution Layer ...1 ")
    classifier.add(Conv2D(16, (3, 3), input_shape = (img_width, img_height, 3), activation = 'relu'))

    print("Max pooling Layer ...1")
    classifier.add(MaxPooling2D(pool_size = (2 ,2)))

    print("Convolution Layer ...2 ")
    classifier.add(Conv2D(32, (3, 3), input_shape=(img_width, img_height, 3), activation='relu'))

    print("Max pooling Layer ...2")
    classifier.add(MaxPooling2D(pool_size=(2, 2)))

    print("Convolution Layer ... 3")
    classifier.add(Conv2D(64, (3, 3), input_shape=(img_width, img_height, 3), activation='relu'))

    print("Max pooling Layer ...3")
    classifier.add(MaxPooling2D(pool_size=(2, 2)))

    print("Form 2D to 1D ...")
    classifier.add(Flatten())

    print("Fully connected Layer ...")
    classifier.add(Dense(units = 128, activation = 'relu'))

    classifier.add(Dense(units = 4, activation = 'softmax'))

    classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

    train_datagen = ImageDataGenerator(rescale = 1. /255,shear_range = 0.2,zoom_range = 0.2,horizontal_flip = True)
    test_datagen = ImageDataGenerator(rescale = 1./255)

    training_set = train_datagen.flow_from_directory('./data/training_set',target_size = (img_width , img_height),batch_size = 16,class_mode = 'categorical')
    test_set = test_datagen.flow_from_directory('./data/test_set',target_size = (img_width, img_height),batch_size = 16,class_mode = 'categorical')

    #checkpoint = ModelCheckpoint('best_weights.hdf5', monitor='val_acc', verbose=1,save_best_only=True,mode='min')

    pandas.DataFrame(classifier.fit_generator(training_set,steps_per_epoch = 800,epochs = 5,validation_data = test_set,validation_steps = 200).history).to_csv("history.csv")

    workbook = xlsxwriter.Workbook("cnn_weights.xlsx")
    test = workbook.add_worksheet()
    cnt = 0
    for layer in classifier.layers:
        weights = layer.get_weights()
        c1 = 0;
        c2 = 0
        for i in weights:
            for j in i:
                test.write(c1, c2, str(j))
                c2 += 1
            c1 += 1
    workbook.close()





    # Plot training & validation accuracy values
    # plt.plot(history.history['acc'])
    # plt.plot(history.history['val_acc'])
    # plt.title('Model accuracy')
    # plt.ylabel('Accuracy')
    # plt.xlabel('Epoch')
    # plt.legend(['Train', 'Test'], loc='upper left')
    # plt.show()
    # plt.savefig("acc.png")
    #
    # # Plot training & validation loss values
    # plt.plot(history.history['loss'])
    # plt.plot(history.history['val_loss'])
    # plt.title('Model loss')
    # plt.ylabel('Loss')
    # plt.xlabel('Epoch')
    # plt.legend(['Train', 'Test'], loc='upper left')
    # plt.show()
    # plt.savefig("loss.png")


#print(testAcc)
def cnn_model_testing(image_path):
    classifier = load_model("best_weights.hdf5")




    test_image = image.load_img(image_path, target_size = (img_width, img_height))
    test_image = test_image.resize((img_width, img_height))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    result = classifier.predict(test_image)
    print(result)



    if result[0][0] == 1:
        prediction = 'Aya'
    elif result[0][1] ==1:
        prediction = 'Foda'
    elif result[0][2] ==1:
        prediction = 'Maii'
    elif result[0][3] ==1:
        prediction = 'Mostafa'

    img = mpimg.imread(image_path)
    imgplot = plt.imshow(img)
    plt.title(prediction)
    imgplot = plt.imshow(img)
    plt.show()


    print (prediction)

cnn_training()




# test_datagen = ImageDataGenerator(rescale=1. / 255)
#     test_set = test_datagen.flow_from_directory('./data/test_set', target_size=(img_width, img_height), batch_size=16,
#                                                 class_mode=None)
#
#     y_pred = np.array(classifier.predict_generator(test_set , 200))
#     #y_pred = np.array( (predictions > 0.5))
#
#     y_test = np.array([1,0,0,0] * 50 + [0,1,0,0] * 50 + [0,0,1,0]*50 + [0,0,0,1]*50)
#     correct = 0
#     print (y_pred.shape)
#     print (y_test.shape)