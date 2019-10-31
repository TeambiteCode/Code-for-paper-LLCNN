'''
harshana.w@eng.pdn.ac.lk wrote this code for http://teambitecode.com/fyp
'''
import tensorflow as tf
import keras
from keras.layers import Input,Dense,Activation,Conv2D,Lambda
from keras.models import Model
import cv2,os
import numpy as np
import keras.backend as K
from keras.models import model_from_json
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from .. Image_handler import ImageHandler
from utils import CustomCallback, DataGenerator, tSNE
import argparse



############ args #######################################
VIEW = False

CONV_BLOCKS = 2
RESIDUAL = True
SSMI_KERNAL = (9, 9)
SSMI_STRIDE = (3, 3)
PATCHSIZE = (41, 41)
IMAGESIZE = (256, 256)
#########################################################


def convBlock(x):
    x1 = Conv2D(64,(1,1),strides=(1,1),padding='same',use_bias=True)(x)

    x2 = Conv2D(64, (3, 3), strides=(1, 1), padding='same', use_bias=True,activation='relu')(x)
    x2 = Conv2D(64, (3, 3), strides=(1, 1), padding='same', use_bias=True)(x2)

    x3 = Lambda(lambda a: a[0] + a[1])([x1, x2])
    x3 = Activation('relu')(x3)

    x4 = Conv2D(64, (3, 3), strides=(1, 1), padding='same', use_bias=True, activation='relu')(x3)
    x4 = Conv2D(64, (3, 3), strides=(1, 1), padding='same', use_bias=True)(x4)

    x5=Lambda(lambda a: a[0]+a[1])([x3,x4])
    out=Activation('relu')(x5)

    return out

def convBlockwoResidual(x):
    x1 = Conv2D(64, (3, 3), strides=(1, 1), padding='same', use_bias=True,activation='relu')(x)
    x2 = Conv2D(64, (3, 3), strides=(1, 1), padding='same', use_bias=True)(x1)
    x3 = Conv2D(64, (3, 3), strides=(1, 1), padding='same', use_bias=True, activation='relu')(x2)
    x4 = Conv2D(64, (3, 3), strides=(1, 1), padding='same', use_bias=True)(x3)
    out=Activation('relu')(x4)
    return out    

def makeLLCNN(inputTensor):

    x = Conv2D(64,(1,1),strides=(1,1),padding='same',use_bias=True)(inputTensor)
    x = Activation('relu')(x)

    for l in range(CONV_BLOCKS):
        if RESIDUAL:
            x = convBlock(x)
        else:
            x = convBlockwoResidual(x)

    out = Conv2D(3, (1, 1), strides=(1, 1), padding='same', use_bias=True)(x)
    return out

def custom_loss(height, width, batchsize, kernal, strides, channels):
    assert len(kernal)==2 and kernal[0]%2==1 and kernal[1]%2==1 # odd sized kernal to make things easy
    assert len(kernal) == len(strides)
    k1 = 0.01
    k2 = 0.03
    bits_per_pixel = 8*channels
    L = 2**bits_per_pixel - 1

    c1 = (k1*L)**2
    c2 = (k2*L)**2
    c1, c2 = 0.0001, 0.001
    print("c1 {} c2 {}".format(c1, c2))
    def SSMI(x, y):
        '''
        x : 4D array of the expected value (samples, **image_patch)
        y : 4D array of the predicted value (samples, **image_patch)
        '''
        assert len(K.int_shape(x))==4
        assert len(K.int_shape(y))==4

        miu_x = K.mean(x, axis=(1,2,3), keepdims=True)
        miu_y = K.mean(y, axis=(1,2,3), keepdims=True)
        sigma_xx = K.mean((x - miu_x)*(x - miu_x), axis=(1,2,3))
        sigma_yy = K.mean((y - miu_y)*(y - miu_y), axis=(1,2,3))
        sigma_xy = K.mean((x - miu_x)*(y - miu_y), axis=(1,2,3))

        miu_x = K.flatten(miu_x)
        miu_y = K.flatten(miu_y)

        return (2*miu_x*miu_y + c1)*(2*sigma_xy + c2)/(miu_x**2 + miu_y**2 + c1)/(sigma_xx + sigma_yy + c2)

    def f(y_true, y_pred):
        padding_pattern = ((kernal[0]//2, kernal[0]//2), (kernal[1]//2, kernal[1]//2))
        y_true = K.spatial_2d_padding(y_true, padding_pattern)
        y_pred = K.spatial_2d_padding(y_pred, padding_pattern)
        
        kernal_x = kernal[0]
        kernal_y = kernal[1]
        stride_x, stride_y = strides
        x = 0
        for idx_x in range(0, height, stride_y):
            print(idx_x)
            y = 0
            for idx_y in range(0, width, stride_x):
                if K.image_data_format() == 'channels_first':
                    tmp=1-SSMI(y_true[:, :, idx_x:idx_x+kernal_x, idx_y:idx_y+kernal_y], y_pred[:, :, idx_x:idx_x+kernal_x, idx_y:idx_y+kernal_y])
                else:
                    tmp=1-SSMI(y_true[:, idx_x:idx_x+kernal_x, idx_y:idx_y+kernal_y, :], y_pred[:, idx_x:idx_x+kernal_x, idx_y:idx_y+kernal_y, :])
                try:
                    loss_ssmi=K.concatenate([loss_ssmi, K.expand_dims(tmp,0)],0)
                except Exception as e:
                    loss_ssmi= K.expand_dims(tmp,0)
                y += 1
            x += 1
        loss_ssmi=K.transpose(loss_ssmi)
        
        true_loss_ssmi = loss_ssmi[:K.shape(y_true)[0]]

        return true_loss_ssmi
    return f


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-p1", "--pathtrue", help="Path to true images folder", type=str, required=True)
    parser.add_argument("-p2", "--pathdark", help="Path to dark images folder", type=str, required=True)
    parser.add_argument("-l", "--loss", help="Loss function used", default='mse')
    parser.add_argument("-e", "--epochs", default=10, help="Number of epochs to train", type=int)
    parser.add_argument("-b", "--batchsize", default=4, help="Batch size", type=int)
    parser.add_argument("-s","--save", help="Save stuff", action="store_true")
    parser.add_argument("-c","--createdb", help="Create DB using gamma correction. Please specify original image folder path.")
    parser.add_argument("-f","--forcetrain", help="Force train without loading weights from hdd", action="store_true")

    args = parser.parse_args()
    
    img_hndlr = ImageHandler(IMAGESIZE, patch_size=PATCHSIZE)

    if args.createdb is not None:
        img_hndlr.create_dataset(args.createdb, args.pathtrue, args.pathdark)
    elif not os.path.exists(args.pathdark):
        raise Exception("Path to dark images do not exist")
    elif not os.path.exists(args.pathtrue):
        raise Exception("Path to true images do not exist")    
    elif img_hndlr.load_images(args.pathtrue, 1)[0].shape != img_hndlr.load_images(args.pathdark, 1)[0].shape != (*IMAGESIZE, 3):
        raise Exception("Image dimensions do not match to the defined image size")

    ############ LOAD IMAGES ################################
    X = img_hndlr.load_images(args.pathdark, 16)
    Y = img_hndlr.load_images(args.pathtrue, 16)

    X = img_hndlr.preprocess_images(X)
    Y = img_hndlr.preprocess_images(Y)
    print("XY shapes",X.shape,Y.shape)

    ############ BUILD MODEL ################################
    # build model
    if os.path.isfile('model.json') and not args.forcetrain:
        json_file = open('model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        print("Loaded model from json")
    else:
        # build from scratch
        inp = Input((*PATCHSIZE, 3))
        out = makeLLCNN(inp)
        model = Model(inp, out)

    # load weights if available
    if os.path.isfile('model.h5') and not args.forcetrain:
        try:
            model.load_weights("model.h5")
            print("Loaded model weights from HFD5")
        except Exception as e:
            args.forcetrain = True
            print(e)

    # compile model
    if args.loss == 'mse' or args.loss == 'mean_squared_error':
        loss = keras.losses.mean_squared_error
    elif args.loss == 'ssmi':
        loss = custom_loss(X[0].shape[0], X[0].shape[1], args.batchsize, SSMI_KERNAL, strides=SSMI_STRIDE, channels=3)
    
    opt = keras.optimizers.SGD(lr=0.01, momentum=0.9, decay=0.0001, nesterov=False)
    #opt = 'adam'
    model.compile(optimizer=opt,loss=loss,metrics=['mean_squared_error'])
    
    print(model.summary())
    

    ############ TRAIN ######################################
    
    callbacks_list = []
    if args.save:
        callbacks_list.append(ModelCheckpoint("weights-{epoch:02d}.hdf5", monitor='val_loss', verbose=1, save_best_only=True, mode='min'))
        callbacks_list.append(CustomCallback(model, img_hndlr, img_hndlr.load_images(args.pathdark, 3), view=VIEW))
        
    #tensorboard = TensorBoard(log_dir='./logs', write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None, update_freq='epoch')
    callbacks_list.append(EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=10, verbose=0, mode='auto', baseline=None))
    
    # train
    if not os.path.isfile('model.h5') or args.forcetrain:
        model.fit(X, Y, epochs=args.epochs, batch_size=args.batchsize, validation_split=0.1, verbose=1, callbacks=callbacks_list)
        
        # train using a generator. !USE THIS FOR LARGE DATASETS!
        # traindata = DataGenerator(args.pathdark, args.pathtrue, img_hndlr)
        # model.fit_generator(traindata, epochs=3, verbose=1, callbacks=callbacks_list, validation_data=None, shuffle=True)
    
    if args.save:
        # save model to JSON
        model_json = model.to_json()
        with open("model.json", "w") as json_file:
            json_file.write(model_json)
            print("Model saved to json.")
        # save weights
        model.save_weights("model.h5")
        print("Saved model weights to HFD5")

    ############ PREDICT ####################################
    print("Predicting {} images".format(len(X)))
    Y_hat = model.predict(X)
    eval_loss = K.eval(loss(Y.astype(np.float64), Y_hat.astype(np.float64)))
    eval_loss = eval_loss.reshape((eval_loss.shape[0], -1))
    
    ############ SAVE RESULTS ###############################
    np.save('loss_values_{}.npy'.format(args.loss), eval_loss)
    
    Y_hat = img_hndlr.inv_preprocess_images(Y_hat)
    Y = img_hndlr.inv_preprocess_images(Y)
    X = img_hndlr.inv_preprocess_images(X)
    img_hndlr.save_images("pred/", np.hstack([X, Y_hat, Y]))
    
    #########################################################
    #all = np.concatenate([X,Y,Y_hat], axis=0)
    #all = all.reshape((all.shape[0],-1))
    #label = np.arange(3).repeat(X.shape[0])
    
    #tSNE(all, label)

    
