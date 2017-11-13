from generator1 import *
from keras import regularizers
from keras.optimizers import SGD, RMSprop
from keras.models import Model, load_model
from keras.callbacks import CSVLogger, ModelCheckpoint
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.layers import Dense, Lambda, Dropout, GlobalAveragePooling2D, BatchNormalization

epochs          = 100
batch_size      = 32
widths, heights = 256, 256

base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(widths, heights, 3))    # 迁移学习，载入InceptionV3的权重，拿来直接用

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)  # new FC layer, random init
x = BatchNormalization()(x)
x = Dropout(0.1)(x)
predictions = Dense(2, activation='softmax')(x)
#predictions = Dense(2, activation='softmax', kernel_regularizer=regularizers.l2(0.01))(x)  # new softmax layer
model = Model(inputs=base_model.input, outputs=predictions)
for layer in base_model.layers:
    layer.trainable = True
model.compile(optimizer=SGD(lr=0.01), loss='categorical_crossentropy', metrics=['accuracy'])

#model = load_model('model_1000_8_13.h5')

#model.compile(optimizer=SGD(lr=0.00007, momentum=0.5, decay=0.05), loss='categorical_crossentropy', metrics=['accuracy'])
#model.compile(optimizer=RMSprop(lr=0.05,epsilon=1.0,decay=0.5), loss='categorical_crossentropy', metrics=['accuracy'])

csvlogger = CSVLogger('log10032100_02.log', append=True)
model_check = ModelCheckpoint('model10032100_02_p.h5', monitor='val_loss', save_best_only=True)

model.fit_generator(generator(train_set=True, batch_size=batch_size, widths=widths, heights=heights),
                    steps_per_epoch=100,
                    epochs=epochs,
                    validation_data=generator(train_set=False, batch_size=batch_size, widths=widths, heights=heights),
                    validation_steps=50,
                    verbose=1,
                    workers=100,
                    max_q_size=128,
                    callbacks=[csvlogger, model_check])

model.save('model10032100_02.h5')






