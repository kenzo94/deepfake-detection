import tensorflow as tf
import os

#lr
optimizer = tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9) # momentum
optimizer = tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9, nesterov=True) # momentum nach nesterov
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999) #adam optimierung

#model
base_model = tf.keras.applications.Xception(weights="image_net", include_top=False) 
#global avg pooling layer und dense ausgabeschicht wird weggelassen
avg = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
output = tf.keras.layers.Dense(n_classes, activation="softmax")(avg)
model = tf.keras.Model(inputs=base_model.input, outputs=output)

for layer in base_model.layers:
    layer.trainable = False
    
optimizer = tf.keras.optimizers.SGD(learning_rate=0.2, momentum=0.9, decay=0.01) #decay = power scheduling; momentum = optimierer
model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])

#callbacks
checkpoint_filepath = '.\\tmp_checkpoint'
print('Creating Directory: ' + checkpoint_filepath)
os.makedirs(checkpoint_filepath, exist_ok=True)

custom_callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor = 'val_loss',
        mode = 'min',
        patience = 5,
        verbose = 1
    ),
    tf.keras.callbacks.ModelCheckpoint(
        filepath = os.path.join(checkpoint_filepath, 'best_model.h5'),
        monitor = 'val_loss',
        mode = 'min',
        verbose = 1,
        save_best_only = True
    )
]

# Train network
history = model.fit(train_set, epochs=5, validation_data=valid_set)
print(history.history)

#nach einigen epochen die schichten wieder auftauen
# kleinere lernrate nehmen nehmen um die gewichte der vortrainierten gewichte nicht zu beschädigen

for layer in base_model.layers[:-1]: #wie viele layers wieder auftauen?
    layer.trainable = True 
    
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, decay=0.001)
model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
history = model.fit(train_set, epochs=5, validation_data=valid_set)