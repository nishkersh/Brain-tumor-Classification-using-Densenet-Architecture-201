# Brain-tumor-Classification-using-Densenet-Architecture-201
Here The code is being explained line by line
1. Importing Keras Modules and Libraries:
   - `from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization`: Import specific layer types such as Dense (fully connected), Flatten, Dropout, and BatchNormalization from the Keras Layers module, which are used to build the neural network architecture.
   - `from tensorflow.keras.models import Model`: Import the Model class from the Keras models module, used to create and manage neural network models.
   - `from keras.applications.densenet import DenseNet201`: Import the DenseNet201 pre-trained model architecture from the Keras applications.

2. Importing Other Libraries:
   - `from tensorflow.keras.preprocessing import image`: Import the image module from Keras preprocessing, which is used for image data processing.
   - `from tensorflow.keras.preprocessing.image import ImageDataGenerator`: Import the ImageDataGenerator class from Keras preprocessing to generate augmented image data for training deep learning models.
   - `from tensorflow.keras.models import Sequential`: Import the Sequential class from Keras models, used to create a sequential model (a linear stack of layers).
   - `import numpy as np`: Import NumPy, a library for numerical computing, and alias it as `np`.
   - `import matplotlib.pyplot as plt`: Import the pyplot module from Matplotlib, a plotting library, and alias it as `plt`.

- `IMAGE_SIZE = [128, 128]`: This line is a Python variable assignment. It creates a variable named `IMAGE_SIZE` and assigns it a list containing two integer values: 128 and 128.

  Explanation:
  - `IMAGE_SIZE`: This is the name of the variable. It is used to store information about the size of images that will be processed in the code.
  - `=`: This is the assignment operator in Python. It assigns the value on the right side of the `=` to the variable on the left side.
  - `[128, 128]`: This is a list literal in Python. It represents a list containing two elements, both of which are integers. In this case, the list contains two integers, 128 and 128.

  Therefore, the line of code creates a variable `IMAGE_SIZE` that will be used to represent the width and height of images in the code, and the width and height are both set to 128 pixels. This variable can be used throughout the code to ensure consistency in image size when performing image-related operations.
- `r = DenseNet201(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)`: This line of code creates a DenseNet201 model using Keras in TensorFlow.

  Explanation:
  - `r`: This is the variable name assigned to the DenseNet201 model.
  - `DenseNet201`: This is the pre-trained model architecture that will be used. DenseNet201 is a deep convolutional neural network architecture that has 201 layers.
  - `input_shape=IMAGE_SIZE + [3]`: This specifies the input shape of the model. It is a tuple representing the height, width, and number of color channels in the input images. The `IMAGE_SIZE` variable is used here, and `[3]` indicates that the input images have three color channels (RGB).
  - `weights='imagenet'`: This specifies that the model should be initialized with pre-trained weights from the ImageNet dataset. These pre-trained weights can help the model perform well on a wide range of image-related tasks.
  - `include_top=False`: This parameter determines whether to include the fully connected layers (top) of the model. In this case, `include_top` is set to `False`, indicating that the dense layers at the top of the model will be excluded. This is useful when using the model for transfer learning, where you may want to add your own custom top layers for a specific task.

  The variable `r` now holds the DenseNet201 model with pre-trained weights and configured to accept input images of size `IMAGE_SIZE`, with three color channels (RGB).
for layer in r.layers:
  layer.trainable = False

- `x = Flatten()(r.output)`: This line of code applies the `Flatten` layer to the output of the previously defined DenseNet201 model `r`. The `Flatten` layer reshapes the output tensor from the previous model into a 1D vector. This is necessary to connect the output to the subsequent Dense layers.

- `x = BatchNormalization()(x)`: This line applies the `BatchNormalization` layer to the tensor `x`. Batch normalization normalizes the activations of the previous layer across each batch of data, helping to stabilize and speed up the training process.

- `x = Dense(256, activation='relu')(x)`: This line adds a fully connected (Dense) layer to the network. The `Dense` layer has 256 neurons, and the activation function used is the rectified linear unit (ReLU). ReLU is a popular choice for activation functions as it introduces non-linearity and helps the network learn complex patterns.

- `prediction = Dense(4, activation='softmax')(x)`: This line adds another fully connected (Dense) layer, which serves as the output layer of the neural network. The output layer has 4 neurons, corresponding to the number of classes in the classification problem. The activation function used in this case is the softmax function, which transforms the output scores into probabilities. The softmax activation ensures that the sum of the probabilities for all classes is equal to 1, making it suitable for multi-class classification tasks.

At this point, the variable `prediction` represents the final output of the neural network, containing the predicted probabilities for each class of the classification problem.

- `model = Model(inputs=r.input, outputs=prediction)`: This line of code creates a new Keras `Model` object. The `Model` class in Keras allows you to define a model by specifying its input and output tensors.

  Explanation:
  - `r.input`: This is the input tensor of the previously defined DenseNet201 model `r`. It represents the input layer of the model.
  - `prediction`: This is the output tensor defined in the previous lines, representing the final output of the neural network.

  By using `Model(inputs=r.input, outputs=prediction)`, you are defining a new model that takes the input from the `r` model and produces the output tensor `prediction`.

- `model.summary()`: This line of code displays a summary of the created model, showing the structure of the model along with the number of parameters in each layer.

  The `model.summary()` method provides a concise overview of the neural network architecture, including the layer type, output shape, and the number of trainable parameters. It's a useful tool to quickly inspect the model and ensure everything is set up correctly.

- `model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])`: This line of code compiles the previously defined Keras model.

  Explanation:
  - `loss='categorical_crossentropy'`: This specifies the loss function used during training. Categorical cross-entropy is commonly used for multi-class classification problems, where each input sample can belong to one of several classes. It measures the difference between the predicted probabilities and the true one-hot encoded labels.

  - `optimizer='adam'`: This specifies the optimizer used during training. The Adam optimizer is an adaptive learning rate optimization algorithm that is widely used in deep learning. It automatically adjusts the learning rate based on the gradients of the model's parameters, making it more effective in finding the optimal weights.

  - `metrics=['accuracy']`: This specifies the evaluation metric used during training and testing. In this case, 'accuracy' is used, which measures the percentage of correctly classified samples out of the total samples. It is a common metric for classification tasks.

  By calling `model.compile`, the model is configured for training. It sets up the loss function, the optimizer, and the evaluation metric for the neural network. Once compiled, the model is ready to be trained on a dataset using the `fit` method

- `from keras.preprocessing.image import ImageDataGenerator`: This line imports the `ImageDataGenerator` class from Keras, which is used for real-time data augmentation during training and data preprocessing.

- `train_datagen = ImageDataGenerator(rescale = 1./255, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)`: This block of code creates an `ImageDataGenerator` object for training data. It performs data augmentation to increase the diversity of the training dataset and prevent overfitting.

  - `rescale = 1./255`: This scales the pixel values of the input images to a range of [0, 1]. Dividing by 255 converts the pixel values from the original range [0, 255] to the [0, 1] range.

  - `shear_range = 0.2`: This applies random shear transformations to the images. Shearing shifts the pixels in a fixed direction, introducing a slant to the image.

  - `zoom_range = 0.2`: This applies random zoom-in or zoom-out transformations to the images.

  - `horizontal_flip = True`: This enables random horizontal flipping of the images. It provides more training samples by flipping the images horizontally.

- `test_datagen = ImageDataGenerator(rescale = 1./255)`: This block of code creates an `ImageDataGenerator` object for the test data. It performs data preprocessing for the test set but does not include data augmentation, as data augmentation should only be applied to the training set.

- `training_set = train_datagen.flow_from_directory('Training', target_size = (128, 128), batch_size = 32, class_mode = 'categorical')`: This line of code creates a data generator for the training set using the `flow_from_directory` method. It loads images from the 'Training' directory and applies the data augmentation defined in `train_datagen`.

  - `target_size = (128, 128)`: This specifies the target size of the images. All images will be resized to the dimensions (128, 128) during data loading.

  - `batch_size = 32`: This sets the batch size for training. The generator will yield batches of 32 images and their corresponding labels during each iteration.

  - `class_mode = 'categorical'`: This specifies the type of labels. Since this is a multi-class classification problem (four classes), 'categorical' class_mode is used. The labels will be one-hot encoded.

- `test_set = test_datagen.flow_from_directory('Testing', target_size = (128, 128), batch_size = 32, class_mode = 'categorical')`: This line of code creates a data generator for the test set using the `flow_from_directory` method. It loads images from the 'Testing' directory and applies the rescaling defined in `test_datagen`.

  - `target_size = (128, 128)`: This specifies the target size of the images. All images will be resized to the dimensions (128, 128) during data loading.

  - `batch_size = 32`: This sets the batch size for testing. The generator will yield batches of 32 images and their corresponding labels during each iteration.

  - `class_mode = 'categorical'`: This specifies the type of labels. Since this is a multi-class classification problem (four classes), 'categorical' class_mode is used. The labels will be one-hot encoded.

The `ImageDataGenerator` objects `training_set` and `test_set` allow you to efficiently load and preprocess batches of images during model training and evaluation. They also perform data augmentation for the training set to enhance the model's generalization capability.

- `from tensorflow import keras`: This line imports the `keras` module from TensorFlow, which is used to access various functionalities for building and training deep learning models.

- `early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)`: This line creates an `EarlyStopping` callback using the `keras.callbacks` module. The `EarlyStopping` callback is used during model training to stop the training process early if a certain condition is met.

  - `monitor='val_loss'`: This specifies the metric to monitor for early stopping. In this case, it monitors the validation loss (`val_loss`). The training process will stop early if the validation loss stops decreasing.

  - `patience=10`: This parameter determines the number of epochs with no improvement in the monitored metric before the training process stops. In this case, if there is no improvement in the validation loss for 10 consecutive epochs, the training will stop early.

The `EarlyStopping` callback is useful for preventing overfitting and optimizing training time. It helps to stop training once the model's performance on the validation set has reached its peak and starts to degrade.

This callback can be passed to the `fit` method of a Keras model as a list, allowing the model to use it during the training process.

- `r = model.fit(`: This line of code starts the training process of the Keras model using the `fit` method.

- `training_set`: This is the data generator for the training set obtained from `ImageDataGenerator`. It provides batches of training data during each epoch of training.

- `validation_data=test_set`: This specifies the data generator for the validation set. It provides batches of validation data during each epoch of training. The validation data comes from the `test_set`, which was also created using the `ImageDataGenerator`.

- `epochs=10`: This parameter determines the number of epochs (complete passes through the entire training dataset) for which the model will be trained.

- `steps_per_epoch=len(training_set)`: This sets the number of steps (batches) to be processed in each epoch. It is equal to the total number of samples in the training set divided by the batch size.

- `validation_steps=len(test_set)`: This sets the number of steps (batches) to be processed in each validation epoch. It is equal to the total number of samples in the test set divided by the batch size.

- `callbacks=[early_stop]`: This specifies the list of callbacks to be used during training. In this case, the `early_stop` callback created previously will be used for early stopping based on the validation loss.

The `model.fit()` method performs the actual training of the model using the specified data generators and settings. It iterates through the training data for the specified number of epochs, updating the model's weights to minimize the loss function.

During the training process, the model's performance on the validation set will also be monitored. If the validation loss stops decreasing or no improvement is observed for the specified number of epochs (as determined by the `early_stop` callback), the training process will stop early.

After the training is complete, the history of the training process (e.g., loss and accuracy values) will be stored in the variable `r`.

- `plt.plot(r.history['loss'], label='train loss')`: This line of code plots the training loss values over the epochs. It accesses the training loss values from the `r.history` dictionary, which contains the training history returned by the `model.fit()` method. The `'loss'` key corresponds to the training loss.

- `plt.plot(r.history['val_loss'], label='val loss')`: This line of code plots the validation loss values over the epochs. It accesses the validation loss values from the `r.history` dictionary, which also contains the validation loss returned by the `model.fit()` method. The `'val_loss'` key corresponds to the validation loss.

- `plt.legend()`: This line of code adds a legend to the plot, displaying labels for the two lines ('train loss' and 'val loss').

- `plt.show()`: This displays the plot on the screen.

- `plt.savefig('LossVal_loss')`: This saves the plot as an image file named 'LossVal_loss'. The `savefig` function allows you to save the plot as an image in various formats (e.g., PNG, JPG, PDF) for later use or visualization.

The provided code creates a plot that shows the training loss and validation loss over the epochs during the model training process. This visualization helps to assess the model's performance and check for any overfitting or underfitting issues. If both training loss and validation loss decrease together, it indicates that the model is learning well. However, if the validation loss starts increasing while the training loss continues to decrease, it may indicate overfitting.

The `plt.show()` function is used to display the plot, and `plt.savefig()` is used to save the plot as an image.

- `plt.plot(r.history['accuracy'], label='train accuracy')`: This line of code plots the training accuracy values over the epochs. It accesses the training accuracy values from the `r.history` dictionary, which contains the training history returned by the `model.fit()` method. The `'accuracy'` key corresponds to the training accuracy.

- `plt.plot(r.history['val_accuracy'], label='val accuracy')`: This line of code plots the validation accuracy values over the epochs. It accesses the validation accuracy values from the `r.history` dictionary, which also contains the validation accuracy returned by the `model.fit()` method. The `'val_accuracy'` key corresponds to the validation accuracy.

- `plt.legend()`: This line of code adds a legend to the plot, displaying labels for the two lines ('train accuracy' and 'val accuracy').

- `plt.show()`: This displays the plot on the screen.

- `plt.savefig('AccVal_acc')`: This saves the plot as an image file named 'AccVal_acc'. The `savefig` function allows you to save the plot as an image in various formats (e.g., PNG, JPG, PDF) for later use or visualization.

The updated code creates a plot that shows the training accuracy and validation accuracy over the epochs during the model training process. This visualization helps to assess the model's performance and convergence during training. A consistent increase in both training and validation accuracy indicates that the model is learning effectively.

Just like before, the `plt.show()` function is used to display the plot, and `plt.savefig()` is used to save the plot as an image.



















































  
