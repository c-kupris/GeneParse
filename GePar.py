"""Goal - Build an RNN model that will elucidate the relationships between genes known to be involved in lung cancer
pathogenesis. In this specific case, for training purposes, data will be taken from here:
https://www.ncbi.nlm.nih.gov/gene/?term=lung+cancer."""

# imports
import keras
import numpy
import os
import tensorflow  # Uses version 2.16.1

print("Device CPU count: " + os.cpu_count().__str__())

directory = "/Users/chandlerkupris/Desktop/GeneticDeterminantsOfLungCancer"

# Load training data file (stored locally at the fname and cache_dir name below).
lung_gene_training_data_file = keras.utils.get_file(fname="gene_result.txt",
                                                    origin="https://www.ncbi.nlm.nih.gov/gene/?term=lung+cancer")

lung_gene_training_data = keras.utils.text_dataset_from_directory(directory=directory)

lung_gene_training_data = tensorflow.data.Dataset.zip(lung_gene_training_data)

batch_size = 53
while batch_size < 70543:
    lung_gene_training_data = lung_gene_training_data.batch(batch_size=batch_size, drop_remainder=True)

print("Lung gene training dataset: " + str(lung_gene_training_data))

# Create a Keras Sequential model.
model = keras.Sequential()

# Define the hyperparamters for the model.
epochs = 1
steps_per_epoch = 1

# Make the layers to add to the model.
input_layer = keras.layers.InputLayer(input_shape=(11, 121, 53),
                                      batch_size=batch_size)

'''embedding_layer = keras.layers.Embedding(mask_zero=True,
                                         input_dim=70543,
                                         output_dim=92160)'''

lstm_cell_one = keras.layers.LSTMCell(units=512)

# You need to place the LSTMCells in (their own) RNN layers (the LSTMCells are just that - cells, not layers).
rnn_layer_one = keras.layers.RNN(cell=lstm_cell_one)

output_layer = keras.layers.Layer()

# Add the layers to the model.
model.add(input_layer)
# model.add(embedding_layer)
model.add(rnn_layer_one)
model.add(output_layer)

model.build(input_shape=(batch_size, 11, 121, 53))

model.summary()

loss = keras.losses.CategoricalCrossentropy()

optimizer = keras.optimizers.Lion()

metrics = [keras.metrics.CategoricalAccuracy()]

# Compile the model.
model.compile(optimizer=optimizer,
              loss=loss,
              metrics=metrics,
              run_eagerly=True)

classes = numpy.arange(start=0,
                       stop=53,
                       step=1)

for x in lung_gene_training_data:
    if x is tensorflow.string:
        x = classes[lung_gene_training_data[x]]
        encoded_data = keras.utils.to_categorical(x, classes)
        one_hot_encoded_data = tensorflow.data.Dataset.zip(encoded_data)
        print("one-hot encoded data: " + str(one_hot_encoded_data))
        print(x)

print(lung_gene_training_data)

model.fit(x=lung_gene_training_data,
          epochs=epochs,
          steps_per_epoch=steps_per_epoch)
