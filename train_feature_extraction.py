import pickle
import tensorflow as tf
from sklearn.model_selection import train_test_split
from alexnet import AlexNet
import time

nb_classes =43
epochs = 15
BATCH_SIZE = 100
# TODO: Load traffic signs data.
training_file = "./traffic-signs-data/train.p"
with open(training_file, mode='rb') as f:
    train = pickle.load(f)

X_train, y_train = train['features'], train['labels']

# TODO: Split data into training and validation sets.
import tensorflow as tf
from sklearn.utils import shuffle

X_train, y_train = shuffle(X_train, y_train)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.1 , random_state=30)


# TODO: Define placeholders and resize operation.
features = tf.placeholder(tf.float32,(None,32,32,3) )
labels= tf.placeholder(tf.int64, None)
resized = tf.image.resize_images(features, (227,227))  


# TODO: pass placeholder as first argument to `AlexNet`.
fc7 = AlexNet(resized, feature_extract=True)
# NOTE: `tf.stop_gradient` prevents the gradient from flowing backwards
# past this point, keeping the weights before and up to `fc7` frozen.
# This also makes training faster, less work to do!
fc7 = tf.stop_gradient(fc7)
shape = (fc7.get_shape().as_list()[-1], nb_classes)

# TODO: Add the final layer for traffic sign classification.
fc8W = tf.Variable(tf.truncated_normal(shape, stddev= 0.01))
fc8b = tf.Variable(tf.zeros(nb_classes))
logits = tf.nn.xw_plus_b(fc7, fc8W, fc8b)


# TODO: Define loss, training, accuracy operations.
# HINT: Look back at your traffic signs project solution, you may
# be able to reuse some the code.
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer()
training_operation = optimizer.minimize(loss_operation, var_list=[fc8W, fc8b])
# TODO: Train and evaluate the feature extraction model.
init_op = tf.global_variables_initializer()
preds = tf.arg_max(logits, 1)
accuracy_op = tf.reduce_mean(tf.cast(tf.equal(preds, labels), tf.float32))

def evaluate(X_data, y_data, sess):
    num_examples = X_data.shape[0]
    total_accuracy = 0
    total_loss = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        loss_op, accuracy = sess.run([loss_operation, accuracy_op], feed_dict={features: batch_x, labels: batch_y})
        total_accuracy += (accuracy * len(batch_x))
        total_loss += (loss_op*length(batch_x))
    return total_loss / num_examples, total_accuracy / num_examples
    
with tf.Session() as sess:
    sess.run(init_op)

    for i in range(epochs):
        # training
        X_train, y_train = shuffle(X_train, y_train)
        t0 = time.time()
        for offset in range(0, X_train.shape[0], BATCH_SIZE):
            end = offset + BATCH_SIZE
            sess.run(training_operation, feed_dict={features: X_train[offset:end], labels: y_train[offset:end]})

        val_loss, val_acc = evaluate(X_valid, y_valid, sess)
        print("Epoch", i+1)
        print("Time: %.3f seconds" % (time.time() - t0))
        print("Validation Loss =", val_loss)
        print("Validation Accuracy =", val_acc)
        print("")