import tensorflow as tf
import string

BATCH_SIZE = 50
MAX_WORDS_IN_REVIEW = 250  # Maximum length of a review to consider
EMBEDDING_SIZE = 50  # Dimensions for each word vector

stop_words = set({'ourselves', 'hers', 'between', 'yourself', 'again',
                  'there', 'about', 'once', 'during', 'out', 'very', 'having',
                  'with', 'they', 'own', 'an', 'be', 'some', 'for', 'do', 'its',
                  'yours', 'such', 'into', 'of', 'most', 'itself', 'other',
                  'off', 'is', 's', 'am', 'or', 'who', 'as', 'from', 'him',
                  'each', 'the', 'themselves', 'below', 'are', 'we',
                  'these', 'your', 'his', 'through', 'don', 'me', 'were',
                  'her', 'more', 'himself', 'this', 'down', 'should', 'our',
                  'their', 'while', 'above', 'both', 'up', 'to', 'ours', 'had',
                  'she', 'all', 'no', 'when', 'at', 'any', 'before', 'them',
                  'same', 'and', 'been', 'have', 'in', 'will', 'on', 'does',
                  'yourselves', 'then', 'that', 'because', 'what', 'over',
                  'why', 'so', 'can', 'did', 'not', 'now', 'under', 'he', 'you',
                  'herself', 'has', 'just', 'where', 'too', 'only', 'myself',
                  'which', 'those', 'i', 'after', 'few', 'whom', 't', 'being',
                  'if', 'theirs', 'my', 'against', 'a', 'by', 'doing', 'it',
                  'how', 'further', 'was', 'here', 'than', 'br'})


def strip_punctuation(input):
    return ''.join(c for c in input if c not in string.punctuation)
  
def preprocess(review):
  
    """
    Apply preprocessing to a single review. You can do anything here that is manipulation
    at a string level, e.g.
        - removing stop words
        - stripping/adding punctuation
        - changing case
        - word find/replace
    RETURN: the preprocessed review in string form.
    """
    processed_review = strip_punctuation(review.lower())

    for word in stop_words:
        processed_review = processed_review.replace(' ' + word + ' ', ' ')

    processed_review = processed_review.split()

    return processed_review


def define_graph():
    """
    Implement your model here. You will need to define placeholders, for the input and labels,
    Note that the input is not strings of words, but the strings after the embedding lookup
    has been applied (i.e. arrays of floats).

    In all cases this code will be called by an unaltered runner.py. You should read this
    file and ensure your code here is compatible.

    Consult the assignment specification for details of which parts of the TF API are
    permitted for use in this function.

    You must return, in the following order, the placeholders/tensors for;
    RETURNS: input, labels, optimizer, accuracy and loss
    """

    num_units = 128
    num_layers = 2

    input_data = tf.placeholder(tf.float32, [BATCH_SIZE, MAX_WORDS_IN_REVIEW, EMBEDDING_SIZE], name="input_data")
    labels = tf.placeholder(tf.float32, [BATCH_SIZE, 2], name="labels")

    dropout_keep_prob = tf.placeholder_with_default(input=tf.constant(1.0, shape=[]), shape=[])

    # GRU 1 Layer
    """
    GRUCell = tf.contrib.rnn.GRUCell(num_units=num_units)
    GRUCell = tf.contrib.rnn.DropoutWrapper(cell=GRUCell, output_keep_prob=dropout_keep_prob)
    cell = tf.contrib.rnn.MultiRNNCell([GRUCell])
    """

    # GRU 2 Layer
    """
    layers = [tf.contrib.rnn.DropoutWrapper(cell=tf.contrib.rnn.GRUCell(num_units = num_units), output_keep_prob=dropout_keep_prob) for _ in range(num_layers)]
    cell = tf.nn.rnn_cell.MultiRNNCell(layers)
    """

    # LSTM 1 Layer
    """
    lstmCell = tf.contrib.rnn.BasicLSTMCell(num_units=num_units)
    lstmCell = tf.contrib.rnn.DropoutWrapper(cell=lstmCell, output_keep_prob=dropout_keep_prob)
    cell = tf.contrib.rnn.MultiRNNCell([lstmCell])
    """

    # LSTM 2 Layer
    """
    layers = [tf.contrib.rnn.DropoutWrapper(cell=tf.contrib.rnn.BasicLSTMCell(num_units = num_units), output_keep_prob=dropout_keep_prob) for _ in range(num_layers)]
    cell = tf.nn.rnn_cell.MultiRNNCell(layers)
    """

    layers = [tf.contrib.rnn.DropoutWrapper(cell=tf.contrib.rnn.GRUCell(num_units = num_units, activation=tf.nn.relu), output_keep_prob=dropout_keep_prob) for _ in range(num_layers)]
    cell = tf.nn.rnn_cell.MultiRNNCell(layers)

    value, _ = tf.nn.dynamic_rnn(cell, input_data, dtype=tf.float32)

    w = tf.Variable(tf.truncated_normal([num_units, 2], stddev=2.0/tf.sqrt(float(num_units))))
    b = tf.Variable(tf.constant(0.1, shape=[2]))

    value = tf.transpose(value, [1, 0, 2])
    last = tf.gather(value, int(value.get_shape()[0]) - 1)
    prediction = (tf.matmul(last, w) + b)

    correctPred = tf.equal(tf.argmax(prediction,1), tf.argmax(labels,1))
    Accuracy = tf.reduce_mean(input_tensor=tf.cast(correctPred, tf.float32), name="accuracy")

    loss = tf.reduce_mean(input_tensor=tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=labels), name="loss")
    optimizer = tf.train.AdamOptimizer().minimize(loss)

    return input_data, labels, dropout_keep_prob, optimizer, Accuracy, loss
