import tensorflow as tf


def read_data(filename_queue):
    reader = tf.TextLineReader()
    key, record_string = reader.read(filename_queue)
    record_defaults = [[1], [1], [1]]
    col1, col2, col3 = tf.decode_csv(record_string, record_defaults=record_defaults)
    features = tf.pack([col1, col2])
    label = col3
    return features, label


def input_pipeline(filenames, batch_size, num_epochs=None):
    filename_queue = tf.train.string_input_producer(filenames,
                                                    num_epochs=num_epochs,
                                                    shuffle=False)
    example, label = read_data(filename_queue)
    # min_after_dequeue defines how big a buffer we will randomly sample
    #   from -- bigger means better shuffling but slower start up and more
    #   memory used.
    # capacity must be larger than min_after_dequeue and the amount larger
    #   determines the maximum we will prefetch.  Recommendation:
    #   min_after_dequeue + (num_threads + a small safety margin) * batch_size
    min_after_dequeue = 10000
    capacity = min_after_dequeue + 3 * batch_size
    example_batch, label_batch = tf.train.shuffle_batch([example, label],
                                                        batch_size=batch_size,
                                                        capacity=capacity,
                                                        min_after_dequeue=min_after_dequeue)
    return example_batch, label_batch


with tf.Session() as sess:

    example, label = input_pipeline(filenames=["../Databases/ratings100k.csv"],
                                    batch_size=10)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    for i in range(1200):
        # Retrieve a batch instance:
        example_array, label_array = sess.run([example, label])
        print(example_array)
        print(label_array)
        coord.request_stop()
        coord.join(threads)