# define the global parameters for training and reading into the subsequent programmes
"""
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('batch_size', 32,
                            'Number of images to process in a batch.')
tf.app.flags.DEFINE_integer('image_size', 299,
                           'Provide square images of this size.')
tf.app.flags.DEFINE_integer('num_preprocess_threads', 4,
                            'Number of preprocessing threads per tower.'
                            'Please make this a multiple of 4.')
tf.app.flags.DEFINE_integer('num_readers', 4,
                            'Number of parallel readers during train.')

# Images are preprocessed asynchronously using multiple threads specified by
# --num_preprocss_threads and the resulting processed images are stored in a
# random shuffling queue. The shuffling queue dequeues --batch_size images
# for processing on a given Inception tower. A larger shuffling queue guarantees
# better mixing across examples within a batch and results in slightly higher
# predictive performance in a trained model. Empirically,
# --input_queue_memory_factor=16 works well. A value of 16 implies a queue size
# of 1024*16 images. Assuming RGB 299x299 images, this implies a queue size of
# 16GB. If the machine is memory limited, then decrease this factor to
# decrease the CPU memory footprint, accordingly.
tf.app.flags.DEFINE_integer('input_queue_memory_factor', 16,
                            'Size of the queue of preprocessed images. '
                            'Default is ideal but try smaller values, e.g. '
                            '4, 2 or 1, if host memory is constrained. See '
                            'comments in code for more details.')

"""

batch_size=32
image_size=299
num_preprocess_threads = 4
num_readers=4
input_queue_memory_factor =16

def print_all():
    print """Asynchronously process images using multiple threads specified by
the following parameters and the resulting processed images are stored in a
random shuffling queue.
    """
    print 'FLAGS.batch_size = %d' %batch_size
    print 'FLAGS.image_size = %d' %image_size
    print 'FLAGS.num_preprocess_threads = %d' %num_preprocess_threads
    print 'FLAGS.num_readers = %d' %num_readers
    print 'FLAGS.input_queue_memory_factor = %d' %input_queue_memory_factor
