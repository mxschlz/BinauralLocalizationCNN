import tensorflow as tf
import numpy as np


def flip(x: tf.Tensor) -> tf.Tensor:
    """Flip augmentation

    Args:
        x: Image to flip

    Returns:
        Augmented image
    """
    x = tf.image.random_flip_left_right(x)
    x = tf.image.random_flip_up_down(x)

    return x


def rotate(x: tf.Tensor) -> tf.Tensor:
    """Rotation augmentation

    Args:
        x: Image

    Returns:
        Augmented image
    """

    return tf.image.rot90(x, tf.random_uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))


def zoom(x: tf.Tensor) -> tf.Tensor:
    """Zoom augmentation

    Args:
        x: Image

    Returns:
        Augmented image
    """

    # Generate 20 crop settings, ranging from a 1% to 20% crop.
    scales = list(np.arange(0.8, 1.0, 0.01))
    boxes = np.zeros((len(scales), 4))

    for i, scale in enumerate(scales):
        x1 = y1 = 0.5 - (0.5 * scale)
        x2 = y2 = 0.5 + (0.5 * scale)
        boxes[i] = [x1, y1, x2, y2]

    def random_crop(img):
        # Create different crops for an image
        crops = tf.image.crop_and_resize([img], boxes=boxes, box_ind=np.zeros(len(scales)), crop_size=(32, 32))
        # Return a random crop
        return crops[tf.random_uniform(shape=[], minval=0, maxval=len(scales), dtype=tf.int32)]

    choice = tf.random_uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)

    # Only apply cropping 50% of the time
    return tf.cond(choice < 0.5, lambda: x, lambda: random_crop(x))


if __name__ == "__main__":
    import glob

    import glob
    from CNN_util import get_feature_dict, build_tfrecords_iterator
    import tensorflow as tf

    # training data path
    train_data_path = "tfrecords/msl/*train.tfrecords"
    # get all files
    stim_files = glob.glob(train_data_path)
    # get feature dict
    stim_feature = get_feature_dict(stim_files[0])
    # get the stim dataset
    stim_dset = build_tfrecords_iterator(train_data_path, stim_feature)
    stim_iter = stim_dset.make_initializable_iterator()
    data_samp = stim_iter.get_next()

    augmentations = [flip, zoom, rotate]

    for f in augmentations:
        dataset = data_samp.map(lambda x: tf.cond(tf.random_uniform([], 0, 1) > 0.75, lambda: f(x), lambda: x),
                                num_parallel_calls=4)
    dataset = stim_dset.map(lambda x: tf.clip_by_value(x, 0, 1))


