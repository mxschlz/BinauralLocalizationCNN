import numpy as np
import tensorflow as tf


# Function to apply random horizontal flip
def random_flip(image):
    return tf.image.random_flip_left_right(image)


def random_gaussian_noise(image):
    noise = tf.random.normal(shape=tf.shape(image), mean=0.0, stddev=0.1)
    return tf.clip_by_value(image + noise, 0.0, 1.0)


def random_contrast(image):
    return tf.image.random_contrast(image, lower=0.5, upper=1.5)


def random_brightness(image):
    return tf.image.random_brightness(image, max_delta=0.2)


# Create a function to apply a random augmentation
def apply_random_augmentation(image):
    # Randomly choose an augmentation function to apply
    augmentation_functions = [random_flip, random_gaussian_noise, random_brightness, random_contrast]
    randint = np.random.randint(len(augmentation_functions))

    # Apply the chosen augmentation function
    image = augmentation_functions[randint](image)
    print(f"Applying Augmentation Operation: {augmentation_functions[randint]}")
    return image


if __name__ == "__main__":
    import glob

    import glob
    from CNN_util import get_feature_dict, build_tfrecords_iterator

    # training data path
    train_data_path = "*train*.tfrecords"
    # get all files
    stim_files = glob.glob(train_data_path)
    # get feature dict
    stim_feature = get_feature_dict(stim_files[0])
    # get the stim dataset
    stim_dset = build_tfrecords_iterator(train_data_path, stim_feature)
    stim_iter = stim_dset.make_initializable_iterator()
    data_samp = stim_iter.get_next()
    image = data_samp["train/image"]
    label = data_samp["train/binary_label"]
    # Apply data augmentation to your dataset
    data = random_flip(image)
