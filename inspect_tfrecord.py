import json
import io
import sys
import warnings
from functools import lru_cache
from pathlib import Path
from pprint import PrettyPrinter
import json
from collections import Counter
from typing import Tuple, List

import scipy as sp
import tensorflow as tf
from tensorflow.core.example.feature_pb2 import Feature
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

from persistent_cache import persistent_cache


# features = {
#     "train/image": tf.train.Feature(bytes_list=tf.train.BytesList(value=[])),
#     "train/image_width": tf.train.Feature(int64_list=tf.train.Int64List(value=[])),
#     "train/image_height": tf.train.Feature(int64_list=tf.train.Int64List(value=[])),
#     "train/elev": tf.train.Feature(int64_list=tf.train.Int64List(value=[])),
#     "train/azim": tf.train.Feature(int64_list=tf.train.Int64List(value=[]))
# }

# Recursively generate pretty print of arbitrary objects
def generate_pprint(obj, level_indent="  ", max_depth=None, verbose_output=True,
                    justify_output=True, prevent_loops=True, prevent_revisit=False,
                    explore_objects=True, excluded_ids=[], visited_ids=[],
                    path_ids=[], current_depth=0, suppress_none_values=True):
    """Recursively generates pretty print of arbitrary objects.

    Recursively generates pretty print of contents of arbitrary objects. Contents
    are represented as lines containing key-value pairs. Recursion may be affected
    by various parameters (see below).

    Arguments:
        max_depth:       Maximum allowed depth of recursion. If depth is exceeded,
                         recursion is stopped.
        verbose_output:  Produce verbose output. Adds some additional details for
                         certain data types
        justify_output:  Justify output. Produces output in block-like appearance
                         with equal spacing between keys and values.
        prevent_loops:   Detect and prevent recursion loops by keeping track of
                         already visited objects within current recursion path.
        prevent_revisit: Detect and prevent revisiting of already visited objects.
                         While 'prevent_loops' prevents revisiting objects only
                         within one single recursion path, this prevents revisiting
                         objects globally across all recursion paths.
        explore_objects: Explore (i.e. recurse into) arbitrary objects. If enabled,
                         arbitrary objects not matching base types are explored.
                         If disabled, only certain types of objects are explored
                         (tuple, list, dict, set/frozenset). Note that this does
                         not affect the initially provided object (which is always
                         explored).
        excluded_ids:    List of object IDs to exclude from exploration (i.e. re-
                         cursion). Recursion is stopped if object with matching
                         ID is encountered.
        suppress_none_values: Leave out None values and objects whose values are
                         all None.
        visited_ids,     Internal variables used to control recursion flow, loop
        path_ids,        detection and revisit detection. Never provide or modify
        current_depth:   these!

    Returns:
        Generated pretty print output as list of lines (strings).

    Raises:
        TypeError:       Object or value has unsupported type (should never occur)
        AssertionError:  Assertion failed, most likely exposing a bug (should never
                         occur)
    """

    output = []
    indent = level_indent * current_depth

    # Check if object has already been visited within current recursion path.
    # If so, we encoutered a loop and thus need to break off recursion. If
    # not, continue and add object to list of visited objects within current
    # recursion path
    if (prevent_loops == True):
        if (id(obj) in path_ids):
            output.append(indent + "<recursion loop detected>")
            return output
        path_ids.append(id(obj))

    # Check if object has already been visited. If so, we're not going to visit
    # it again and break off recursion. If not, continue and add current object
    # to list of visited objects
    if (prevent_revisit == True):
        if (id(obj) in visited_ids):
            output.append(indent + "<item already visited>")
            return output
        visited_ids.append(id(obj))

    # Check if maximum allowed depth of recursion has been exceeded. If so, break
    # off recursion
    if (max_depth != None and current_depth > max_depth):
        output.append(indent + "<recursion limit reached>")
        return output

    # Check if object is supposed to be excluded. If so, break of recursion
    if (id(obj) in excluded_ids):
        output.append(indent + "<item is excluded>")
        return output

    # Determine keys and associated values
    if (isinstance(obj, dict)):
        keys = obj.keys()
        values = obj
    elif (isinstance(obj, tuple) or isinstance(obj, list)):
        keys = range(0, len(obj))
        values = obj
    elif (isinstance(obj, set) or isinstance(obj, frozenset)):
        keys = range(0, len(obj))
        values = [item for item in obj]
    elif (isinstance(obj, object)):
        keys = [item for item in dir(obj) if (not item.startswith("_"))]
        values = {key: getattr(obj, key) for key in keys}
    else:  # should never occur as everything in Python is an 'object' and should be caught above, but better be safe than sorry
        raise TypeError("unsupported object type: '%s'" % type(obj))

    # Define key string templates. If output is to be justified, determine maximum
    # length of key string and adjust templates accordingly
    kstmp1 = kstmp2 = "%s"
    if (justify_output == True):
        maxlen = 0
        for key in keys:
            klen = len(str(key))
            if (klen > maxlen):
                maxlen = klen
        kstmp1 = "%-" + str(maxlen + 3) + "s"  # maxlen+3: surrounding single quotes + trailing colon
        kstmp2 = "%-" + str(maxlen + 1) + "s"  # maxlen+1: trailing colon

    intermediate_output = []
    # Process keys and associated values
    for key in keys:
        value = values[key]

        # Generate key string
        keystr = kstmp1 % ("'" + str(key) + "':") if (isinstance(obj, dict) and isinstance(key, str)) else kstmp2 % (
                str(key) + ":")

        # Generate value string
        valstr = ""
        exp_obj = False
        if (isinstance(value, dict)):
            valstr = "<dict, %d items, class '%s'>" % (len(value), type(value).__name__) if (
                    verbose_output == True) else "<dict, %d items>" % len(value)
        elif (isinstance(value, tuple)):
            valstr = "<tuple, %d items, class '%s'>" % (len(value), type(value).__name__) if (
                    verbose_output == True) else "<tuple, %d items>" % len(value)
        elif (isinstance(value, list)):
            valstr = "<list, %d items, class '%s'>" % (len(value), type(value).__name__) if (
                    verbose_output == True) else "<list, %d items>" % len(value)
        elif (isinstance(value, set)):  # set and frozenset are distinct
            valstr = "<set, %d items, class '%s'>" % (len(value), type(value).__name__) if (
                    verbose_output == True) else "<set, %d items>" % len(value)
        elif (isinstance(value, frozenset)):  # set and frozenset are distinct
            valstr = "<frozenset, %d items, class '%s'>" % (len(value), type(value).__name__) if (
                    verbose_output == True) else "<frozenset, %d items>" % len(value)
        elif (isinstance(value, range)):
            valstr = "<range, start %d, stop %d, step %d>" % (value.start, value.stop, value.step) if (
                    verbose_output == True) else "<range(%d,%d,%d)>" % (value.start, value.stop, value.step)
        elif (isinstance(value, bytes)):
            valstr = "<bytes, %d bytes>" % len(value)
        elif (isinstance(value, bytearray)):
            valstr = "<bytearray, %d bytes>" % len(value)
        elif (isinstance(value, memoryview)):
            valstr = "<memoryview, %d bytes, object %s>" % (len(value), type(value.obj).__name__) if (
                    verbose_output == True) else "<memoryview, %d bytes>" % len(value)
        elif (isinstance(value, bool)):  # needs to be above int as 'bool' also registers as 'int'
            valstr = "%s" % value
        elif (isinstance(value, int)):
            valstr = "%d (0x%x)" % (value, value)
        elif (isinstance(value, float)):
            valstr = "%s" % str(
                value)  # str(value) provides best representation; alternatives: '%e|%E|%f|%F|%g|%G' % value
        elif (isinstance(value, complex)):
            valstr = "%s" % str(value)
        elif (isinstance(value, str)):
            # valstr = "%s" % repr(value) # using repr(value) to encode escape sequences (e.g. '\n' instead of actual newline); repr() adds surrounding quotes for strings (style based on contents)
            # valstr = "%r" % value # alternative, seems to be the same as repr(value)
            valstr = "'%s'" % repr(value)[1:-1]  # this seems to be the only way to always get a single-quoted string
        elif (value == None):
            valstr = "None"
        elif isinstance(value,
                        type):  # checks if object is 'class' (https://stackoverflow.com/a/10123520/1976617); needs to be above 'callable' as 'class' also registers as 'callable'
            # valstr = "<class '%s'>" % type(value).__name__ if (verbose_output == True) else "<class>"
            valstr = "<class '%s.%s'>" % (value.__module__, value.__name__) if (verbose_output == True) else "<class>"
        elif (
                callable(
                    value)):  # catches everything callable, i.e. functions, methods, classes (due to constructor), etc.
            # valstr = "<callable, %s>" % repr(value)[1:-1] if (verbose_output == True) else "<callable>"
            # valstr = "<callable, class '%s'>" % type(value).__name__ if (verbose_output == True) else "<callable>"
            valstr = "<callable, class '%s.%s'>" % (value.__class__.__module__, value.__class__.__name__) if (
                    verbose_output == True) else "<callable>"
        elif (
                isinstance(value,
                           object)):  # this has to be last in line as *everything* above also registers as 'object'
            # valstr = "<object, class '%s'>" % type(value).__name__ if (verbose_output == True) else "<object>"
            valstr = "<object, class '%s.%s'>" % (value.__class__.__module__, value.__class__.__name__) if (
                    verbose_output == True) else "<object>"
            if (explore_objects == True):
                exp_obj = True  # this ensures we only explore objects that do not represent a base type (i.e. everything listed above)
        else:  # should never occur as everything in Python is an 'object' and should be caught above, but better be safe than sorry
            # valstr = "'%s'" % str(value) if (verbose_output == True) else str(value)
            raise TypeError("unsupported value type: '%s'" % type(value))

        # Explore value object recursively if it meets certain criteria
        if (isinstance(value, dict) or isinstance(value, tuple) or isinstance(value, list) or isinstance(value, set) or
                isinstance(value, frozenset) or (isinstance(value, object) and exp_obj == True)):
            # These may be used to prevent recursion beforehand, i.e. before calling this
            # function again, as an alternative to checks at beginning of function. Leaving
            # this here for future reference
            # if (prevent_loops == True and id(value) in path_ids):
            #    #output[-1] += " <recursion loop detected>"
            #    output[-1] += " <recursion loop>"
            #    continue
            # if (prevent_revisit == True and id(value) in visited_ids):
            #    #output[-1] += " <item already visited>"
            #    output[-1] += " <already visited>"
            #    continue
            # if (max_depth != None and current_depth+1 > max_depth):
            #    #output[-1] += " <recusion limit reached>"
            #    output[-1] += " <recursion limit>"
            #    continue
            # if (id(value) in excluded_ids):
            #    #output[-1] += " <item is excluded>"
            #    output[-1] += " <item excluded>"
            #    continue
            recursion_output = generate_pprint(value, level_indent=level_indent, max_depth=max_depth,
                                      verbose_output=verbose_output,
                                      justify_output=justify_output, prevent_loops=prevent_loops,
                                      prevent_revisit=prevent_revisit,
                                      explore_objects=explore_objects, excluded_ids=excluded_ids,
                                      visited_ids=visited_ids,
                                      path_ids=path_ids, current_depth=current_depth + 1)
            if recursion_output:
                intermediate_output.extend(recursion_output)
        elif (suppress_none_values and valstr != "None") or not suppress_none_values:
            intermediate_output.append(indent + keystr + " " + valstr)

    if intermediate_output:
        output.extend(intermediate_output)

    # Remove object from list of visited objects within current recursion path
    # (part of recursion loop detection; this 'rolls back' the laid out path)
    if (prevent_loops == True):
        assert len(path_ids) > 0 and path_ids[-1] == id(
            obj), "last item in list of path objects not existing or not matching object"
        path_ids.pop()

    # Return generated output
    return output


# Convenience wrapper for generate_pprint()
def print_pprint(*args, **kwargs):
    output = generate_pprint(*args, **kwargs)
    for line in output:
        print(line)


@persistent_cache
def extract_elev_azim(dataset: tf.data.TFRecordDataset) -> (List, List):
    # assert key is not None, 'Key must be provided for caching'
    elevs = []
    azims = []
    for raw_record in tqdm(dataset, desc='Extracting elevations and azimuths'):
        example = tf.train.Example()
        example.ParseFromString(raw_record.numpy())
        # Extract the label from the example
        elev = example.features.feature['train/elev'].int64_list.value[0]
        azim = example.features.feature['train/azim'].int64_list.value[0]

        elevs.append(elev)
        azims.append(azim)
    return elevs, azims


def inspect_data(path: Path):
    dataset = tf.data.TFRecordDataset(path, compression_type="GZIP")
    elevs, azims = extract_elev_azim(dataset, persistent_cache_key=path)

    print('#################################################################')
    print('#################### .tfrecord data analysis ####################')
    print('#################################################################')
    print(f'Dataset path: {path}')
    print(f'Contains {dataset.cardinality()} examples (-1 is infinite cardinality, -2 is unknown cardinality)')
    print(f'\nOptions (showing only populated values): ')
    print_pprint(dataset.options(), suppress_none_values=True)

    labels = np.array([elevs, azims]).T
    print(f'\nFound {len(labels)} examples')
    unique_labels, unique_label_counts = np.unique(labels, axis=0, return_counts=True)
    # Calculate smallest elevation step
    elev_step = np.min(np.diff(np.unique(elevs)))
    print(f'\nExtracted elev range (min / max / smallest step): {min(elevs)} / {max(elevs)} / {elev_step}')
    possible_elevs = (max(elevs) - min(elevs)) // elev_step + 1
    print(f'Possible elevs: {possible_elevs}')
    print(f'Elevs found: {len(np.unique(elevs))}')
    missing_elevs = [elev for elev in range(min(elevs), max(elevs) + 1, elev_step) if elev not in elevs]
    print(f'Missing elevations: {missing_elevs}')

    azim_step = np.min(np.diff(np.unique(azims)))
    print(f'\nExtracted azim range (min / max / smallest step): {min(azims)} / {max(azims)} / {azim_step}')
    # Warn the user if not 0 <= azim <= 355
    if min(azims) < 0 or max(azims) > 355:
        warnings.warn('Azimuths are not in the range 0 <= azim <= 355')
    possible_azims = (max(azims) - min(azims)) // azim_step + 1
    print(f'Possible elevs: {possible_azims}')
    print(f'Azims found: {len(np.unique(azims))}')
    missing_azims = [azim for azim in range(min(azims), max(azims) + 1, azim_step) if azim not in azims]
    print(f'Missing azimuths: {missing_azims}')

    print(f'\nPossible speaker locations: {possible_elevs * possible_azims}')
    print(f'Speaker locations in data: {len(np.unique(labels, axis=0))}')
    print('\n\n')

    # Plot the speaker locations, area of circle is proportional to number of examples
    normalized_circle_sizes = 50 * (unique_label_counts / unique_label_counts.max())
    plt.scatter(unique_labels[:, 1], unique_labels[:, 0], s=normalized_circle_sizes)
    plt.xlabel('Azimuth')
    plt.ylabel('Elevation')
    plt.title(path.as_posix(), fontdict={'fontsize': 10})
    # subtitle path name
    plt.suptitle('Data Distribution')
    plt.show()


def inspect_tfrecord():
    """
    Inspect the structure of a .tfrecord file
    unfinished
    Returns:

    """
    # Each  in a .tfrecord file is of type:
    # A .tfrecord file has the following type structure:
    # List[Dict[str,                 -> Each row / record is an Example, which is a dict that
    #                                   contains a key/value store "features"
    #                 List[int64],      where each feature can be of a different type
    #                 List[float]]]

    # record = tf.io.parse_single_example(raw_record, features)
    # print(example["train/image_width"])
    for key, value in dict(example.features.feature).items():
        print(f'Key: {key}\n'
              f'Value Type: {type(value)}\n'
              f'Value Length: {len(str(value))}\n'
              f'Value: {str(value)[:100]}\n')

    example_bytes = dict(example.features.feature)['train/image'].bytes_list.value[0]
    print(np.frombuffer(example_bytes, dtype=np.float32).shape)
    # Orig data from McDermott is at 48kHz, own data is at 8kHz. The paper states it should be at 8kHz.
    example_numpy_array = np.frombuffer(example_bytes, dtype=np.float32).reshape(39, 8000, 2)
    normalized_array = example_numpy_array * (255.0 / example_numpy_array.max())
    # print(example_numpy_array[:,:,1].shape)
    img = Image.fromarray(normalized_array[:, :, 1])
    img = img.resize((4800, 3900), resample=Image.Resampling.NEAREST)
    # img = img.crop((0, 0, 16000, 20))
    # img = img.resize((16000, 20000))
    # img.show()
    # print(image_array)


def downsample(in_path: Path, out_path: Path):
    # Load the dataset, downsample images to 8kHz, and save to a new dataset
    raw_dataset = tf.data.TFRecordDataset(in_path, compression_type="GZIP")
    c = 0
    for _ in raw_dataset:
        c += 1
    # Create a new dataset
    writer = tf.io.TFRecordWriter(out_path, options="GZIP")

    for raw_record in tqdm(raw_dataset, total=c):
        example = tf.train.Example()
        example.ParseFromString(raw_record.numpy())
        example_dict = dict(example.features.feature)
        example_bytes = dict(example.features.feature)['train/image'].bytes_list.value[0]
        example_numpy_array = np.frombuffer(example_bytes, dtype=np.float32).reshape(39, 48000, 2)
        downsampled = sp.signal.resample(example_numpy_array, 8000, axis=1)
        # print(downsampled.shape)
        # sys.exit()
        # Save the downsampled image to the new dataset
        example_dict['train/image'].bytes_list.value[0] = downsampled.tobytes()
        example = tf.train.Example(features=tf.train.Features(feature=example_dict))
        writer.write(example.SerializeToString())
    writer.close()


class CroppingPrettyPrinter(PrettyPrinter):
    # From: https://stackoverflow.com/a/38534524
    def __init__(self, *args, **kwargs):
        self.maxlist = kwargs.pop('maxlist', 6)
        return PrettyPrinter.__init__(self, *args, **kwargs)

    def _format(self, obj, stream, indent, allowance, context, level):
        if isinstance(obj, list):
            # If object is a list, crop a copy of it according to self.maxlist
            # and append an ellipsis
            if len(obj) > self.maxlist:
                cropped_obj = obj[:self.maxlist] + ['...']
                return PrettyPrinter._format(
                    self, cropped_obj, stream, indent,
                    allowance, context, level)

        # Let the original implementation handle anything else
        # Note: No use of super() because PrettyPrinter is an old-style class
        return PrettyPrinter._format(
            self, obj, stream, indent, allowance, context, level)


if __name__ == "__main__":
    # downsample(Path("./data/training_data_orig_mcdermott.tfrecord"), Path("./data/training_data_orig_mcdermott_8kHz.tfrecord"))

    inspect_data(Path("./data/training_data_2024-09-23_16-38-47/training-data_hrtf-default.tfrecord"))
    inspect_data(Path("./data/training_data_orig_mcdermott_8kHz.tfrecord"))
    # inspect_data(Path("./data/training_data_orig_mcdermott.tfrecord"))
