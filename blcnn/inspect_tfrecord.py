from pathlib import Path
import logging
from pathlib import Path
from typing import List

import coloredlogs
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from blcnn.util import CNNpos_to_loc
from persistent_cache import persistent_cache

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
coloredlogs.install(level='DEBUG', logger=logger, fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


# Features in a tfrecord file, more are possible
# features = {
#     "train/image": tf.train.Feature(bytes_list=tf.train.BytesList(value=[])),
#     "train/image_width": tf.train.Feature(int64_list=tf.train.Int64List(value=[])),
#     "train/image_height": tf.train.Feature(int64_list=tf.train.Int64List(value=[])),
#     "train/elev": tf.train.Feature(int64_list=tf.train.Int64List(value=[])),
#     "train/azim": tf.train.Feature(int64_list=tf.train.Int64List(value=[]))
# }


def main() -> None:
    inspect_data(Path("/Users/david/Repositories/ma/BinauralLocalizationCNN/data/cochleagrams/uso_500ms_raw_slab_kemar_onlyHRTF/cochleagrams.tfrecord"))


def inspect_data(path: Path):
    dataset = tf.data.TFRecordDataset(path, compression_type="GZIP")
    elevs, azims = extract_elev_azim(dataset)#, persistent_cache_key=path)

    print('#################################################################')
    print('#################### .tfrecord data analysis ####################')
    print('#################################################################')
    print(f'Dataset path: {path}')
    print(f'Contains {dataset.cardinality()} examples (-1 is infinite cardinality, -2 is unknown cardinality)')
    print(f'\nOptions (showing only populated values): ')
    print_pprint(dataset.options(), suppress_none_values=True)

    labels = np.array([elevs, azims]).T
    print(f'\nFound {len(labels)} examples')
    print('Examples:\n', labels[:5])
    unique_labels, unique_label_counts = np.unique(labels, axis=0, return_counts=True)

    # Elevations
    elev_step = np.min(np.diff(np.unique(elevs)))  # smallest elevation step
    print(f'\nExtracted elev range (min / max / smallest step): {min(elevs)} / {max(elevs)} / {elev_step}')
    possible_elevs = (max(elevs) - min(elevs)) // elev_step + 1
    print(f'Possible elevs: {possible_elevs}')
    print(f'Elevs found: {len(np.unique(elevs))}')
    missing_elevs = [elev for elev in range(min(elevs), max(elevs) + 1, elev_step) if elev not in elevs]
    print(f'Missing elevations: {missing_elevs}')

    # Azimuths
    azim_step = np.min(np.diff(np.unique(azims)))
    print(f'\nExtracted azim range (min / max / smallest step): {min(azims)} / {max(azims)} / {azim_step}')
    # Warn the user if not 0 <= azim <= 355
    if min(azims) < 0 or max(azims) > 355:
        logger.warning('Azimuths are not in the range 0 <= azim <= 355')
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
    plt.title(str(path).split("data/")[-1], fontdict={'fontsize': 10})
    plt.suptitle('Data Distribution')
    plt.show()

    # Print all information in the first example
    raw_example = dataset.take(1).get_single_element().numpy()
    example = tf.train.Example()
    example.ParseFromString(raw_example)
    print('Contents of one Example in the dataset:')
    for key, value in example.features.feature.items():
        if key == 'train/image':
            print(f'example[\'{key}\'] = {str(value)[:50]} ... {str(value)[-50:]}\n')
        else:
            print(f'example[\'{key}\'] = {value}\n')

    # Plot the first cochleagram
    image_bytes = example.features.feature['train/image'].bytes_list.value[0]
    example_numpy_array = np.frombuffer(image_bytes, dtype=np.float32)
    normalized_array = example_numpy_array * (255.0 / example_numpy_array.max())
    try:
        reshaped = normalized_array.reshape(39, 48000, 2)
    except ValueError:
        try:
            reshaped = normalized_array.reshape(39, 8000, 2)
        except ValueError:
            logger.error('Could not reshape array, cochleagram should be 48000 or 8000 samples long')
    img = Image.fromarray(reshaped[:, :, 1])
    # img = img.crop((12000, 0, 16000, 39))
    img = img.resize((4000, 1000), resample=Image.Resampling.NEAREST)
    ImageDraw.Draw(img).text(
        xy=(20, 20),
        text=f'Example cochleagram from data/{str(path).split("data/")[-1]},\nimage normalized and resized (might lose detail)',
        fill=255,
        font=ImageFont.load_default(size=36))
    img.show()


# @persistent_cache
def extract_elev_azim(dataset: tf.data.TFRecordDataset) -> (List, List):
    # assert key is not None, 'Key must be provided for caching'
    elevs = []
    azims = []
    for raw_record in tqdm(dataset, desc='Extracting elevations and azimuths'):
        example = tf.train.Example()
        example.ParseFromString(raw_record.numpy())
        # Extract the label from the example
        azim, elev = CNNpos_to_loc(example.features.feature['train/target'].int64_list.value[0])
        # elev = example.features.feature['train/elev'].int64_list.value[0]
        # azim = example.features.feature['train/azim'].int64_list.value[0]

        elevs.append(elev)
        azims.append(azim)
    return elevs, azims


def print_pprint(*args, **kwargs):
    output = generate_pprint(*args, **kwargs)
    for line in output:
        print(line)


# Adapted from https://gist.github.com/fonic/dfcaa350d5fb57835cf6c1689d251912
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


if __name__ == "__main__":
    main()