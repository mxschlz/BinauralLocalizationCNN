# Parts of the TF1 API utilized in the code
-> Shown are aliases organized by the namespace they belong to

## Look in-depth at
- tf.Session -> as_default. When to use that?
  - Construct graph outside of Session, then run it in Session
  - A Session owns physical resources -> use sess.close() or run in with block to free them up
  - tf.Session can be used directly as context manager -> closes automatically after block
  - tf.Session.as_default() can also be used as context manager -> sets this session as default session, but doesn't manage the Session's lifecycle
  - as_default() just makes sure that sess.run() and tf.Tensor.eval() calls use this session
  - You could nest context managers, first the Session and then the as_default() one
- tf.Variable
- tf.Summary
  - Protocol buffer for storing data to be visualized in TensorBoard
- tf.train.Example
- tf.Operation;  How do I see what function call is an Op and what is not?
  - node in a TF Graph taking 0 or more Tensors and producing 0 or more Tensors
  - e.g., c = tf.matmul(a, b) -> c is an Op, not a Tensor or number; bit confusing
  - Executing .run() on an Op will execute the Op; shortcut for sess.run(op) / tf.compat.v1.get_default_session().run(op)
- Reuse (look at Variable Scope How To)
- Devices -> How to make it run on CPU vs GPU in laptop vs supercomputer environments. Like test and prod environments?
- How do contexts work and which of them do I need to take extra care about (Session, control_dependencies, device, ...)
- Why are some classes capitalized and others not?
- What parts of contrib.graph_editor are used?
- parse_single_example better to batch maybe?

- Protocol Buffers -> Google's language-neutral, platform-neutral, extensible mechanism for serializing structured data, like XML but smaller, faster, simpler

## tf top level namespace
tf.add_to_collection() -> Add a value to a collection given its name; wraps Graph.add_to_collection() using default graph
tf.cast() -> Cast a tensor to a new types
tf.constant -> Create a constant tensor
tf.control_dependencies -> creates context (with ctrl_dep(...):) within which the Operations will wait for the specified Tensors / Operations; wraps tf.Graph.control_dependencies using default graph
tf.convert_to_tensor() -> Converts a value to a tensor
tf.decode_raw() -> Convert raw byte strings into tensors (deprecated arguments)
tf.device() -> creates context manager (with dvc(...):) specifying the default device to use; wraps tf.Graph.device using default graph
tf.get_collection -> Get a collection given its name; wraps Graph.get_collection() using default graph
tf.get_default_graph -> Returns default graph for the current thread
tf.get_variable() -> Gets variable w/ these parameters for reuse, or creates new one
tf.global_variables() -> Returns global variables (shared across machines in distr env)
tf.group() -> Groups multiple Operations into one Operation; all inner Ops will be done when the grouped Op is finished
tf.pad() -> Pads a tensor as specified
tf.reset_default_graph() -> Clears default graph stack and resets global default graph. def graph is property of current thread and only applies to this thread. Calling it inside Session or using Tensors and Ops after calling it is undefined 
tf.reshape() -> Reshape a tensor, total number of elements must be the same
tf.slice() -> Extracts a slice from a tensor, like a subset of the tensor
tf.squeeze() -> Removes dimensions of size 1 from the shape of a tensor
tf.stop_gradient -> Stops gradient computation
tf.zeros_like() -> Creates tensor of all zeros with same shape and type as input

## Classes (in top level namespace)
tf.ConfigProto -> Configurations for tf.Session 
tf.Session -> Execute Operation objects and evaluate Tensors
tf.Variable -> Create a variable tensor (looks and acts like a tensor; needs initial value; type and shape are inferred and fixed; value can be changed)
tf.name_scope -> Creates a context manager for use when defining a Python op; adds prefix to all ops created withing the scope; good for organizing ops and visualizing the graph

## Other members
tf.float32
tf.float64
tf.int64
tf.string

## tf.contrib
-> not in TF2!
tf.contrib.data.parallel_interleave() -> parallel version of tf.data.Dataset.interleave (deprecated)
tf.contrib.framework.get_variables_to_restore() -> Gets the list of the variables to restore; unsure what that means...
tf.contrib.graph_editor -> Module: Graph editor
tf.contrib.layers.apply_regularization() -> Returns the summed penalty by applying specified regularizer to the weights_list.

## tf.data
tf.data.Dataset -> Represents a potentially large set of elements; can repr. input pipeline and logical plan of transformations
tf.data.Dataset.from_generator() -> Creates a Dataset whose elements are generated by generator
tf.data.Dataset.list_files() -> A dataset of all files matching one or more glob patterns
tf.data.TFRecordDataset() ! doesn't exist

## tf.errors
tf.errors.DataLossError -> Unrecoverable data loss or corruption
tf.errors.InvalidArgumentError -> Op received invalid argument
tf.errors.OutOfRangeError -> Op iterated past valid input range
tf.errors.ResourceExhaustedError -> Some resource has been exhausted

## tf.graph_util
-> Helpers to manipulate a tensor graph
tf.graph_util.convert_variables_to_constants() -> Replaces all vars in graph with constants (after training, to describe graph in single file and remove var loading ops) (deprecated)

## tf.GraphKeys
Keys to be used with tf.get_collection() and tf.add_to_collection() etc.
tf.GraphKeys.REGULARIZATION_LOSSES -> regularization losses collected during graph construction
tf.GraphKeys.TRAINABLE_VARIABLES -> Variable(trainable=True) adds it to the collection with this key
tf.GraphKeys.UPDATE_OPS -> Associated collection collects batch normalization update operations that need to be run during training

## tf.initializers
tf.global_variables_initializer -> An Op that inits global vars in the graph; alias of tf.initializers.global_variables()
tf.initialize_variables() -> Same as passing list of initializers to Group()
tf.local_variables_initializer -> same but for local vars
tf.variables_initializer -> Returns Op that can be run to initialize the given variables; alias of tf.initializers.variables()

## tf.io
tf.FixedLenFeature -> A feature in a TFRecord file with fixed length
tf.VarLenFeature -> A feature in a TFRecord file with variable length
tf.parse_single_example() -> Parses a single Example proto

## tf.layers
tf.layers.batch_normalization -> Functional interface for batch normalization layer (deprecated)
tf.layers.dropout -> Applies dropout to input (deprecated)

## tf.linalg
tf.matmul() -> Matrix multiplication

## tf.math
tf.add() -> element-wise addition
tf.argmax -> Return index of largest value across axis
tf.div() -> element-wise division (deprecated, use tf.math.divide)
tf.equal -> Return True if two tensors are equal element-wise
tf.math.top_k -> Returns k largest elements along axis
tf.pow() -> element-wise power
tf.reduce_mean() -> Compute mean across a specified dimension of a tensor
tf.scalar_mul() -> multiply a scalar with a tensor
tf.sqrt() -> element-wise square root

## tf.metrics
tf.metrics.auc -> Computes the approximate AUC via a Riemann sum

## tf.nn
- Wrappers for primitive Neural Net (NN) Operations
tf.nn.bias_add -> Adds bias
tf.nn.conv2d -> Computes 2D convolution
tf.nn.maxpool -> Performs max pooling on input
tf.nn.relu -> Computes rectified linear
tf.nn.sigmoid -> Computes sigmoid
tf.nn.sigmoid_cross_entropy_with_logits() -> Computes sigmoid cross entropy given logits, i.e., on non-normalized data; usually feeds into softmax layer
tf.nn.softmax -> Computes softmax
tf.nn.sparse_softmax_cross_entropy_with_logits() -> Computes sparse softmax cross entropy between logits and labels; input is unscaled logits
tf.nn.top_k -> alias of tf.math.top_k

## tf.python_io
-> Directly manipulating TFRecord-formatted files
tf.python.ops.control_flow_ops -> doesn't exist
tf.python.ops.gradients -> doesn't exist
tf.python_io.TFRecordOptions -> Options for manipulating TFRecord files
tf.python_io.tf_record_iterator -> Iterator reading records from TFRecord file (deprecated)

## tf.sparse
tf.SparseTensor -> Tensor w/ few non-zero values; specified by list of indices w/ non-zero values and list of values
tf.sparse.from_dense() -> dense to sparse tensor
tf.sparse.to_dense() -> sparse to dense tensor
tf.sparse.to_indicator() -> sparse to indicator tensor, replacing last dimension with # of elements in that dimension and True where non-zero

## tf.summary
tf.summary.FileWriter -> Asynchronously writes Summary protocol buffers to event files

## tf.train
tf.train.AdamOptimizer -> Optimizer that implements the Adam algorithm
tf.train.Example.FromString -> unsure
tf.train.Saver -> Operators for saving and restoring variables to and from checkpoints
tf.train.load_checkpoint() -> Returns CheckpointReader for checkpoint found in ckpt_dir_or_file