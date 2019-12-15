"""Hyperparameter values."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import numbers
import re
import six

# Define the regular expression for parsing a single clause of the input
# (delimited by commas).  A legal clause looks like:
#   <variable name>[<index>]? = <rhs>
# where <rhs> is either a single token or [] enclosed list of tokens.
# For example:  "var[1] = a" or "x = [1,2,3]"
PARAM_RE = re.compile(r"""
  (?P<name>[a-zA-Z][\w\.]*)      # variable name: "var" or "x"
  (\[\s*(?P<index>\d+)\s*\])?  # (optional) index: "1" or None
  \s*=\s*
  ((?P<val>[^,\[]*)            # single value: "a" or None
   |
   \[(?P<vals>[^\]]*)\])       # list of values: None or "1,2,3"
  ($|,\s*)""", re.VERBOSE)


def _parse_fail(name, var_type, value, values):
    """Helper function for raising a value error for bad assignment."""
    raise ValueError(
        'Could not parse hparam \'%s\' of type \'%s\' with value \'%s\' in %s' %
        (name, var_type.__name__, value, values))


def _reuse_fail(name, values):
    """Helper function for raising a value error for reuse of name."""
    raise ValueError('Multiple assignments to variable \'%s\' in %s' % (name,
                                                                        values))


def _process_scalar_value(name, parse_fn, var_type, m_dict, values,
                          results_dictionary):
    """Update results_dictionary with a scalar value.
  Used to update the results_dictionary to be returned by parse_values when
  encountering a clause with a scalar RHS (e.g.  "s=5" or "arr[0]=5".)
  Mutates results_dictionary.
  Args:
    name: Name of variable in assignment ("s" or "arr").
    parse_fn: Function for parsing the actual value.
    var_type: Type of named variable.
    m_dict: Dictionary constructed from regex parsing.
      m_dict['val']: RHS value (scalar)
      m_dict['index']: List index value (or None)
    values: Full expression being parsed
    results_dictionary: The dictionary being updated for return by the parsing
      function.
  Raises:
    ValueError: If the name has already been used.
  """
    try:
        parsed_value = parse_fn(m_dict['val'])
    except ValueError:
        _parse_fail(name, var_type, m_dict['val'], values)

    # If no index is provided
    if not m_dict['index']:
        if name in results_dictionary:
            _reuse_fail(name, values)
        results_dictionary[name] = parsed_value
    else:
        if name in results_dictionary:
            # The name has already been used as a scalar, then it
            # will be in this dictionary and map to a non-dictionary.
            if not isinstance(results_dictionary.get(name), dict):
                _reuse_fail(name, values)
        else:
            results_dictionary[name] = {}

        index = int(m_dict['index'])
        # Make sure the index position hasn't already been assigned a value.
        if index in results_dictionary[name]:
            _reuse_fail('{}[{}]'.format(name, index), values)
        results_dictionary[name][index] = parsed_value


def _process_list_value(name, parse_fn, var_type, m_dict, values,
                        results_dictionary):
    """Update results_dictionary from a list of values.
  Used to update results_dictionary to be returned by parse_values when
  encountering a clause with a list RHS (e.g.  "arr=[1,2,3]".)
  Mutates results_dictionary.
  Args:
    name: Name of variable in assignment ("arr").
    parse_fn: Function for parsing individual values.
    var_type: Type of named variable.
    m_dict: Dictionary constructed from regex parsing.
      m_dict['val']: RHS value (scalar)
    values: Full expression being parsed
    results_dictionary: The dictionary being updated for return by the parsing
      function.
  Raises:
    ValueError: If the name has an index or the values cannot be parsed.
  """
    if m_dict['index'] is not None:
        raise ValueError('Assignment of a list to a list index.')
    elements = filter(None, re.split('[ ,]', m_dict['vals']))
    # Make sure the name hasn't already been assigned a value
    if name in results_dictionary:
        raise _reuse_fail(name, values)
    try:
        results_dictionary[name] = [parse_fn(e) for e in elements]
    except ValueError:
        _parse_fail(name, var_type, m_dict['vals'], values)


def _cast_to_type_if_compatible(name, param_type, value):
    """Cast hparam to the provided type, if compatible.
  Args:
    name: Name of the hparam to be cast.
    param_type: The type of the hparam.
    value: The value to be cast, if compatible.
  Returns:
    The result of casting `value` to `param_type`.
  Raises:
    ValueError: If the type of `value` is not compatible with param_type.
      * If `param_type` is a string type, but `value` is not.
      * If `param_type` is a boolean, but `value` is not, or vice versa.
      * If `param_type` is an integer type, but `value` is not.
      * If `param_type` is a float type, but `value` is not a numeric type.
  """
    fail_msg = (
            "Could not cast hparam '%s' of type '%s' from value %r" %
            (name, param_type, value))

    # Some callers use None, for which we can't do any casting/checking. :(
    if issubclass(param_type, type(None)):
        return value

    # Avoid converting a non-string type to a string.
    if (issubclass(param_type, (six.string_types, six.binary_type)) and
            not isinstance(value, (six.string_types, six.binary_type))):
        raise ValueError(fail_msg)

    # Avoid converting a number or string type to a boolean or vice versa.
    if issubclass(param_type, bool) != isinstance(value, bool):
        raise ValueError(fail_msg)

    # Avoid converting float to an integer (the reverse is fine).
    if (issubclass(param_type, numbers.Integral) and
            not isinstance(value, numbers.Integral)):
        raise ValueError(fail_msg)

    # Avoid converting a non-numeric type to a numeric type.
    if (issubclass(param_type, numbers.Number) and
            not isinstance(value, numbers.Number)):
        raise ValueError(fail_msg)

    return param_type(value)


def parse_values(values, type_map, ignore_unknown=False):
    """Parses hyperparameter values from a string into a python map.
  `values` is a string containing comma-separated `name=value` pairs.
  For each pair, the value of the hyperparameter named `name` is set to
  `value`.
  If a hyperparameter name appears multiple times in `values`, a ValueError
  is raised (e.g. 'a=1,a=2', 'a[1]=1,a[1]=2').
  If a hyperparameter name in both an index assignment and scalar assignment,
  a ValueError is raised.  (e.g. 'a=[1,2,3],a[0] = 1').
  The hyperparameter name may contain '.' symbols, which will result in an
  attribute name that is only accessible through the getattr and setattr
  functions.  (And must be first explicit added through add_hparam.)
  WARNING: Use of '.' in your variable names is allowed, but is not well
  supported and not recommended.
  The `value` in `name=value` must follows the syntax according to the
  type of the parameter:
  *  Scalar integer: A Python-parsable integer point value.  E.g.: 1,
     100, -12.
  *  Scalar float: A Python-parsable floating point value.  E.g.: 1.0,
     -.54e89.
  *  Boolean: Either true or false.
  *  Scalar string: A non-empty sequence of characters, excluding comma,
     spaces, and square brackets.  E.g.: foo, bar_1.
  *  List: A comma separated list of scalar values of the parameter type
     enclosed in square brackets.  E.g.: [1,2,3], [1.0,1e-12], [high,low].
  When index assignment is used, the corresponding type_map key should be the
  list name.  E.g. for "arr[1]=0" the type_map must have the key "arr" (not
  "arr[1]").
  Args:
    values: String.  Comma separated list of `name=value` pairs where
      'value' must follow the syntax described above.
    type_map: A dictionary mapping hyperparameter names to types.  Note every
      parameter name in values must be a key in type_map.  The values must
      conform to the types indicated, where a value V is said to conform to a
      type T if either V has type T, or V is a list of elements of type T.
      Hence, for a multidimensional parameter 'x' taking float values,
      'x=[0.1,0.2]' will parse successfully if type_map['x'] = float.
    ignore_unknown: Bool. Whether values that are missing a type in type_map
      should be ignored. If set to True, a ValueError will not be raised for
      unknown hyperparameter type.
  Returns:
    A python map mapping each name to either:
    * A scalar value.
    * A list of scalar values.
    * A dictionary mapping index numbers to scalar values.
    (e.g. "x=5,L=[1,2],arr[1]=3" results in {'x':5,'L':[1,2],'arr':{1:3}}")
  Raises:
    ValueError: If there is a problem with input.
    * If `values` cannot be parsed.
    * If a list is assigned to a list index (e.g. 'a[1] = [1,2,3]').
    * If the same rvalue is assigned two different values (e.g. 'a=1,a=2',
      'a[1]=1,a[1]=2', or 'a=1,a=[1]')
  """
    results_dictionary = {}
    pos = 0
    while pos < len(values):
        m = PARAM_RE.match(values, pos)
        if not m:
            raise ValueError('Malformed hyperparameter value: %s' % values[pos:])
        # Check that there is a comma between parameters and move past it.
        pos = m.end()
        # Parse the values.
        m_dict = m.groupdict()
        name = m_dict['name']
        if name not in type_map:
            if ignore_unknown:
                continue
            raise ValueError('Unknown hyperparameter type for %s' % name)
        type_ = type_map[name]

        # Set up correct parsing function (depending on whether type_ is a bool)
        if type_ == bool:

            def parse_bool(value):
                if value in ['true', 'True']:
                    return True
                elif value in ['false', 'False']:
                    return False
                else:
                    try:
                        return bool(int(value))
                    except ValueError:
                        _parse_fail(name, type_, value, values)

            parse = parse_bool
        else:
            parse = type_

        # If a singe value is provided
        if m_dict['val'] is not None:
            _process_scalar_value(name, parse, type_, m_dict, values,
                                  results_dictionary)

        # If the assigned value is a list:
        elif m_dict['vals'] is not None:
            _process_list_value(name, parse, type_, m_dict, values,
                                results_dictionary)

        else:  # Not assigned a list or value
            _parse_fail(name, type_, '', values)

    return results_dictionary


class HParams(object):
    """Class to hold a set of hyperparameters as name-value pairs.
  A `HParams` object holds hyperparameters used to build and train a model,
  such as the number of hidden units in a neural net layer or the learning rate
  to use when training.
  You first create a `HParams` object by specifying the names and values of the
  hyperparameters.
  To make them easily accessible the parameter names are added as direct
  attributes of the class.  A typical usage is as follows:
  ```python
  # Create a HParams object specifying names and values of the model
  # hyperparameters:
  hparams = HParams(learning_rate=0.1, num_hidden_units=100)
  # The hyperparameter are available as attributes of the HParams object:
  hparams.learning_rate ==> 0.1
  hparams.num_hidden_units ==> 100
  ```
  Hyperparameters have type, which is inferred from the type of their value
  passed at construction type.   The currently supported types are: integer,
  float, boolean, string, and list of integer, float, boolean, or string.
  You can override hyperparameter values by calling the
  [`parse()`](#HParams.parse) method, passing a string of comma separated
  `name=value` pairs.  This is intended to make it possible to override
  any hyperparameter values from a single command-line flag to which
  the user passes 'hyper-param=value' pairs.  It avoids having to define
  one flag for each hyperparameter.
  The syntax expected for each value depends on the type of the parameter.
  See `parse()` for a description of the syntax.
  Example:
  ```python
  # Define a command line flag to pass name=value pairs.
  # For example using argparse:
  import argparse
  parser = argparse.ArgumentParser(description='Train my model.')
  parser.add_argument('--hparams', type=str,
                      help='Comma separated list of "name=value" pairs.')
  args = parser.parse_args()
  ...
  def my_program():
    # Create a HParams object specifying the names and values of the
    # model hyperparameters:
    hparams = tf.HParams(learning_rate=0.1, num_hidden_units=100,
                         activations=['relu', 'tanh'])
    # Override hyperparameters values by parsing the command line
    hparams.parse(args.hparams)
    # If the user passed `--hparams=learning_rate=0.3` on the command line
    # then 'hparams' has the following attributes:
    hparams.learning_rate ==> 0.3
    hparams.num_hidden_units ==> 100
    hparams.activations ==> ['relu', 'tanh']
    # If the hyperparameters are in json format use parse_json:
    hparams.parse_json('{"learning_rate": 0.3, "activations": "relu"}')
  ```
  """

    _HAS_DYNAMIC_ATTRIBUTES = True  # Required for pytype checks.

    def __init__(self, model_structure=None, **kwargs):
        """Create an instance of `HParams` from keyword arguments.
    The keyword arguments specify name-values pairs for the hyperparameters.
    The parameter types are inferred from the type of the values passed.
    The parameter names are added as attributes of `HParams` object, so they
    can be accessed directly with the dot notation `hparams._name_`.
    Example:
    ```python
    # Define 3 hyperparameters: 'learning_rate' is a float parameter,
    # 'num_hidden_units' an integer parameter, and 'activation' a string
    # parameter.
    hparams = tf.HParams(
        learning_rate=0.1, num_hidden_units=100, activation='relu')
    hparams.activation ==> 'relu'
    ```
    Note that a few names are reserved and cannot be used as hyperparameter
    names.  If you use one of the reserved name the constructor raises a
    `ValueError`.
    Args:
      model_structure: An instance of ModelStructure, defining the feature
        crosses to be used in the Trial.
      **kwargs: Key-value pairs where the key is the hyperparameter name and
        the value is the value for the parameter.
    Raises:
      ValueError: If both `hparam_def` and initialization values are provided,
        or if one of the arguments is invalid.
    """
        # Register the hyperparameters and their type in _hparam_types.
        # This simplifies the implementation of parse().
        # _hparam_types maps the parameter name to a tuple (type, bool).
        # The type value is the type of the parameter for scalar hyperparameters,
        # or the type of the list elements for multidimensional hyperparameters.
        # The bool value is True if the value is a list, False otherwise.
        self._hparam_types = {}
        self._model_structure = model_structure
        for name, value in six.iteritems(kwargs):
            self.add_hparam(name, value)

    def add_hparam(self, name, value):
        """Adds {name, value} pair to hyperparameters.
    Args:
      name: Name of the hyperparameter.
      value: Value of the hyperparameter. Can be one of the following types:
        int, float, string, int list, float list, or string list.
    Raises:
      ValueError: if one of the arguments is invalid.
    """
        # Keys in kwargs are unique, but 'name' could the name of a pre-existing
        # attribute of this object.  In that case we refuse to use it as a
        # hyperparameter name.
        if getattr(self, name, None) is not None:
            raise ValueError('Hyperparameter name is reserved: %s' % name)
        if isinstance(value, (list, tuple)):
            if not value:
                raise ValueError(
                    'Multi-valued hyperparameters cannot be empty: %s' % name)
            self._hparam_types[name] = (type(value[0]), True)
        else:
            self._hparam_types[name] = (type(value), False)
        setattr(self, name, value)

    def set_hparam(self, name, value):
        """Set the value of an existing hyperparameter.
    This function verifies that the type of the value matches the type of the
    existing hyperparameter.
    Args:
      name: Name of the hyperparameter.
      value: New value of the hyperparameter.
    Raises:
      KeyError: If the hyperparameter doesn't exist.
      ValueError: If there is a type mismatch.
    """
        param_type, is_list = self._hparam_types[name]
        if isinstance(value, list):
            if not is_list:
                raise ValueError(
                    'Must not pass a list for single-valued parameter: %s' % name)
            setattr(self, name, [
                _cast_to_type_if_compatible(name, param_type, v) for v in value])
        else:
            if is_list:
                raise ValueError(
                    'Must pass a list for multi-valued parameter: %s.' % name)
            setattr(self, name, _cast_to_type_if_compatible(name, param_type, value))

    def del_hparam(self, name):
        """Removes the hyperparameter with key 'name'.
    Does nothing if it isn't present.
    Args:
      name: Name of the hyperparameter.
    """
        if hasattr(self, name):
            delattr(self, name)
            del self._hparam_types[name]

    def parse(self, values):
        """Override existing hyperparameter values, parsing new values from a string.
    See parse_values for more detail on the allowed format for values.
    Args:
      values: String.  Comma separated list of `name=value` pairs where 'value'
        must follow the syntax described above.
    Returns:
      The `HParams` instance.
    Raises:
      ValueError: If `values` cannot be parsed or a hyperparameter in `values`
      doesn't exist.
    """
        type_map = {}
        for name, t in self._hparam_types.items():
            param_type, _ = t
            type_map[name] = param_type

        values_map = parse_values(values, type_map)
        return self.override_from_dict(values_map)

    def override_from_dict(self, values_dict):
        """Override existing hyperparameter values, parsing new values from a dictionary.
    Args:
      values_dict: Dictionary of name:value pairs.
    Returns:
      The `HParams` instance.
    Raises:
      KeyError: If a hyperparameter in `values_dict` doesn't exist.
      ValueError: If `values_dict` cannot be parsed.
    """
        for name, value in values_dict.items():
            self.set_hparam(name, value)
        return self

    def set_model_structure(self, model_structure):
        self._model_structure = model_structure

    def get_model_structure(self):
        return self._model_structure

    def to_json(self, indent=None, separators=None, sort_keys=False):
        """Serializes the hyperparameters into JSON.
    Args:
      indent: If a non-negative integer, JSON array elements and object members
        will be pretty-printed with that indent level. An indent level of 0, or
        negative, will only insert newlines. `None` (the default) selects the
        most compact representation.
      separators: Optional `(item_separator, key_separator)` tuple. Default is
        `(', ', ': ')`.
      sort_keys: If `True`, the output dictionaries will be sorted by key.
    Returns:
      A JSON string.
    """

        def remove_callables(x):
            """Omit callable elements from input with arbitrary nesting."""
            if isinstance(x, dict):
                return {k: remove_callables(v) for k, v in six.iteritems(x)
                        if not callable(v)}
            elif isinstance(x, list):
                return [remove_callables(i) for i in x if not callable(i)]
            return x

        return json.dumps(
            remove_callables(self.values()),
            indent=indent,
            separators=separators,
            sort_keys=sort_keys)

    def parse_json(self, values_json):
        """Override existing hyperparameter values, parsing new values from a json object.
    Args:
      values_json: String containing a json object of name:value pairs.
    Returns:
      The `HParams` instance.
    Raises:
      KeyError: If a hyperparameter in `values_json` doesn't exist.
      ValueError: If `values_json` cannot be parsed.
    """
        values_map = json.loads(values_json)
        return self.override_from_dict(values_map)

    def values(self):
        """Return the hyperparameter values as a Python dictionary.
    Returns:
      A dictionary with hyperparameter names as keys.  The values are the
      hyperparameter values.
    """
        return {n: getattr(self, n) for n in self._hparam_types.keys()}

    def get(self, key, default=None):
        """Returns the value of `key` if it exists, else `default`."""
        if key in self._hparam_types:
            # Ensure that default is compatible with the parameter type.
            if default is not None:
                param_type, is_param_list = self._hparam_types[key]
                type_str = 'list<%s>' % param_type if is_param_list else str(param_type)
                fail_msg = ("Hparam '%s' of type '%s' is incompatible with "
                            'default=%s' % (key, type_str, default))

                is_default_list = isinstance(default, list)
                if is_param_list != is_default_list:
                    raise ValueError(fail_msg)

                try:
                    if is_default_list:
                        for value in default:
                            _cast_to_type_if_compatible(key, param_type, value)
                    else:
                        _cast_to_type_if_compatible(key, param_type, default)
                except ValueError as e:
                    raise ValueError('%s. %s' % (fail_msg, e))

            return getattr(self, key)

        return default

    def __contains__(self, key):
        return key in self._hparam_types

    def __str__(self):
        return str(sorted(self.values().items()))

    def __repr__(self):
        return '%s(%s)' % (type(self).__name__, self.__str__())

    @staticmethod
    def _get_kind_name(param_type, is_list):
        """Returns the field name given parameter type and is_list.
    Args:
      param_type: Data type of the hparam.
      is_list: Whether this is a list.
    Returns:
      A string representation of the field name.
    Raises:
      ValueError: If parameter type is not recognized.
    """
        if issubclass(param_type, bool):
            # This check must happen before issubclass(param_type, six.integer_types),
            # since Python considers bool to be a subclass of int.
            typename = 'bool'
        elif issubclass(param_type, six.integer_types):
            # Setting 'int' and 'long' types to be 'int64' to ensure the type is
            # compatible with both Python2 and Python3.
            typename = 'int64'
        elif issubclass(param_type, (six.string_types, six.binary_type)):
            # Setting 'string' and 'bytes' types to be 'bytes' to ensure the type is
            # compatible with both Python2 and Python3.
            typename = 'bytes'
        elif issubclass(param_type, float):
            typename = 'float'
        else:
            raise ValueError('Unsupported parameter type: %s' % str(param_type))

        suffix = 'list' if is_list else 'value'
        return '_'.join([typename, suffix])


def basic_params1():
    """A set of basic hyperparameters."""
    return HParams(
        # If the problem consists of variable-length sequences
        # (see problem.batch_size_means_tokens()), then this is the number
        # of tokens per batch per GPU or per TPU core.  Otherwise, this is
        # the number of examples per GPU or per TPU core.
        batch_size=4096,
        batch_shuffle_size=512,
        # If True, then if the features are of variable length, the batch_size is
        # used as the actual batch size (and not tokens per batch).
        use_fixed_batch_size=False,
        num_hidden_layers=4,
        kernel_height=3,
        kernel_width=1,
        hidden_size=64,
        compress_steps=0,
        # All hyperparameters ending in "dropout" are automatically set to 0.0
        # when not in training mode.
        dropout=0.2,
        clip_grad_norm=2.0,
        grad_noise_scale=0.0,
        summarize_grads=False,
        # Flag for whether mlperf mode is on
        mlperf_mode=False,
        # Whether to log the name and size of every variable
        summarize_vars=False,
        initializer="orthogonal",
        initializer_gain=1.5,
        label_smoothing=0.1,
        optimizer="adam",
        optimizer_adam_epsilon=1e-6,
        optimizer_adam_beta1=0.85,
        optimizer_adam_beta2=0.997,
        optimizer_momentum_momentum=0.9,
        optimizer_momentum_nesterov=False,
        optimizer_adafactor_beta1=0.0,
        optimizer_adafactor_beta2=0.999,
        optimizer_adafactor_factored=True,
        optimizer_adafactor_decay_type="pow",
        optimizer_adafactor_memory_exponent=0.8,
        optimizer_adafactor_clipping_threshold=1.0,
        optimizer_adafactor_multiply_by_parameter_scale=True,
        # Number of accumulating steps for multi step optimizers.
        optimizer_multistep_accumulate_steps=0,
        # Loss scaling used.
        # Generally only necessary with mixed precision training.
        # Mixed precision training only supports exponential scaling currently
        # To disable the scaler, see to 0/False
        mixed_precision_optimizer_loss_scaler="exponential",
        # Determines the initial loss scaling value for mixed precision
        mixed_precision_optimizer_init_loss_scale=2 ** 15,
        # Whether to zero gradients that were not computed, so that the
        # appropriate slots are created. Useful for sharing checkpoints between
        # models with different sets of heads.
        optimizer_zero_grads=False,
        weight_decay=1e-6,
        weight_noise=0.0,
        # Defines the learning rate as a product of named functions.
        # Available functions are listed in learning_rate._LEARNING_RATE_FUNCTIONS
        # e.g. "constant*linear_warmup*rsqrt_decay*rsqrt_hidden_size"
        learning_rate_schedule="legacy",
        learning_rate_constant=1.0,
        # If learning_rate_schedule=="legacy",
        # then we specify decay scheme here.  Warmup is always exponential,
        # except with "noam" learning rate decay scheme.
        # see optimize.legacy_learning_rate_schedule()
        # TODO(noam): migrate everyone away from this.
        learning_rate_decay_scheme="none",
        # decay_steps and decay_staircase for learning_rate_decay_scheme=="exp"
        learning_rate_decay_steps=5000,
        learning_rate_decay_staircase=False,
        learning_rate_minimum=None,
        learning_rate_decay_rate=1.0,
        learning_rate_warmup_steps=100,
        learning_rate_cosine_cycle_steps=250000,
        learning_rate=0.1,
        sampling_method="argmax",  # "argmax" or "random"
        sampling_temp=1.0,  # temperature for sampling
        sampling_keep_top_k=-1,  # If >0, ignore all but the top k logits
        # expand the logits a piece at a time - saves memory.
        factored_logits=False,
        multiply_embedding_mode="sqrt_depth",
        # Parameters related to mixtures of experts.
        moe_hidden_sizes="2048",  # hidden layer sizes (comma-separated)
        moe_num_experts=64,  # number of experts per layer
        moe_k=2,  # how many experts to use for each batch element
        moe_loss_coef=1e-2,
        # Sequences of operations to perform on layer input and layer output.
        # Used by common_layers.layer_preprocess, common_layers.layer_postprocess
        # Each character represents an operation:
        # none: no preprocessing
        #    d: apply dropout
        #    n: apply normalization (see norm_type and norm_epsilon)
        #    a: add layer input (residual connection - only during postprocess)
        # The special string "none" is used instead of the empty string
        # to indicate no pre/postprocessing, since the empty string causes
        # trouble for hyperparameter tuning.
        # TODO(noam): The current settings ("", "dan") are the published version
        # of the transformer.  ("n", "da") seems better for harder-to-learn
        # models, so it should probably be the default.
        layer_preprocess_sequence="none",
        layer_postprocess_sequence="dan",
        # dropout rate to use during layer_preprocess and layer_postprocess
        layer_prepostprocess_dropout=0.1,
        # broadcast dimensions for layer_prepostprocess_dropout
        # a comma-separated list of integers.
        # see common_layers.dropout_with_broadcast_dims()
        # Change this to "1" to save memory.
        layer_prepostprocess_dropout_broadcast_dims="",
        # dropout some symbols (set them to 0) before embedding.
        symbol_dropout=0.0,
        # What type of normalization to use
        norm_type="layer",  # "batch", layer", "noam", "none".
        # epsilon parameter to normalization function
        norm_epsilon=1e-6,
        # pad vocabularies so that this value divides the vocabulary size.
        vocab_divisor=1,
        # During training, we drop sequences whose inputs and targets are shorter
        # than min_length
        min_length=0,
        # During training, we drop sequences whose inputs or targets are longer
        # than max_length.
        # If max_length==0, we use hparams.batch_size instead.
        max_length=0,
        # Pack examples on the fly.
        pack_dataset=False,
        # Use custom ops not included in standard tensorflow.
        use_custom_ops=True,
        # Split targets on the first axis into chunks of this length.
        split_targets_chunk_length=0,
        split_targets_max_chunks=100,
        split_targets_strided_training=False,
        # Maximum length in the smallest length bucket.  Setting this
        # flag too high will result in wasteful padding of short
        # sequences.  Due to some (hopefully) temporary hacks in the
        # data reading and batching code, setting this flag too low
        # results in a very long batch-shuffling queue.
        # TODO(noam): change this once the Datasets API changes.
        min_length_bucket=8,
        # This flag controls the number of length buckets in the data
        # reader.  The buckets have maximum lengths from
        # min_bucket_length to (max_length or batch_size), increasing
        # (approximately) by factors of length_bucket_step.
        length_bucket_step=1.1,
        # If set to True, drop sequences longer than max_length during eval.
        # This affects the validity of the evaluation metrics.
        eval_drop_long_sequences=False,
        # If True, run the model autoregressively instead of teacher-forcing
        # during eval
        eval_run_autoregressive=False,
        # (For features with symbol modality) If True, share all of the
        # input embeddings, target embeddings, and softmax weights.
        shared_embedding_and_softmax_weights=False,
        # (For features with symbol modality) If True, share the input embeddings
        # and target embeddings.
        shared_embedding=False,
        # (For features with symbol modality) Number to shard embeddings by.
        symbol_modality_num_shards=1,
        # Feature transformations are optional dictionaries comprising key-value
        # pairs of a feature name (str) and its transformation (function). If not
        # specified, T2TModel applies a default transformation according to the
        # feature's modality. Bottom is applicable to all features; loss, top, and
        # weights_fn are only applicable to target features.
        # TODO(trandustin): `name` is an optional hparam for legacy reasons,
        # defining variable scope names. Remove this hparam in the future.
        bottom={},
        loss={},
        name={},
        top={},
        weights_fn={},
        # The maximum length of "input" sequence.
        # Sequences longer than this value will be truncated. 0 or negative values
        # mean there is no maximum or truncation.
        # You can change this behavior by overriding preprocess_example() method
        # in your problem class.
        max_input_seq_length=0,
        # The maximum length of "target" sequence.
        # Sequences longer than this value will be truncated. 0 or negative values
        # mean there is no maximum or truncation.
        # You can change this behavior by overriding preprocess_example() method
        # in your problem class.
        max_target_seq_length=0,
        # if nonzero, we split the target sequences on example read.
        # This is for use with language modeling problems with fixed length
        # examples.  e.g.  The examples may be written with length 65536, but we
        # want to split each example into 64 examples of length 1024.
        split_to_length=0,
        # Video settings: how many frames to batch on input and targets.
        video_num_input_frames=1,
        video_num_target_frames=1,
        # This flag allows us to optionally treat a seq-to-seq problem
        # as a language model.  Legal values are:
        #
        # "none" - Do not prepend the inputs to the targets.
        # "prepend_inputs_masked_attention"
        #     replace "targets" in preprocessing with
        #     tf.concat([inputs, [0], targets], axis=1)
        #     i.e. we prepend the inputs to the targets with a single
        #     padding token in between.  Use masked self-attention on the
        #     entire resulting sequence.  During training, we compute losses on
        #     the combined sequence.  During eval, we compute the metrics
        #     on only the targets portion.
        # "prepend_inputs_full_attention"
        #     similar to the previous option except that each
        #     position in the inputs portion can see the
        #     entire inputs portion.  This removes the challenge of
        #     autoregressively predicting the inputs portion.
        prepend_mode="none",
        # Scheduled sampling is interesting for auto-regressive models.
        # It runs an additional step using the generated output as autoregressive
        # targets, which can improve the models inference results later. The
        # parameter scheduled_sampling_prob determines with what probability
        # will such additional step be run. It's turned off (0.0) by default.
        # This probability will exponentially warm up for the number of
        # steps determined by scheduled_sampling_warmup_steps.
        # The tensor used for the n-th pass will consist of outputs from
        # the (n-1)-th pass mixed with gold truth, with the proportion of gold
        # determined by scheduled_sampling_gold_mixin_prob. Control the number
        # of passes with scheduled_sampling_num_passes.
        scheduled_sampling_prob=0.0,
        scheduled_sampling_method="parallel",  # parallel or sequential.
        scheduled_sampling_warmup_steps=50000,
        scheduled_sampling_gold_mixin_prob=0.5,
        scheduled_sampling_num_passes=1,
        scheduled_sampling_warmup_schedule="exp",  # exp, linear, or sigmoid.

        # This setting controls whether to copy variables around in a daisy chain
        # (if true) or leave their placement to TensorFlow. It only affects multi
        # device training and mostly should be turned on for performance. One
        # exception are recurrent models: with dynamic loops it must be off.
        daisy_chain_variables=True,
        # If True in PREDICT mode, then last-position-only optimizations are not
        # used.
        force_full_predict=False,
        # Set this for pure model parallelism.  There is only one data shard.
        no_data_parallelism=False,
        # dtype used for activations. - "float32" or "bfloat16"
        # activation_dtype="bfloat16" currently only works on TPU.
        #    It lowers activation-memory usage
        #    and does not appear to affect quality.
        #    You can train on TPU with activation_dtype="bfloat16" and evaluate
        #    on CPU/GPU with activation_dtype="float32"
        activation_dtype="float32",
        # dtype used for parameters: "float32" or "bfloat16"
        # bfloat16 currently only works with optimizer="adafactor".
        #   The savings in memory allow for training larger models.
        #   Weights are encoded as (w*128)^8, using pseudostochastic
        #   roundoff.  Initial experiments show that model quality is similar
        #   to baseline for about 3M training steps, but worse thereafter.
        weight_dtype="float32",
        # Directory containing a checkpoint for a pretrained model. This will only
        # be used if a new run is being started. Parameters not found in the
        # pretrained model will be randomly initialized. Superfluous parameters in
        # the pretrained model will be ignored.
        pretrained_model_dir="",
        # Threshold used for two cases: the primary task probability for the
        # constant mixing schedule, and the exponential schedule limit for when
        # mixing should stop (eg: 0.5 means stop at 50-50 mixing, 0.8 means stop
        # at 20-80 mixing for the primary-others mixing case.)
        multiproblem_schedule_threshold=0.5,
        # For more than 2 tasks, we may want to specify per-task thresholds here.
        # In that case, this needs to be a string with as many floating point
        # numbers as the number of tasks in the multi-problem. These numbers
        # are later normalized to add up to 1 and taken as probabilities for
        # each task. This enforces a constant mixing schedule and if this is
        # empty then the threshold from above is used for the first task and
        # the other tasks get the remaining probability split uniformly.
        multiproblem_per_task_threshold="",
        # The number of examples at which the proportion of the mixed in datasets
        # is multiproblem_schedule_threshold
        multiproblem_schedule_max_examples=1e7,
        # When training multiproblems, we can mix the data according to different
        # schedules. Example: a constant schedule mixing 20-80 between the primary
        # and other tasks.
        # A list of supported schedules can be found in
        # `data_generators.multi_problem.py`.
        multiproblem_mixing_schedule="constant",
        # A boolean that decides whether input sequence losses and target label
        # losses in classification problems should be reweighted.
        multiproblem_reweight_label_loss=False,
        # How much weight the targets in classification problems receive. Inputs
        # receive 1 minus this weight.
        multiproblem_label_weight=0.5,
        # Hyperparameters for relative attention.
        # The maximum relative positional distance to learn an embedding for.
        max_relative_position=0,
        # If heads share the same relative embedding.
        heads_share_relative_embedding=False,
        # If relative embedding terms are added to values too.
        add_relative_to_values=False,
        # If enable the host_call which is executed every training step.
        # There could be a performance drop if host_call function is slow and
        # cannot keep up with the TPU-side computation.
        tpu_enable_host_call=False,
        # Pad batch dim of inputs to nearest multiple of batch multiple.
        pad_batch=False,
        # When true, do not evaluate on the language model data when running the
        # multiproblem since it can take a while. If False, set eval_steps to
        # something large like 6000 or 10000.
        multiproblem_target_eval_only=False,
        # Max out the vocab size to a power of 2 for efficiency and to reserve
        # extra space in the vocabulary for new task ids and label classes.
        multiproblem_vocab_size=-1,
        # When using multiproblem with generation tasks, need to truncate the
        # inputs and targets manually before concatenating them.
        multiproblem_max_input_length=-1,
        multiproblem_max_target_length=-1,
        # If positive, makes training targets fixed-length in MultiProblem.
        multiproblem_fixed_train_length=-1,
        # Load weights from a second model. For instance, when using
        # pre-trained weights, you might want to initialize the encoder
        # and decoder by loading different models.
        warm_start_from_second="",
        # Area attention hyper parameters
        area_value_mode="none",
        area_key_mode="none",
        # Using area attention for the number of layers from the bottom
        num_area_layers=0,
        max_area_width=1,
        max_area_height=1,
        memory_height=1,
        # Whether to use GPU automatic mixed precision (via graph rewrite)
        gpu_automatic_mixed_precision=False,
    )


def transformer_base_v1():
    """Set of hyperparameters."""
    hparams = basic_params1()
    hparams.norm_type = "layer"
    hparams.hidden_size = 512
    hparams.batch_size = 4096
    hparams.max_length = 256
    hparams.clip_grad_norm = 0.  # i.e. no gradient clipping
    hparams.optimizer_adam_epsilon = 1e-9
    hparams.learning_rate_schedule = "legacy"
    hparams.learning_rate_decay_scheme = "noam"
    hparams.learning_rate = 0.1
    hparams.learning_rate_warmup_steps = 4000
    hparams.initializer_gain = 1.0
    hparams.num_hidden_layers = 6
    hparams.initializer = "uniform_unit_scaling"
    hparams.weight_decay = 0.0
    hparams.optimizer_adam_beta1 = 0.9
    hparams.optimizer_adam_beta2 = 0.98
    hparams.num_sampled_classes = 0
    hparams.label_smoothing = 0.1
    hparams.shared_embedding_and_softmax_weights = True
    hparams.symbol_modality_num_shards = 16

    # Add new ones like this.
    hparams.add_hparam("filter_size", 2048)
    # Layer-related flags. If zero, these fall back on hparams.num_hidden_layers.
    hparams.add_hparam("num_encoder_layers", 0)
    hparams.add_hparam("num_decoder_layers", 0)
    # Attention-related flags.
    hparams.add_hparam("num_heads", 8)
    hparams.add_hparam("attention_key_channels", 0)
    hparams.add_hparam("attention_value_channels", 0)
    hparams.add_hparam("ffn_layer", "dense_relu_dense")
    hparams.add_hparam("parameter_attention_key_channels", 0)
    hparams.add_hparam("parameter_attention_value_channels", 0)
    # All hyperparameters ending in "dropout" are automatically set to 0.0
    # when not in training mode.
    hparams.add_hparam("attention_dropout", 0.0)
    hparams.add_hparam("attention_dropout_broadcast_dims", "")
    hparams.add_hparam("relu_dropout", 0.0)
    hparams.add_hparam("relu_dropout_broadcast_dims", "")
    hparams.add_hparam("pos", "timing")  # timing, none
    hparams.add_hparam("nbr_decoder_problems", 1)
    hparams.add_hparam("proximity_bias", False)
    hparams.add_hparam("causal_decoder_self_attention", True)
    hparams.add_hparam("use_pad_remover", True)
    hparams.add_hparam("self_attention_type", "dot_product")
    hparams.add_hparam("conv_first_kernel", 3)
    hparams.add_hparam("attention_variables_3d", False)
    hparams.add_hparam("use_target_space_embedding", True)
    # These parameters are only used when ffn_layer=="local_moe_tpu"
    hparams.add_hparam("moe_overhead_train", 1.0)
    hparams.add_hparam("moe_overhead_eval", 2.0)
    hparams.moe_num_experts = 16
    hparams.moe_loss_coef = 1e-3
    # If specified, use this value instead of problem name in metrics.py.
    # This is useful for programs that can automatically compare experiments side
    #   by side based on the same metric names.
    hparams.add_hparam("overload_eval_metric_name", "")
    # For making a transformer encoder unidirectional by using masked
    # attention.
    hparams.add_hparam("unidirectional_encoder", False)
    # For hard attention.
    hparams.add_hparam("hard_attention_k", 0)
    hparams.add_hparam("gumbel_noise_weight", 0.0)
    return hparams


def transformer_base_v2():
    """Set of hyperparameters."""
    hparams = transformer_base_v1()
    hparams.layer_preprocess_sequence = "n"
    hparams.layer_postprocess_sequence = "da"
    hparams.layer_prepostprocess_dropout = 0.1
    hparams.attention_dropout = 0.1
    hparams.relu_dropout = 0.1
    hparams.learning_rate_warmup_steps = 8000
    hparams.learning_rate = 0.2
    return hparams


def transformer_base_v3():
  """Base parameters for Transformer model."""
  # Update parameters here, then occasionally cut a versioned set, e.g.
  # transformer_base_v2.
  hparams = transformer_base_v2()
  hparams.optimizer_adam_beta2 = 0.997
  # New way of specifying learning rate schedule.
  # Equivalent to previous version.
  hparams.learning_rate_schedule = (
      "constant*linear_warmup*rsqrt_decay*rsqrt_hidden_size")
  hparams.learning_rate_constant = 2.0
  return hparams


def transformer_base():
  """Base parameters for Transformer model."""
  hparams = transformer_base_v3()
  return hparams