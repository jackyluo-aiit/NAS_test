import abc
import tensorflow as tf

class PluginBase(object, metaclass=abc.ABCMeta):
    # metaclass: the stuff that creates classes.

    @abc.abstractmethod
    def load(self, input):
        """Retrieve data from the input source and return an object."""
        return

    @abc.abstractmethod
    def save(self, output, data):
        """Save the data object to the output."""
        return


x=tf.Variable([[[1,1],
                 [2,2]],
               [[3,3],
                [4,4]]],tf.float32)
print(x.shape)
x=tf.pad(x,[[0,0],[1,0],[0,0]])
print(x)

