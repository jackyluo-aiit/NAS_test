import abc
from abc_tutorials.abc_base import PluginBase


class RegisteredImplementation1(object):
    def load(self, input):
        return input.read()

    def save(self, output, data):
        return output.write(data)


PluginBase.register(RegisteredImplementation1)# regsiter RegisteredImplementation1 as an instance of PluginBase.


if __name__ == '__main__':
    print ('Subclass:', issubclass(RegisteredImplementation1, PluginBase))
    print ('Instance:', isinstance(RegisteredImplementation1(), PluginBase))