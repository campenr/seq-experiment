import functools
from copy import deepcopy

class MyClass1(object):

    def __init__(self, my_list=None):

        self.my_list = my_list

    def __str__(self):

        return('<my_list=%s>' % self.my_list)

    pass


class MyClass2(object):

    def __init__(self, name, _class):

        self.name = name
        self._class = _class

    def __getitem__(self, item):

        new_class = deepcopy(self._class)
        new_class.my_list = self._class.my_list[item]

        return new_class


ix_dict = {'fx': MyClass2}

name = list(ix_dict.keys())[0]
print('name: ', name)
indexer = ix_dict[name]
print('indexer: ', indexer)

_indexer = functools.partial(indexer, name)
setattr(MyClass1, name, property(_indexer))

my_class = MyClass1([0,1,2,3,4,5])
print(my_class)

# test slicing
print(my_class.fx[2])

my_class2 = deepcopy(my_class)
print(my_class2.fx[2:4])




# _indexer = functools.partial(indexer, name)
# setattr(cls, name, property(_indexer, doc=indexer.__doc__))

