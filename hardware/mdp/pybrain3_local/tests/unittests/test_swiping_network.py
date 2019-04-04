"""

Build a 2-dimensional BorderSwipingNetwork:
    
    >>> n = buildSwipingNetwork(2)
        
Check its gradient:

    >>> from pybrain3_local.tests import gradientCheck
    >>> gradientCheck(n)
    Perfect gradient
    True

Try writing it to an xml file, reread it and determine if it looks the same:
    
    >>> from pybrain3_local.tests import xmlInvariance
    >>> xmlInvariance(n)
    Same representation
    Same function
    Same class
    
"""

__author__ = 'Tom Schaul, tom@idsia.ch'


from pybrain3_local3 import ModuleMesh, LinearLayer
from pybrain3_local3.structure.networks import BorderSwipingNetwork
from pybrain3_local3.tests import runModuleTestSuite
    
    
def buildSwipingNetwork(dimensions = 3):
    d = tuple([2] * dimensions)    
    inmesh = ModuleMesh.constructWithLayers(LinearLayer, 1, d, 'in')
    hmesh = ModuleMesh.constructWithLayers(LinearLayer, 1, tuple(list(d)+[2**len(d)]), 'h')
    outmesh = ModuleMesh.constructWithLayers(LinearLayer, 1, d, 'out')
    return BorderSwipingNetwork(inmesh, hmesh, outmesh)



if __name__ == "__main__":
    runModuleTestSuite(__import__('__main__'))

