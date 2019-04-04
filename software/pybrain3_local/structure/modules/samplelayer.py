#! /usr/bin/env python2.5
# -*- coding: utf-8 -*-


__author__ = ('Christian Osendorfer, osendorf@in.tum.de; '
              'Justin S Bayer, bayerj@in.tum.de')
             
    
from scipy import random
    
from pybrain3_local.structure.modules.neuronlayer import NeuronLayer
     
     
class SampleLayer(NeuronLayer):
    """Baseclass for all layers that have stochastic output depending on the 
    incoming weight."""
    
    
class BernoulliLayer(SampleLayer):
    
    def _forwardImplementation(self, inbuf, outbuf):
        outbuf[:] = inbuf <= random.random(inbuf.shape)
