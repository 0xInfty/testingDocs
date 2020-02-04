# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 19:45:05 2020

@author: Valeria
"""

from lantz.drivers.examples import LantzSignalGenerator

inst = LantzSignalGenerator('TCPIP::localhost::5678::SOCKET')
inst.initialize()
print(inst.idn)
inst.finalize()