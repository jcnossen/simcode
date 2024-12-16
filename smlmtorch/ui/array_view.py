# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 18:33:21 2022

@author: jelmer
"""


def isnotebook():
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter


def image_view(img, **kwargs):
    if isnotebook():
        raise RuntimeError('No array viewer implemented for jupyter notebooks')
    
    import smlmtorch.ui.array_view_pyqt as pyqt_ui
    return pyqt_ui.image_view(img, **kwargs)
        

def array_plot(data, **kwargs):
    if isnotebook():
        raise RuntimeError('No array viewer implemented for jupyter notebooks')
    
    import smlmtorch.ui.array_viewer_pyqt as pyqt_ui
    return pyqt_ui.array_plot(data, **kwargs)
        
