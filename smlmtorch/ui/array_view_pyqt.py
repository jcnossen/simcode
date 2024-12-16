"""
super light n-d array viewer (like napari) to help debugging
author: jelmer cnossen 2021/2022
license: public domain
"""
from pyqtgraph import ImageItem
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QSlider,QMenuBar
from PyQt5 import QtWidgets
import pyqtgraph as pg
from smlmtorch.ui.qt_util import needs_qt

import numpy as np



    

#class ImageViewWindow(QtWidgets.QMainWindow):
class PlotViewWindow(QtWidgets.QDialog):
    def __init__(self, data, x=None, event_listener = None, axis_labels=None, plot_dim = None, legend = None, *args, **kwargs):
        super(PlotViewWindow, self).__init__(*args, **kwargs)
        
        if plot_dim is not None:
            raise NotImplementedError()

        if plot_dim  is not None:            
            slider_shape = np.take(data, 0, plot_dim).shape
        else:
            slider_shape = data.shape
        
        self.setWindowTitle('Plot Viewer')
        self.event_listener = event_listener
        
        self.menu=QMenuBar()
        ltop = QtWidgets.QVBoxLayout()
       
        layout = QtWidgets.QGridLayout()#QtWidgets.QVBoxLayout()
        ltop.addLayout(layout)
        ltop.setMenuBar(self.menu)
        #btnstart = QtWidgets.QPushButton("Take ZStack")
        #btnstart.setMaximumSize(100,32)
        #btnstart.clicked.connect(self.recordZStack)

        ctlLayout = QtWidgets.QGridLayout()
        #ctlLayout = QtWidgets.QHBoxLayout()

        sliders=[]        
        for i in range( len(data.shape)-1 ):
            slider = QSlider(Qt.Horizontal)
            slider.setFocusPolicy(Qt.StrongFocus)
            slider.setTickPosition(QSlider.TicksBothSides)
            slider.setTickInterval(10)
            slider.setSingleStep(1)
            slider.setMaximum(data.shape[i]-1)
            slider.valueChanged.connect(self.sliderChange)
            if axis_labels is not None:
                layout.addWidget(QtWidgets.QLabel(axis_labels[i]), i,0)
                layout.addWidget(slider, i, 1)#, 0, 0)
            else:
                layout.addWidget(slider, i, 0)#, 0, 0)
            sliders.append(slider)
        self.sliders=sliders

        #layout.addWidget(btnstart)#, 0, 0)

        self.info = QtWidgets.QLabel()
        ctlLayout.addWidget(self.info, 0, 2, Qt.AlignLeft)

        layout.addLayout(ctlLayout, len(data.shape)-1, 0)

        self.widget = pg.GraphicsLayoutWidget()
        ltop.addWidget(self.widget)

        self.plots = []
        if plot_dim is not None:
            for i in range(data.shape[plot_dim]):
                self.plots.append(self.widget.addPlot(title=legend[i] if legend is not None else None))
        else:
            self.plots = [self.widget.addPlot()]
            
        self.plot_dim = plot_dim
        self.curves = [None]*len(self.plots)    
        for i,p in enumerate(self.plots):
            p.setDownsampling(mode='peak')
            p.setAspectLocked(False)
            p.setClipToView(True)
            self.curves[i] = p.plot()
            
        self.axis_labels = axis_labels
        self.setLayout(ltop)
        self.data = data
        self.x = x
                
        self.sliderChange()

    def mouseClicked(self, event):
        pos = event.scenePos() # event.pos() depends on which item is clicked

        image_pos = self.imv.mapFromDevice(pos)
        #print(f'device pos: {pos}, image pos: {image_pos}')

        if self.event_listener is not None:
            self.event_listener('click', self, event, image_pos)

    def sliderChange(self):
        d = self.data
        ix=[]
        for s in self.sliders:
            ix.append(s.value())
            d=d[s.value()]
            
        for i,c in enumerate(self.curves):
            if self.x is not None:
                c.setData(self.x, d)
            else:
                c.setData(d)
            
        self.info.setText("["+','.join([str(i) for i in ix])+']')

    def update(self):
        ...

    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        #self.closeCamera()
        ...

        
    def closeEvent(self, event):
        #print('close event')
        #self.closeCamera()
        event.accept()

    

#class ImageViewWindow(QtWidgets.QMainWindow):
class ImageViewWindow(QtWidgets.QDialog):
    def __init__(self, img, event_listener = None, axis_labels = None, 
                 on_mouse_click = None,
                 on_slider_change = None, *args, **kwargs):
        super(ImageViewWindow, self).__init__(*args, **kwargs)
        
        self.setWindowTitle('Image Viewer')
        
        self.on_slider_change = on_slider_change
        self.on_mouse_click = on_mouse_click

        self.menu=QMenuBar()
        ltop = QtWidgets.QVBoxLayout()
       
        layout = QtWidgets.QGridLayout()#QtWidgets.QVBoxLayout()
        ltop.addLayout(layout)
        ltop.setMenuBar(self.menu)
        #btnstart = QtWidgets.QPushButton("Take ZStack")
        #btnstart.setMaximumSize(100,32)
        #btnstart.clicked.connect(self.recordZStack)

        ctlLayout = QtWidgets.QGridLayout()
        #ctlLayout = QtWidgets.QHBoxLayout()

        sliders=[]        
        for i in range( len(img.shape)-2 ):
            slider = QSlider(Qt.Horizontal)
            slider.setFocusPolicy(Qt.StrongFocus)
            slider.setTickPosition(QSlider.TicksBothSides)
            slider.setTickInterval(10)
            slider.setSingleStep(1)
            slider.setMaximum(img.shape[i]-1)
            slider.valueChanged.connect(self.sliderChange)
            if axis_labels is not None:
                layout.addWidget(QtWidgets.QLabel(axis_labels[i]), i,0)
                layout.addWidget(slider, i, 1)#, 0, 0)
            else:
                layout.addWidget(slider, i, 0)#, 0, 0)
            sliders.append(slider)
        self.sliders=sliders

        #layout.addWidget(btnstart)#, 0, 0)

        self.info = QtWidgets.QLabel()
        ctlLayout.addWidget(self.info, 0, 2, Qt.AlignLeft)

        layout.addLayout(ctlLayout, len(img.shape)-2, 0)
        view = pg.ImageView()
        ltop.addWidget(view)

        self.view = view
        self.imv = view.imageItem
        self.scatterPlot = pg.ScatterPlotItem()
        view.addItem(self.scatterPlot)
        view.setPredefinedGradient('inferno')

        #w = pg.GraphicsLayoutWidget()
        #layout.addWidget(w)
        
        self.imv.scene().sigMouseClicked.connect(self.mouseClicked)    
        #image_pos = self.getView().mapViewToScene(event.pos())

        #v = w.addViewBox(row=0, col=0)
        #self.imv = ImageItem()
        #v.addItem(self.imv)

        self.setLayout(ltop)
         
        #v = w.addViewBox(row=0, col=1)
        #self.zstackImv = ImageItem()
        #v.addItem(self.zstackImv)

        self.imv.setImage(img)
        self.data = img        
        self.sliderChange()

    def mouseClicked(self, event):
        pos = event.scenePos() # event.pos() depends on which item is clicked

        image_pos = self.imv.mapFromDevice(pos)
        #print(f'device pos: {pos}, image pos: {image_pos}')

        if self.on_mouse_click is not None:
            self.on_mouse_click(self, event, image_pos)

    @property
    def indices(self):
        return [s.value() for s in self.sliders]

    def sliderChange(self):
        d = self.data
        ix=[]
        for s in self.sliders:
            ix.append(s.value())
            d=d[s.value()]
        self.imv.setImage(d.T)
        self.info.setText("["+','.join([str(i) for i in ix])+']')

        if self.on_slider_change is not None:
            self.on_slider_change(ix, self)

    def drawPoints(self, pts=None, clear=True):
        if clear:
            self.scatterPlot.clear()
        if pts is not None:
            self.scatterPlot.setData(pos=pts, size=10, pen=pg.mkPen('b', width=2))

    def update(self):
        ...

    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        #self.closeCamera()
        ...
        
    def closeEvent(self, event):
        #print('close event')
        #self.closeCamera()
        event.accept()



@needs_qt
def image_view(img, title=None, modal=True, parent=None, **kwargs):
    if getattr(img, "cpu", None): # convert torch tensors without importing torch
        img = img.detach().cpu().numpy()
    w = ImageViewWindow(np.array(img), parent=parent, **kwargs)
    w.setModal(modal)
    if title is not None:
        w.setWindowTitle(title)
    if modal:
        w.exec_()
        return 

    w.show()
    return w



@needs_qt
def array_plot(data, title=None, modal=True, parent=None, **kwargs):
    if getattr(data, "cpu", None): # convert torch tensors without importing torch
        data = data.detach().cpu().numpy()
    w = PlotViewWindow(np.array(data), parent=parent, **kwargs)
    w.setModal(modal)
    if title is not None:
        w.setWindowTitle(title)
    if modal:
        w.exec_()
        return 

    w.show()
    return w


view = image_view

if __name__ == '__main__':
    
    # Test image view
    img = np.random.uniform(0, 100, size=(20,5,200,200)).astype(np.uint8)

    def click_handler(wnd : ImageViewWindow, ev, pos):
        w = 10
        wnd.view.addItem(pg.RectROI((pos.x() - w/2, pos.y()-w/2), (w,w)))

    def draw_random_pts(ix, wnd):
        wnd.drawPoints(np.random.uniform(0, img.shape[-1], size=(10,2)))
    
    v=view(img, axis_labels=['a','b'], 
           on_mouse_click = click_handler, 
           on_slider_change = draw_random_pts)
    #array_plot(img, x=np.linspace(0,1,img.shape[-1]), axis_labels=['a', 'b','c'])
    
    
    