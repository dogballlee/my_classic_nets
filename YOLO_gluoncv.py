import mxnet as mx
from gluoncv import model_zoo, data, utils
from matplotlib import pyplot as plt

net = model_zoo.get_model('yolo3_darknet53_voc', pretrained=True)

x, img = data.transforms.presets.yolo.load_test('D:/download/GSG9.jpg', 416)
print('Shape of pre-processed image:', x.shape)
class_IDs, scores, bounding_boxes = net(x)
ax = utils.viz.plot_bbox(img, bounding_boxs[0], scores[0], class_IDs[0], class_names=net.classes)
plt.show()
