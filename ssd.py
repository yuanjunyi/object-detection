import gluonbook as gb
from mxnet import autograd, contrib, gluon, image, init, nd
from mxnet.gluon import loss as gloss, nn
import time


def cls_predictor(num_anchors, num_classes):
    '''
    num_anchors: number of anchors for each pixel(x, y)
    num_classes: number of classes in the dataset

    For each pixel(x, y), we will generate num_anchors anchors.
    For each anchor, we need to classify it in (num_classes + 1) classes.
    Class 0 represents background. So there are
    num_anchors * (num_classes+1) channels in total.

    padding is used to preserve the width and height.
    '''
    return nn.Conv2D(channels=num_anchors*(num_classes+1),
                     kernel_size=3,
                     padding=1)


def bbox_predictor(num_anchors):
    '''
    For each anchor, we need to predict its offset to the true
    bounding box. The offset is described by a vecotr of 4,
    representing the offset along x and y axes for the bottom-left
    and top-right points.
    '''
    return nn.Conv2D(channels=num_anchors*4, kernel_size=3, padding=1)


def flatten_pred(pred):
    '''
    For a predition of shape (batch_size, channels, height, width),
    transfer it to a 2d array of batch_size row and height*width*channels column.

    E.g. batch_size = 1, channels=3, height=2, width=3
    Without transpose, pred looks like:
    R1 R2 R3
    R4 R5 R6

    G1 G2 G3
    G4 G5 G6

    B1 B2 B3
    B4 B5 B6

    flattened looks like:
    R1 R2 R3 R4 R5 R6 G1 G2 G3 G4 G5 G6 B1 B2 B3 B4 B5 B6

    With transpose(axes=(0, 2, 3, 1)), pred looks like
    (R1,G1,B1) (R2,G2,B2) (R3,G3,B3)
    (R4,G4,B4) (R5,G5,B5) (R6,G6,B6)

    flattened looks like:
    R1 G1 B1 R2 G2 B2 R3 G3 B3 R4 G4 B4 R5 G5 B5 R6 B6 G6
    '''
    return pred.transpose(axes=(0, 2, 3, 1)).flatten()


def concat_preds(preds):
    '''
    flatten_pred(p) makes p a 2d array of batch_size row.
    concat extends each row.
    In the end, return a 2d array of batch_size row.

    Applying * on any iterable object like list, unpack the elements.
    concat([a,b,c]) raises assertion error because [a,b,c] is not an ndarray.
    concat(*[a,b,c]) equals concat(a,b,c) and passes.
    '''
    return nd.concat(*[flatten_pred(p) for p in preds], dim=1)


def down_sample_blk(num_filters):
    '''
    nn.MaxPool2D(2) halves the height and width of the input.

    Pass the output from down_sample_blk to cls_predictor,
    the 3x3 conv in cls_predictor actually covers a 10x10 area in
    the input of down_sample_blk.

    E.g.
    x x x                required input for a 3x3 conv in cls_predictor
    y y y y y y          before MaxPool2D(2)
    z z z z z z z z      required input for a 3x3 conv in down_sample_blk
    0 1 2 3 4 5 6 7 8 9  required input for a 3x3 conv in down_sample_blk
    '''
    blk = nn.HybridSequential()
    for _ in range(2):
        blk.add(nn.Conv2D(channels=num_filters, kernel_size=3, padding=1),
                nn.BatchNorm(in_channels=num_filters),
                nn.Activation('relu'))
    blk.add(nn.MaxPool2D(2))
    blk.hybridize()
    return blk


def body_blk():
    blk = nn.HybridSequential()
    for num_filters in [16, 32, 64]:
        blk.add(down_sample_blk(num_filters))
    return blk


def get_blk(i):
    if i == 0:
        blk = body_blk()
    elif i == 4:
        blk = nn.GlobalMaxPool2D()
    else:
        blk = down_sample_blk(128)
    return blk


def single_scale_forward(x, blk, size, ratio, cls_predictor, bbox_predictor):
    y = blk(x)
    anchor = contrib.ndarray.MultiBoxPrior(y, sizes=size, ratios=ratio)
    cls_pred = cls_predictor(y)
    bbox_pred = bbox_predictor(y)
    return (y, anchor, cls_pred, bbox_pred)


class TinySSD(nn.Block):
    def __init__(self, num_classes, **kwargs):
        super(TinySSD, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.num_anchors = 4
        self.sizes = [[0.2, 0.272], [0.37, 0.447],
                      [0.54, 0.619], [0.71, 0.79], [0.88, 0.961]]
        self.ratios = [[1, 2, 0.5]] * 5

        for i in range(5):
            setattr(self, 'blk_%d' % i, get_blk(i))
            setattr(self, 'cls_%d' % i, cls_predictor(self.num_anchors,
                                                      self.num_classes))
            setattr(self, 'bbox_%d' % i, bbox_predictor(self.num_anchors))

    def forward(self, x):
        anchors, cls_preds, bbox_preds = [None] * 5, [None] * 5, [None] * 5
        for i in range(5):
            x, anchors[i], cls_preds[i], bbox_preds[i] = single_scale_forward(
                x, getattr(self, 'blk_%d' % i), self.sizes[i], self.ratios[i],
                getattr(self, 'cls_%d' % i), getattr(self, 'bbox_%d' % i))
        return (nd.concat(*anchors, dim=1),
                concat_preds(cls_preds).reshape((0, -1, self.num_classes + 1)),
                concat_preds(bbox_preds))


def calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks):
    cls_loss = gloss.SoftmaxCrossEntropyLoss()
    bbox_loss = gloss.L1Loss()
    cls = cls_loss(cls_preds, cls_labels)
    bbox = bbox_loss(bbox_preds * bbox_masks, bbox_labels * bbox_masks)
    return cls + bbox


def cls_metric(cls_preds, cls_labels):
    return (cls_preds.argmax(axis=-1) == cls_labels).mean().asscalar()


def bbox_metric(bbox_preds, bbox_labels, bbox_masks):
    return (bbox_labels - bbox_preds * bbox_masks).abs().mean().asscalar()


if __name__ == '__main__':
    batch_size = 32
    train_data, test_data = gb.load_data_pikachu(batch_size)
    '''
    train_data.data_shape is (3, 256, 256)
    test_data.data_shape is (1, 5)

    1 is the number of bbox in each image in pikachu dataset.
    We need to make sure each image has the same number of bbox.
    For those have fewer bboxes, fill illegal bbox to reach this number
    so that images can be processed in batch.

    Each bbox has 5 elements [class, x, y, x, y]. class -1 represnets
    illegal.

    Each image is required to have at least 3 bbox in GPU implementation.
    '''
    train_data.reshape(label_shape=(3, 5))
    ctx = gb.try_gpu()
    net = TinySSD(num_classes=2)
    net.initialize(init=init.Xavier(), ctx=ctx)
    trainer = gluon.Trainer(net.collect_params(),
                            'sgd', {'learning_rate': 0.1, 'wd': 5e-4})

    for epoch in range(5):
        acc, mae = 0, 0
        train_data.reset()  # Resets the iterator to the beginning of the data.
        tic = time.time()
        for i, batch in enumerate(train_data):
            # batch.data is a list of length 1
            X = batch.data[0].as_in_context(ctx)
            Y = batch.label[0].as_in_context(ctx)
            with autograd.record():
                anchors, cls_preds, bbox_preds = net(X)
                bbox_labels, bbox_masks, cls_labels = contrib.nd.MultiBoxTarget(
                    anchors, Y, cls_preds.transpose(axes=(0, 2, 1)))
                l = calc_loss(cls_preds, cls_labels,
                              bbox_preds, bbox_labels, bbox_masks)

            l.backward()
            trainer.step(batch_size)
            acc += cls_metric(cls_preds, cls_labels)
            mae += bbox_metric(bbox_preds, bbox_labels, bbox_masks)
        print('epoch %d, class err %.2e, bbox mae %.2e, time %.1f sec' %
              (epoch, 1 - acc / (i + 1), mae / (i + 1), time.time() - tic))
