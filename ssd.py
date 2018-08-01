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


if __name__ == '__main__':
    print(1)
