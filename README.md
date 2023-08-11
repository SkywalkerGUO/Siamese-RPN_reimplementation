These files are code that we rewrote:

loss.py: calculate cross-entropy loss and smoothL1 loss of RPN

anchor.py: some functions about handling anchor. 
    -intersection_over_union: calculate iou
    -nms: implement Non-Maximum Suppression
    -generate_anchors: generate all anchors with size of 19*19

network: implement SiamRPN network

tracker: implement SiamRPN tracker, which is used when testing OTB2013 and OTB2015

siamrpn_tracker: another implementation of SiamRPN tracker. We change the number of channels of convolution layers in siamrpn, and it is used when testing GOT-10K

Code Reference: https://github.com/HonglinChu/SiamTrackers
