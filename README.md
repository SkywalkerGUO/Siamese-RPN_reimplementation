# Modifications
These files are code that we rewrote:

loss.py: calculate cross-entropy loss and smoothL1 loss of RPN

anchor.py: some functions about handling anchor. 

    intersection_over_union: calculate iou
    nms: implement Non-Maximum Suppression
    generate_anchors: generate all anchors with size of 19*19

network: implement SiamRPN network

tracker: implement SiamRPN tracker, which is used when testing OTB2013 and OTB2015

siamrpn_tracker: another implementation of SiamRPN tracker. We change the number of channels of convolution layers in siamrpn, and it is used when testing GOT-10K

Code Reference: https://github.com/HonglinChu/SiamTrackers
# Result
![image](https://github.com/SkywalkerGUO/Siamese-RPN_reimplementation/assets/94382041/f3769e57-ce0a-4581-8772-239f1ed2fa11)
![image](https://github.com/SkywalkerGUO/Siamese-RPN_reimplementation/assets/94382041/46595471-207e-47db-8b00-236f98bb8fc7)
![image](https://github.com/SkywalkerGUO/Siamese-RPN_reimplementation/assets/94382041/be014dd4-6804-42f2-9a97-79b3ac510e8c)
