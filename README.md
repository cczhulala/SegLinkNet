# SegLinkNet
We achieved left ventricular segmentation of cardiac MRI images by improving SegNet, which we call SegLinkNet. The main contributions are as follows: 
(1) the addition of hop connections; 
(2) Modify the ReLu activation function to the SeLu activation function

Note:
3, which contains three inference files, one for the original SegNet, one for Bayesian SegNet, and one for SegLinkNet. Rename this file to inference and
replace the file in tfmodel to train the model.
