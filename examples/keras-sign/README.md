# Sign Language Classifier

In this problem the source data is 28x28 pixel grayscale images of a hands making sign language (there are only 24 categories as j and z require movement).  The training and test data is stored in a CSV as pixel values between 0 & 255.  The challenge is to create an ML classifier that performs the best on the test dataset.

`perceptron.py` is very simple and lacks normalization.  Your first step should likely be to normalize the input data to be between 0 & 1, then create a Concurrent Neural Net but be careful not to overfit.  Transfer learning, data augmentation, and increasing the size of your dataset are more advanced approaches to achieve higher accuracy. 

## Resources

* https://google.com
* https://keras.io
* https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
* http://empslocal.ex.ac.uk/people/staff/np331/index.php?section=FingerSpellingDataset