I worked on two models to translate American Sign Language to English.

The first one is an encoder-decoder model using LSTM cells, this code is what I am working in my capstone project, and I'd appreciate your feedback on it.
1 epoch took me 11 hrs to train 15,000 videos in 1 GPU.

The second model is what I intended to present as final project for this class. I intended to use Conv3D encoder + LSTM decoder to translate video to
text. I reach to the point to obtain a video vector representation, but couldn't transform this vector as a hidden state for the first LSTM decoder cell.

Utils.py and utilsConv3D.py contains video processing and data loader functions. 
