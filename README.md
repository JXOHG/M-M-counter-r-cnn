**M&M Counter using recurrent Convolutional Neural Network**<br>
This project is for AISE3350: Cyber Physical Systems.<br>

It is recommended to have at least 10+ images for training and annotation.
There are a lot of dead code / prototyping code, but the code that you need are the following:<br>
<ul>
  <li>1. annotation tool.py: GUI tool for annotation and labelling.</li>
  <li>r-cnn.py: trains the r-cnn model to the annotation.json automatically saved from step 1. Usses transfer learning.</li>
  <li>predictor.py: applies the model trained automatically saved from step 2 to a new image, and uses GUI to visualize the result. </li>
</ul>
