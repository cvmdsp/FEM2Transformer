# FEM2Transformer

  Frequency-Domain Enhanced Multi-scale Multi-strategy Window Transformer for Hyperspectral-Multispectral Image Fusion.

  The code is available at https://github.com/cvmdsp/FEM2Transformer.

  ## Requirements

  - Python 3.7+, PyTorch 1.7+, BasicSR, einops, NumPy, SciPy, OpenCV

  The steps for using the FEM2Transformer code are as follows:

  **Data Preprocessing**: Create two folders named "HSI" and "RGB" in both the "Train" and "Test" directories. Put the
  training and testing images into the corresponding "Train" and "Test" folders, where HSI refers to HR-HSI (GT), and
  RGB refers to HR-MSI.

  **Model Training**: Directly run the Train.py file.

  **Model Testing**: Create the folder "\Result\f4\FEM2Transformer" and run the Test.py file.
