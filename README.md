# Internship task in Winstars

### Summary
This repository contains a project on semantic segmentation using airbus ship detection dataset.


- **Task:** Semantic Segmentation <br>
- **Data:** <a href='https://www.kaggle.com/c/airbus-ship-detection/data'>Link</a> <br>
- **Neural Network:** U-Net like architecture
- **Scoring function:** Dice score (f1 score)

### Steps to reproduce:
1. Download data from kaggle competition (link in Summary)
2. Create folder **data** and extract all downloaded data there
3. Run **eda.ipynb** to create training dataframe and create masks from RLE's
4. (Optional) Download pre-trained model from Google Drive (<a href='https://drive.google.com/file/d/19NjEQGBiSSLyoaMuRPE4A20EDLtDRfuQ/view?usp=sharing'>Link</a>) and unpack into **checkpoints** folder 
5. Change function parameters in main.py to either train or load pre-trained and run main.py


### Project Structure:
```bash
├───EDA
│   └───eda.ipynb # EDA and initial data prep
├───models
│   ├───model.py # Define model
│   └───inference.py # Model inference 
├───results # Store model execution results
│   ├───inference # Model inference results
│   └───validation # Model validation data predictions
├───checkpoints # Folder with best model checkpoints
├───constants.py # Declare variables
├───data_prep.py # Create data generators and augment pictures
├───helper_funcs.py
├───metrics.py
├───main.py # Run this file to train or predict
├───.gitignore
└───requirements.txt
```

### Model prediction result
##### Train data
<img src="C:\Users\yevhe\PycharmProjects\WinStars-internship-project\results\validation\7fb130b9b.jpg"/>
<img src="C:\Users\yevhe\PycharmProjects\WinStars-internship-project\results\validation\296e05e3d.jpg"/>

##### Test data
<img src="C:\Users\yevhe\PycharmProjects\WinStars-internship-project\results\inference\d2a2994fc.png"/>
<img src="C:\Users\yevhe\PycharmProjects\WinStars-internship-project\results\inference\ff624dc46.png"/>
<img src="C:\Users\yevhe\PycharmProjects\WinStars-internship-project\results\inference\31a1b6366.png"/>

### Conclusion:
Project was very challenging as I had 0 previous experience with neural networks and keras, but it was very fun nonetheless.