# Internship task in Winstars

### Summary
This repository contains a project on semantic segmentation using airbus ship detection dataset.


- **Task:** Semantic Segmentation <br>
- **Data:** <a href='https://www.kaggle.com/c/airbus-ship-detection/data'>Link</a> <br>
- **Neural Network:** U-Net like architecture
- **Scoring function:** Dice score (f1 score)

### How to run:
* Download data 
  - Download <a href='https://www.kaggle.com/c/airbus-ship-detection/data'>Kaggle Competition</a> data 
  - Download mask images (to save time) <a href='https://drive.google.com/file/d/1jzMSN3sUtAdGSvWUAp9WS_oDfoMVx6YD/view?usp=sharing'>Link</a>
* Prepare folder structure
  - Create `data` folder and extract Kaggle data into it
  - Create `masks_v2` folder inside `data` folder. Extract mask images into `masks_v2` folder
* Run `eda.ipynb` to create training dataframe
* ### To train
```
python train.py --backbone=mobilenetv2
```
`--backbone` - choose UNet encoder backbone <br>
`python train.py -h` for more details
* ### To predict
```
python inference.py --download --visualize_inference
```
`--download` - Download pre-trained model <br>
`--compare_to_gt` - Compare prediction to Ground Truth <br>
`--visualize_inference` - Visualize 10 inference results <br>
`--predict_all` - <span style='color: red'>Takes a long time!</span> Predict on all test images  <br>
`--show_submission` - Visualize random predictions from RLE encoded submission.scv <br>
`python inference.py -h` for more details
### Model results

#### Model info
Trained 30 epochs, on 5k images (4k training and 1k validation), batch size 10, using basic image augmentation to aid in training.
Metric - f1-score, loss function - binary crossentropy.

#### Model training history
<img src="results\f1-score.jpg"/>
<img src="results\loss.jpg"/>

#### Model evaluation
<img src="results\eval.jpg"/>

#### Prediction vs ground truth
<img src="results\validation\a8af12f5b.jpg"/>
<img src="results\validation\e5cb861f3.jpg"/>

#### Inference example
<img src="results\inference\3d75a5157.jpg"/>
<img src="results\inference\582ed5b82.jpg"/>
<img src="results\inference\d6cf01e6f.jpg"/>

### Project Structure:
```bash
├───data # Data folder
│   └─── masks_v2 # RLE decoded masks
├───EDA
│   └───eda.ipynb # EDA and initial data prep
├───train.py # Define model
├───inference.py # Model inference 
├───results # Store model execution results
│   ├───inference # Model inference results
│   ├───validation # Model validation data predictions
│   └───submission # Images from predicted RLE
├───checkpoints # Folder with best model checkpoints
├───constants.py # Declare variables
├───data_prep.py # Create data generators and augment pictures
├───helper_funcs.py
├───metrics.py
├───.gitignore
└───requirements.txt
```
### Kaggle Notebook
Used it to run all model training <a href='https://www.kaggle.com/jeniagerasimov/airbus-semantic-segmantation'>View</a>

### Conclusion:
Project was very challenging as I had 0 previous experience with neural networks and keras, but it was very fun nonetheless.
