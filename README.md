## Final Year Project 2
Title: A Robust Speaker-aware Speech Separation Technique using Composite Speech Models 

Disclaimer: This project borrows reference from another repository - [Google Audiovisual Model](https://github.com/ktam069/speech_separation_modified "Google Audiovisual Model").

### Dependencies
The project runs on Python 3.6. Please download and pip install: 
- keras 2.3.1
- tensorflow 1.13.1
- librosa
- youtube-dl
- pytest
- sox 
- ffmpeg 

### Instructions
------------
#### Dataset 
To prepare the dataset, navigate into the **data** folder and run the following command: 

    python download_dataset.py

Several configurations can be set in the *download_dataset.py* script, such as number of video clips and normalizing audio. 

#### Model Training
To train the audio-only speech separation model, navigate to **model/model_v1** and run the following command: 


    python AO_train.py

To train the audiovisual speech separation model, navigate to **model/model_v2** and run the following command:


    python AV_train.py


#### Model Inference
Copy and paste the saved H5 model file into **saved_xx_models** folders. Afterwards, modify the file path in the *predict_video.py* script and run the following command:


    python predict_video.py

The estimated speech files will be in the **pred** folder.



