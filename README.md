## Final Year Project 2
Title: A Robust Speaker-aware Speech Separation Technique using Composite Speech Models 

Disclaimer: This project borrows reference from another repository - [Google Audiovisual Model](https://github.com/ktam069/speech_separation_modified "Google Audiovisual Model").

### Dependencies
The project runs on Python 3.6. Please download and pip install packages in *requirements.txt*

### Instructions
------------
#### Dataset 
To prepare the dataset, navigate into the **data** folder and run the following command: 

    python download_dataset.py

Several configurations can be set in the *download_dataset.py* script, such as number of video clips and normalizing audio. 

For audiovisual models, navigate to **model/pretrain_model** and generate the face embeddings. Run the following command: 

    python pretrain_load_test.py

Then, rename the output folder to **face_emb** and move it to **data/video**.


#### Model Training
To train the audio-only speech separation model, navigate to **model/model_v1** and run the following command: 


    python AO_train.py

To train the audiovisual speech separation model, navigate to **model/model_v2** and run the following command:


    python AV_train.py



#### Model Inference
Copy and paste the saved H5 model file into **saved_AV/AO_models** folders. For testing data, change the **dl_from_training** variable to **False** in the *download_dataset.py* script.

Afterwards, modify the file path in the *AV/AO_predict_video.py* script and run the following command:


    python AO_predict_video.py
    python AV_predict_video.py

The estimated speech files will be in the respective **pred** folder.



