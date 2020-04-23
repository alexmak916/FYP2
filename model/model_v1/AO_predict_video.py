'''
# ==========================================================
# Script for running a prediction on an input mp4 video
# (using a pretrained v2 AV model)
# ==========================================================
'''

import sys
sys.path.append('../lib')
from tensorflow.keras.models import load_model
import os
import scipy.io.wavfile as wavfile
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import librosa

from keras.models import Model
from mtcnn.mtcnn import MTCNN
import cv2
import matplotlib.pyplot as plt		# used for testing (visualising) purposes only

from keras.models import load_model as load_model_facenet
import matplotlib.image as mpimg

'''Note that the code from the utils script is needed (found under model/lib/)'''
import utils

'''
# ==========================================================
'''

# super parameters
num_people = 2

# PATH
MODEL_PATH = './saved_AO_models/AOmodel-2p-001-0.71298.h5'	# 234K
MODEL_PATH_FACENET = '../pretrain_model/FaceNet_keras/facenet_keras.h5'

# Prediction output location
dir_path_pred = './pred/'
dir_path_mix = '../../data/audio/AV_model_database/mix/'

# Range of first partition and second partition (if the test video range is discontinuous)
PART_ONE_RANGE = (18, 21)
PART_TWO_RANGE = (101, 104)


'''
# ==========================================================
# Main function
# ==========================================================
'''

def main():	
	'''Run prediction with AV model'''
	run_predict(PART_ONE_RANGE, PART_TWO_RANGE)


'''
# ==========================================================
# Custom loss function
# ==========================================================
'''

import keras.backend as K

def audio_discriminate_loss2(gamma=0.1,beta = 2*0.1,num_speaker=2):
	def loss_func(S_true,S_pred,gamma=gamma,beta=beta,num_speaker=num_speaker):
		sum_mtr = K.zeros_like(S_true[:,:,:,:,0])
		for i in range(num_speaker):
			sum_mtr += K.square(S_true[:,:,:,:,i]-S_pred[:,:,:,:,i])
			for j in range(num_speaker):
				if i != j:
					sum_mtr -= gamma*(K.square(S_true[:,:,:,:,i]-S_pred[:,:,:,:,j]))

		for i in range(num_speaker):
			for j in range(i+1,num_speaker):
				#sum_mtr -= beta*K.square(S_pred[:,:,:,i]-S_pred[:,:,:,j])
				#sum_mtr += beta*K.square(S_true[:,:,:,:,i]-S_true[:,:,:,:,j])
				pass
		#sum = K.sum(K.maximum(K.flatten(sum_mtr),0))

		loss = K.mean(K.flatten(sum_mtr))

		return loss
	return loss_func

audio_loss = audio_discriminate_loss2

# super parameters
gamma_loss = 0.1
beta_loss = gamma_loss*2

'''
# ==========================================================
# Predict video
# ==========================================================
'''

def run_predict(part_one=PART_ONE_RANGE, part_two=PART_TWO_RANGE):
	
	'''Load pretrained model'''
	loss_func = audio_loss(gamma=gamma_loss,beta=beta_loss, num_speaker=num_people)
	AO_model = load_model(MODEL_PATH,custom_objects={"tf": tf, 'loss_func': loss_func})

	'''Load audio data'''
	loaded_file = 0
	for i, j in zip(range(part_one[0], part_one[1]), range(part_two[0], part_two[1])):
		try:
			audio_data = np.load(dir_path_mix+"mix-%05d-%05d.npy"%(i, j))
			loaded_file += 1
			print(audio_data.shape)

			'''check shape - first dim should be 298'''
			audio_data = audio_data[:298]
			if len(audio_data) < 298:
				a = np.zeros((298,257,2))
				a[:len(audio_data),:,:] = audio_data
				audio_data = a
			print(audio_data.shape)
			mix_expand = np.expand_dims(audio_data, axis=0)
			print(mix_expand.shape)

			print("===== Completed processing audio =====")

			'''Predict data'''
			cRMs = AO_model.predict(mix_expand)
			cRMs = cRMs[0]
		
			print("===== Completed predicting cRMs =====")

			'''Save output as wav'''
			for k in range(num_people):
				cRM = cRMs[:,:,:,k]
				assert cRM.shape == (298,257,2)
				F = utils.fast_icRM(audio_data,cRM)
				T = utils.fast_istft(F,power=False)
				filename = dir_path_pred+'%05d-%05d_pred_output%d.wav'%(i,j,k)
				wavfile.write(filename,16000,T)
				print("%05d-%05d_pred_output%d.wav created"%(i,j,k))

			print("===== Completed saving output ===== \n")

		except FileNotFoundError:
			print('mix-%05d-%05d.npy is not found'%(i, j))

	print('num of processed audio : %d'%loaded_file)  


'''
# ==========================================================
# Test predicting on a video
# ==========================================================
'''

if __name__ == '__main__':
	main()

