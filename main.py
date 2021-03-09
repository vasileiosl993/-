from __future__ import division
from __future__ import print_function

import sys
import argparse
import cv2
import editdistance
import os
from Loader import Loader, Batch, FilePaths
from Model import Model, DecoderType
from Preprocessor import preprocessingImg
from SpellChecker import correct_sentence




os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def train(model, loader):
	
	epoch = 0 
	bestCER = float('inf') 
	CERnoImprovement = 0 
	stopTraining = 5 
	
	while True:
		epoch += 1
		print('Epoch:', epoch)

		# train
		print('Train Neural Network')
		loader.trainingSet()
		while loader.nextSample():
			iteratorInfo = loader.getIteratorInfo()
			batch = loader.getNextSample()
			loss = model.trainBatch(batch)
			print('Batch:', iteratorInfo[0],'/', iteratorInfo[1], 'Loss:', loss)

		# validate
		CER, wordAccuracy = validate(model, loader)
		
		
		if CER < bestCER:
			print('Character error rate (CER) improved, save model')
			bestCER = CER
			CERnoImprovement = 0
			model.save()
			open(FilePaths.fnAccuracy, 'w').write('Validation character error rate (CER) of saved model: %f%%' % (CER*100.0))
		else:
			print('Character error rate (CER) not improved')
			CERnoImprovement += 1

		if CERnoImprovement >= stopTraining:
			print('Training Stopped, no more improvement since %d epochs.' % stopTraining)
			break


def validate(model, loader):
	
	print('Validate Neural Network')
	loader.validationSet()
	numCharErr = 0
	numCharTotal = 0
	numWordOK = 0
	numWordTotal = 0
	while loader.nextSample():
		iteratorInfo = loader.getIteratorInfo()
		print('Batch:', iteratorInfo[0],'/', iteratorInfo[1])
		batch = loader.getNextSample()
		recognized = model.inferBatch(batch)
		
		print('Ground truth: Recognized')	
		for i in range(len(recognized)):
			numWordOK += 1 if batch.gtTexts[i] == recognized[i] else 0
			numWordTotal += 1
			dist = editdistance.eval(recognized[i], batch.gtTexts[i])
			numCharErr += dist
			numCharTotal += len(batch.gtTexts[i])
			print('[OK]' if dist==0 else '[ERROR:%d]' % dist,'"' + batch.gtTexts[i] + '"', '->', '"' + recognized[i] + '"')
	
	
	CER = numCharErr / numCharTotal
	wordAccuracy = numWordOK / numWordTotal
	print('Character error rate (CER): %f%%. Word accuracy: %f%%.' % (CER*100.0, wordAccuracy*100.0))
	return CER, wordAccuracy


def infer(model, fnImg):

	img = preprocessingImg(cv2.imread(fnImg, cv2.IMREAD_GRAYSCALE), Model.imgSize)
	batch = Batch(None, [img])
	recognized = model.inferBatch(batch)
	print('Recognized:', recognized[0])
	print('Recognized with correction:', correct_sentence(recognized[0]))
	
	return correct_sentence(recognized[0])


def main():
	
	parser = argparse.ArgumentParser()
	parser.add_argument('--train', help='train the NN', action='store_true')
	parser.add_argument('--validate', help='validate the NN', action='store_true')
	

	args = parser.parse_args()

	decoderType = DecoderType.BestPath
		
	if args.train or args.validate:
		
		loader = Loader(FilePaths.fnTrain, Model.batchSize, Model.imgSize, Model.maxTextLen)
		open(FilePaths.fnCharList, 'w').write(str().join(loader.charList))
		
		if args.train:
			model = Model(loader.charList, decoderType)
			train(model, loader)
		elif args.validate:
			model = Model(loader.charList, decoderType, mustRestore=False)
			validate(model, loader)

	else:
		print(open(FilePaths.fnAccuracy).read())
		model = Model(open(FilePaths.fnCharList).read(), decoderType, mustRestore=False)
		infer(model, FilePaths.fnInfer)

def infer_by_webapp(path, option):
	decoderType = DecoderType.BestPath
	print(open(FilePaths.fnAccuracy).read())
	model = Model(open(FilePaths.fnCharList).read(), decoderType, mustRestore=False)
	recognized = infer(model, path)

	return recognized


if __name__ == '__main__':
	main()

