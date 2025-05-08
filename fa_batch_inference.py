import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from os import listdir, path
import numpy as np
import scipy, cv2, os, sys, argparse
import audio as audio_utils
import json, subprocess, random, string
from tqdm import tqdm
from glob import glob
import torch, face_alignment
from models import Wav2Lip

max_len = 21
max_res = 480
start_file, finishfile = 'access_control/start.txt', 'access_control/finished.txt'
## define some exceptions
class AudioTooLong(Exception):
	pass

class VideoTooLong(Exception):
	pass

class ImageTooBig(Exception):
	pass

class FaceNotDetected(Exception):
	pass

def get_smoothened_boxes(boxes, T):
	for i in range(len(boxes)):
		if i + T > len(boxes):
			window = boxes[len(boxes) - T:]
		else:
			window = boxes[i : i + T]
		boxes[i] = np.mean(window, axis=0)
	return boxes

def face_detect(detector, images, batch_size, pads):
	while 1:
		predictions = []
		try:
			for i in tqdm(range(0, len(images), batch_size)):
				predictions.extend(detector.get_detections_for_batch(np.array(images[i:i + batch_size])))
		except RuntimeError:
			if batch_size == 1: raise ImageTooBig('Image too big to run face detection on GPU')
			batch_size //= 2
			# print('Recovering from OOM error; New batch size: {}'.format(batch_size))
			continue
		break

	results = []
	pady1, pady2, padx1, padx2 = pads
	for (_, rect), image in zip(predictions, images):
		if rect is None:
			raise FaceNotDetected('Face not detected!')

		y1 = max(0, rect[1] - pady1)
		y2 = min(image.shape[0], rect[3] + pady2)
		x1 = max(0, rect[0] - padx1)
		x2 = min(image.shape[1], rect[2] + padx2)
		
		results.append([x1, y1, x2, y2])

	boxes = get_smoothened_boxes(np.array(results), T=5)
	results = [[image[y1: y2, x1:x2], (y1, y2, x1, x2), True] for image, (x1, y1, x2, y2) in zip(images, boxes)]

	return results 

def datagen(detector, frames, mels, batch_size, face_det_batch_size, static, pads, img_size):
	img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

	if not static:
		face_det_results = face_detect(detector, frames, face_det_batch_size, pads) # BGR2RGB for CNN face detection
	else:
		face_det_results = face_detect(detector, [frames[0]], face_det_batch_size, pads)

	for i, m in enumerate(mels):
		idx = 0 if static else i%len(frames)
		frame_to_save = frames[idx].copy()
		face, coords, valid_frame = face_det_results[idx].copy()
		if not valid_frame:
			# print ("Skipping {}".format(i))
			continue

		face = cv2.resize(face, (img_size, img_size))
			
		img_batch.append(face)
		mel_batch.append(m)
		frame_batch.append(frame_to_save)
		coords_batch.append(coords)

		if len(img_batch) >= batch_size:
			img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

			img_masked = img_batch.copy()
			img_masked[:, img_size//2:] = 0

			img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
			mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

			yield img_batch, mel_batch, frame_batch, coords_batch
			img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

	if len(img_batch) > 0:
		img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

		img_masked = img_batch.copy()
		img_masked[:, img_size//2:] = 0

		img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
		mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

		yield img_batch, mel_batch, frame_batch, coords_batch

def generate(detector, checkpoint_path, face, audio, static, fps, pads, face_det_batch_size, 
			wav2lip_batch_size, resize_factor, device, basename, results_dir):
	
	img_size = 96
	mel_step_size = 16

	if os.path.isfile(face) and face.split('.')[1] in ['jpg', 'png', 'jpeg']:
		static = True
		face.split('.')[1] in ['jpg', 'png', 'jpeg']
		full_frames = [cv2.imread(face)]
		fps = 25

	else:
		print ("THE INPUT NEEDS TO BE AN IMAGE")


	# yield 'Video Reading Complete...'

	mel_idx_multiplier = 80./fps

	h, w = full_frames[-1].shape[:-1]
	min_side = min(h, w)
	if min_side > max_res:
		scale_factor =  min_side / float(max_res)
		h = int(h/scale_factor)
		w = int(w/scale_factor)
		full_frames = [cv2.resize(f, (w, h)) for f in full_frames]

	# print ("Number of frames available for inference: "+str(len(full_frames)))

	if not audio.endswith('.wav'):
		# print('Extracting raw audio...')
		command = 'ffmpeg -y -i {} -strict -2 {}'.format(audio, 'temp/{}.wav'.format(basename))

		subprocess.call(command, shell=True)
		audio = 'temp/{}.wav'.format(basename)

	wav = audio_utils.load_wav(audio, 16000)
	if len(wav) > max_len * 16000:
		wav = wav[:int(max_len * 16000)]
		# raise AudioTooLong('Audio length {} greater than {}'.format(len(wav) / 16000., max_len))

	mel = audio_utils.melspectrogram(wav)
	# print(mel.shape)

	if np.isnan(mel.reshape(-1)).sum() > 0:
		raise ValueError('Mel contains nan!')

	mel_chunks = []
	i = 0
	while 1:
		start_idx = int(i * mel_idx_multiplier)
		if start_idx + mel_step_size > len(mel[0]):
			break
		mel_chunks.append(mel[:, start_idx : start_idx + mel_step_size])
		i += 1

	# print("Length of mel chunks: {}".format(len(mel_chunks)))

	full_frames = full_frames[:len(mel_chunks)]

	# yield 'If all goes well, you should receive a video result of {} seconds'.format(len(full_frames) / float(fps))

	batch_size = wav2lip_batch_size
	gen = datagen(detector, full_frames.copy(), mel_chunks, batch_size, face_det_batch_size, static, pads, img_size)

	for i, (img_batch, mel_batch, frames, coords) in enumerate(tqdm(gen, 
											total=int(np.ceil(float(len(mel_chunks))/batch_size)))):
		if i == 0:
			del detector
			wav2lip = load_model(checkpoint_path, device)
			frame_h, frame_w = full_frames[0].shape[:-1]
			out = cv2.VideoWriter(path.join('temp/{}.avi'.format(basename)), 
									cv2.VideoWriter_fourcc(*'DIVX'), fps, (frame_w, frame_h))

		img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(device)
		mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(device)

		with torch.no_grad():
			pred = wav2lip(mel_batch, img_batch)

		pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.
		
		for p, f, c in zip(pred, frames, coords):
			y1, y2, x1, x2 = c
			p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))

			f[y1:y2, x1:x2] = p
			out.write(f)

	out.release()

	command = 'ffmpeg -y -i {} -i {} -strict -2 -q:v 1 {}'.format(audio, 'temp/{}.avi'.format(basename), 
				'{}.mp4'.format(os.path.join(results_dir, basename)))
	subprocess.call(command, shell=True)

	generated_video_path = 'static/{}.mp4'.format(basename)
	with open(finishfile, 'w') as finish:
		finish.write(generated_video_path)

### defining the Wav2Lip model
### Better to give it a separate second GPU (you can also give cuda:0, but things might become much slower)
def load_model(path, device):
	model = Wav2Lip()

	# print("Load checkpoint from: {}".format(path))
	checkpoint = checkpoint = torch.load(path)
	s = checkpoint["state_dict"]
	new_s = {}
	for k, v in s.items():
		new_s[k.replace('module.', '')] = v
	model.load_state_dict(new_s)

	model = model.to(device)
	return model.eval()

if __name__ == '__main__':
	try:
		start_access = open(start_file, 'w')

		parser = argparse.ArgumentParser(description='Code to generate talking face using Temporal LipGAN')

		parser.add_argument('--checkpoint_path', type=str, 
							help='logs/', required=True)

		parser.add_argument('--face', type=str, 
							help='Filepath of video/image that contains faces to use', required=True)
		parser.add_argument('--audio', type=str, 
							help='Filepath of video/audio file to use as raw audio source', required=True)
		parser.add_argument('--results_dir', type=str, help='Folder to save all results into', default='static/')

		parser.add_argument('--static', type=bool, 
							help='If True, then use only first video frame for inference', default=False)
		parser.add_argument('--fps', type=float, help='If not specified, will be 25 if input is image, else the input video FPS', 
							default=25., required=False)

		parser.add_argument('--pads', nargs='+', type=int, default=[0, 10, 0, 0], 
							help='Padding (top, bottom, left, right)')

		parser.add_argument('--face_det_batch_size', type=int, 
							help='Single GPU batch size for face detection', default=16)
		parser.add_argument('--wav2lip_batch_size', type=int, help='Single GPU batch size for LipGAN', default=128)

		parser.add_argument('--resize_factor', default=1, type=int)
		parser.add_argument('--basename', default="output", type=str)

		parser.add_argument('--device', default='cuda', help='which device to run Wav2Lip on?', type=str)

		args = parser.parse_args()

		### defining the face detector. Set it to the first GPU
		detector = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, 
												flip_input=False, device='cuda:0')

		# wav2lip = load_model(args.checkpoint_path, args.device)

		# define a random unique name for result
		#basename = ''.join([random.choice(string.ascii_letters) for _ in range(5)])

		generated_video_path = generate(detector, args.checkpoint_path, args.face, args.audio, args.static, 
										args.fps, args.pads, args.face_det_batch_size, 
										args.wav2lip_batch_size, args.resize_factor, args.device, args.basename, args.results_dir)

	except Exception as e:
		exception = repr(e)

		with open(finishfile, 'w') as finish:
			finish.write(exception)
		
		# print(exception)
	finally:
		os.remove(start_file)
