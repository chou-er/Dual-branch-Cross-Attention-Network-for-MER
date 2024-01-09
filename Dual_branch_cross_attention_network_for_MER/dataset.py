
from numpy import *

import torch.utils.data as data

import pandas as pd
import numpy as np

from PIL import Image
import cv2


class Dataset(data.Dataset):

	def __init__(self, root, img_paths, img_labels, transform=None, get_aux=True, aux=None):
		"""Load image paths and labels from gt_file"""
		self.root = root
		self.transform = transform
		self.get_aux = get_aux
		self.img_paths = img_paths
		self.img_labels = img_labels
		self.aux = aux

	def __getitem__(self, idx):
		"""Load image.

		Args:
			idx (int): image idx.

		Returns:
			img (tensor): image tensor

		"""
		img_path1 = self.img_paths[idx]

		img = Image.open(os.path.join(self.root, img_path1)).convert('RGB')
		label = self.img_labels[idx]

		if self.transform:
			img = self.transform(img)

		if self.get_aux:
			return img, label


	def __len__(self):
		return len(self.img_paths)


class Dataset_2flow(data.Dataset):

	def __init__(self, root, images_paths, img_labels, transform=None, get_aux=True, aux=None):
		"""Load image paths and labels from gt_file"""
		self.root = root
		self.transform = transform
		self.get_aux = get_aux
		self.images_paths = images_paths
		self.img_labels = img_labels
		self.aux = aux

	def __getitem__(self, idx):

		img_path1 = self.images_paths[idx][1]     	#1是读取光流图片
		img_path2 = self.images_paths[idx][0]		#0是读取RGB流
		label = self.img_labels[idx]


		img1 = cv2.imread(os.path.join(self.root, img_path1))
		img2 = cv2.imread(os.path.join(self.root, img_path2))

		img1 = cv2.resize(img1, (224, 224))
		img2 = cv2.resize(img2, (224, 224))
		img1 = (img1/255).transpose(2, 0, 1).astype(np.float32)
		img2 = (img2/255).transpose(2, 0, 1).astype(np.float32)


		if self.get_aux:

			return img1, img2, label


	def __len__(self):
		return len(self.images_paths)

class Dataset_flow(data.Dataset):

	def __init__(self, root, img_paths, img_labels, transform=None, get_aux=True, aux=None):
		"""Load image paths and labels from gt_file"""
		self.root = root
		self.transform = transform
		self.get_aux = get_aux
		self.img_paths = img_paths
		self.img_labels = img_labels
		self.aux = aux

	def __getitem__(self, idx):
		"""数据集本质应当是所有数据样本的一个列表，因此每个样本都有对应的索引index。我们取用一个样本最简单的方式就是用该样本的index从数据列表中把它取出来。__getitem__就是做这样一件事。"""

		"""Load image.

		Args:
			idx (int): image idx.

		Returns:
			img (tensor): image tensor

		"""
		img_path1 = self.img_paths[idx][1]			#顶点帧路径
		img_path2 = self.img_paths[idx][0]			#起始帧路径
		# print(img_path2)

		img1 = cv2.imread(os.path.join(self.root, img_path1), 0)			#读取顶点帧图片，且变成了灰度
		img2 = cv2.imread(os.path.join(self.root, img_path2), 0)			#读取起始帧图片，且变成了灰度
		img3 = cv2.imread(os.path.join(self.root, img_path2), 1)			#读取起始帧图片
		hsv = np.zeros_like(img3)											#构造一个与img3同维度的数组，并初始化所有变量为0

		hsv[..., 1] = 255

		flow = cv2.calcOpticalFlowFarneback(img1, img2, None, 0.5, 3, 15, 3, 5, 1.2, 0)

		mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])			#直角坐标转换为极坐标
		hsv[...,0] = ang*180/np.pi/2
		hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
		PIL_image = cv2.cvtColor(hsv,cv2.COLOR_HSV2RGB)				#所有的光流图片

		img = Image.fromarray(PIL_image)			#把矩阵转化为照片



		label = self.img_labels[idx]

		if self.transform:
			img = self.transform(img)

		if self.get_aux:
			return img, label



	def __len__(self):
		return len(self.img_paths)

def get_2flow_paths_and_labels(file_path):
	df = pd.read_csv(file_path)
	path1 = list(df.RGB_apex_path)
	path2 = list(df.opt_flow_path)
	paths = list(zip(path1, path2))
	labels = list(df.label)

	total = [paths] + [labels]

	return total

def get_triple_meta_data(file_path):
	df = pd.read_csv(file_path)

	paths = list(df.apex_frame_path)
	labels = list(df.label)
	return paths, labels

def get_triple_meta_data_flow(file_path):
	df = pd.read_csv(file_path)

	on_paths = list(df.onset_frame_path)
	apex_paths = list(df.apex_frame_path)
	off_paths = list(df.offset_frame_path)

	paths = [(on, apex, off) for (on, apex, off) in zip(on_paths, apex_paths, off_paths)]
	labels = list(df.label)
	return paths, labels

def get_meta_data(file_path):
	df = pd.read_csv(file_path)
	paths = list(df.all_photos)

	labels = list(df.label)

	return paths, labels

def flow_get_meta_data(file_path):
	df = pd.read_csv(file_path)
	paths = list(df.optical_flow)

	labels = list(df.label)

	return paths, labels


def data_split(file_path, subject_out_idx=0):
	"""Split dataset into train set and validation set
	"""
	# data, subject, clipID, label, apex_frame, apex_frame_path
	data_sub_column = 'data_sub'
	
 
	df = pd.read_csv(file_path)
	#print(type(df))
	subject_list = list(df[data_sub_column].unique())
	#print(subject_list)
	subject_out = subject_list[subject_out_idx]
	print('subject_out', subject_out)
	df_train = df[df[data_sub_column] != subject_out]
	df_val = df[df[data_sub_column] == subject_out]

	return df_train, df_val


def upsample_subdata(df, df_four, number=4):
    result = df.copy()
    for i in range(df.shape[0]):
        quotient = number // 1
        remainder = number % 1
        remainder = 1 if np.random.rand() < remainder else 0
        value = quotient + remainder

        tmp = df_four[df_four['data_sub'] == df.iloc[i]['data_sub']]
        tmp = tmp[tmp['clip'] == df.iloc[i]['clip']]
        value = min(value, tmp.shape[0])
        tmp = tmp.sample(int(value))
        result = pd.concat([result, tmp])
    return result


def sample_data(df, df_four):
	df_neg = df[df.label == 0]
	df_pos = df[df.label == 1]
	df_sur = df[df.label == 2]
	print('df_negr', df_neg.shape)
	print('df_posr', df_pos.shape)
	print('df_surr', df_sur.shape)

	num_sur = 4
	num_pos = 5 * df_sur.shape[0] / df_pos.shape[0] - 1
	if num_pos < 1:
		num_pos = 1

	#num_neg = 4
	num_neg = 5 * df_sur.shape[0] / df_neg.shape[0] - 1
	if num_neg < 1:
		num_neg = 0
	#print(num_neg)
	df_neg = upsample_subdata(df_neg, df_four, num_neg)
	df_pos = upsample_subdata(df_pos, df_four, num_pos)
	df_sur = upsample_subdata(df_sur, df_four, num_sur)
	print('df_neg', df_neg.shape)
	print('df_pos', df_pos.shape)
	print('df_sur', df_sur.shape)

	df = pd.concat([df_neg, df_pos, df_sur])
	return df

def get_cls(file_path):
	df = pd.read_csv(file_path)
	labels = list(df.label)
	return labels

