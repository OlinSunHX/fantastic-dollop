from pathlib import Path 
import pickle
import numpy as np 
import torch
import math
from torch.utils.data import Dataset
from skimage import io

class AffectNet(Dataset):
    _expressions = {0: 'neutral', 1:'happy', 2:'sad', 3:'surprise', 4:'fear', 5:'disgust', 6:'anger', 7:'contempt', 8:'none'}
    _expressions_indices = {8: [0, 1, 2, 3, 4, 5, 6, 7], 
                            5: [0, 1, 2, 3, 6]}
    
    def __init__(self, root_path, subset='test',
                 transform_image_shape=None, transform_image=None,
                 n_expression=5, verbose=1, cleaned_set=True):
        self.root_path = Path(root_path).expanduser()
        self.subset = subset
        self.image_path = self.root_path.joinpath(subset)
        self.transform_image_shape = transform_image_shape
        self.transform_image = transform_image
        self.verbose = verbose

        #if cleaned_set and (subset not in ['test', 'val']):
        #    raise ValueError('cleaned_set can only be set to True for the val or test set, train has not been cleaned')
        self.cleaned_set = cleaned_set

        if n_expression not in [5, 8]:
            raise ValueError(f'n_expression should be either 5 or 8, but got n_expression={n_expression}')
        self.n_expression = n_expression

        self.pickle_path = self.root_path.joinpath(f'{subset}_fullpath.pkl')
        # print("pickle_path:",self.pickle_path)
        with open(self.pickle_path, 'br') as f:
            data = pickle.load(f)
        self.data = data
        # print("data：",self.data)
        # the keys are the image names (name.ext)
        self.keys = []
        self.skipped = {'other':[], 'pt_pt_error':[], 'expression':[], 'cleaned':[]}
        # List of each expression to generate weights
        expressions = []
        num=0
        for key, value in data.items():
            # print(key,num)
            if key == 'folder':
                continue
            if (int(value['expression']) not in self._expressions_indices[self.n_expression]):
                self.skipped['expression'].append(key)
                continue
            if self.cleaned_set and (not value['expression_correct']):
                self.skipped['cleaned'].append(key)
                continue

            expression = int(value['expression'])
            if self.cleaned_set:
                #Automatic cleaning : expression has to match the valence and arousal values
                valence = float(value['valence'])
                arousal = float(value['arousal'])
                intensity = math.sqrt(valence**2+arousal**2)

                if expression == 0 and intensity>=0.2:
                    self.skipped['other'].append(key)
                    continue
                elif expression == 1  and (valence<=0 or intensity<=0.2):
                    self.skipped['other'].append(key)
                    continue           
                elif expression == 2  and (valence>=0 or intensity<=0.2):
                    self.skipped['other'].append(key)
                    continue
                elif expression == 3  and (arousal<=0 or intensity<=0.2):
                    self.skipped['other'].append(key)
                    continue
                elif expression == 4  and (not(arousal>=0 and valence<=0) or intensity<=0.2):
                    self.skipped['other'].append(key)
                    continue
                elif expression == 5  and (valence>=0 or intensity<=0.3):
                    self.skipped['other'].append(key)
                    continue
                elif expression == 6  and (arousal<=0 or intensity<=0.2):
                    self.skipped['other'].append(key)
                    continue
                elif expression == 7  and (valence>=0 or intensity<=0.2):
                    self.skipped['other'].append(key)
                    continue
 
                if self.n_expression == 5 and expression == 6:
                    expression = 4
            num+=1
            expressions.append(expression)
            self.keys.append(key)

        expressions = np.array(expressions)
        self.sample_per_class = {label:np.sum(expressions == label) for label in np.unique(expressions)}
        self.expression_weights = np.array([1./self.sample_per_class[e] for e in expressions])
        self.average_per_class = int(np.mean(list(self.sample_per_class.values())))

        if self.verbose:
            skipped = sum([len(self.skipped[key]) for key in self.skipped])
            msg = f' --  {len(self.keys)} images, skipped {len(self.skipped)} images ({len(self.skipped["pt_pt_error"])} with large errors).'
            # print(msg)
            # print(f'Samples per class : {self.sample_per_class}')
        self.keys_anger = ["pickles\jaffe\\fear\\fear_17.jpg",
                      ]
        # self.keys_anger = ["D:\DAQIU\DLGP\SecretJob\\76.emonet-master\emonet-master\pickles\jaffe\\anger\\anger_1.jpg",
        #                    "D:\DAQIU\DLGP\SecretJob\\76.emonet-master\emonet-master\pickles\jaffe\\anger\\anger_6.jpg",
        #                    "D:\DAQIU\DLGP\SecretJob\\76.emonet-master\emonet-master\pickles\jaffe\\anger\\anger_2.jpg",
        #                    "D:\DAQIU\DLGP\SecretJob\\76.emonet-master\emonet-master\pickles\jaffe\\anger\\anger_7.jpg",
        #                    "D:\DAQIU\DLGP\SecretJob\\76.emonet-master\emonet-master\pickles\jaffe\\anger\\anger_3.jpg",
        #                    "D:\DAQIU\DLGP\SecretJob\\76.emonet-master\emonet-master\pickles\jaffe\\anger\\anger_8.jpg",
        #                    "D:\DAQIU\DLGP\SecretJob\\76.emonet-master\emonet-master\pickles\jaffe\\anger\\anger_4.jpg",
        #                    "D:\DAQIU\DLGP\SecretJob\\76.emonet-master\emonet-master\pickles\jaffe\\anger\\anger_9.jpg",
        #                    "D:\DAQIU\DLGP\SecretJob\\76.emonet-master\emonet-master\pickles\jaffe\\anger\\anger_5.jpg",
        #                    "D:\DAQIU\DLGP\SecretJob\\76.emonet-master\emonet-master\pickles\jaffe\\anger\\anger_10.jpg",
        #                    ]
        self.keys_disgust = ["D:\DAQIU\DLGP\SecretJob\\76.emonet-master\emonet-master\pickles\jaffe\\disgust\\disgust_1.jpg",
                           "D:\DAQIU\DLGP\SecretJob\\76.emonet-master\emonet-master\pickles\jaffe\\disgust\\disgust_6.jpg",
                           "D:\DAQIU\DLGP\SecretJob\\76.emonet-master\emonet-master\pickles\jaffe\\disgust\\disgust_2.jpg",
                           "D:\DAQIU\DLGP\SecretJob\\76.emonet-master\emonet-master\pickles\jaffe\\disgust\\disgust_7.jpg",
                           "D:\DAQIU\DLGP\SecretJob\\76.emonet-master\emonet-master\pickles\jaffe\\disgust\\disgust_3.jpg",
                           "D:\DAQIU\DLGP\SecretJob\\76.emonet-master\emonet-master\pickles\jaffe\\disgust\\disgust_8.jpg",
                           "D:\DAQIU\DLGP\SecretJob\\76.emonet-master\emonet-master\pickles\jaffe\\disgust\\disgust_4.jpg",
                           "D:\DAQIU\DLGP\SecretJob\\76.emonet-master\emonet-master\pickles\jaffe\\disgust\\disgust_9.jpg",
                           "D:\DAQIU\DLGP\SecretJob\\76.emonet-master\emonet-master\pickles\jaffe\\disgust\\disgust_5.jpg",
                           "D:\DAQIU\DLGP\SecretJob\\76.emonet-master\emonet-master\pickles\jaffe\\disgust\\disgust_10.jpg",
                           ]
        self.keys_fear = ["D:\DAQIU\DLGP\SecretJob\\76.emonet-master\emonet-master\pickles\jaffe\\fear\\fear_1.jpg",
                           "D:\DAQIU\DLGP\SecretJob\\76.emonet-master\emonet-master\pickles\jaffe\\fear\\fear_6.jpg",
                           "D:\DAQIU\DLGP\SecretJob\\76.emonet-master\emonet-master\pickles\jaffe\\fear\\fear_2.jpg",
                           "D:\DAQIU\DLGP\SecretJob\\76.emonet-master\emonet-master\pickles\jaffe\\fear\\fear_7.jpg",
                           "D:\DAQIU\DLGP\SecretJob\\76.emonet-master\emonet-master\pickles\jaffe\\fear\\fear_3.jpg",
                           "D:\DAQIU\DLGP\SecretJob\\76.emonet-master\emonet-master\pickles\jaffe\\fear\\fear_8.jpg",
                           "D:\DAQIU\DLGP\SecretJob\\76.emonet-master\emonet-master\pickles\jaffe\\fear\\fear_4.jpg",
                           "D:\DAQIU\DLGP\SecretJob\\76.emonet-master\emonet-master\pickles\jaffe\\fear\\fear_9.jpg",
                           "D:\DAQIU\DLGP\SecretJob\\76.emonet-master\emonet-master\pickles\jaffe\\fear\\fear_5.jpg",
                           "D:\DAQIU\DLGP\SecretJob\\76.emonet-master\emonet-master\pickles\jaffe\\fear\\fear_10.jpg",
                           ]
        self.keys_happiness = ["D:\DAQIU\DLGP\SecretJob\\76.emonet-master\emonet-master\pickles\jaffe\\happiness\\happiness_1.jpg",
                           "D:\DAQIU\DLGP\SecretJob\\76.emonet-master\emonet-master\pickles\jaffe\\happiness\\happiness_6.jpg",
                           "D:\DAQIU\DLGP\SecretJob\\76.emonet-master\emonet-master\pickles\jaffe\\happiness\\happiness_2.jpg",
                           "D:\DAQIU\DLGP\SecretJob\\76.emonet-master\emonet-master\pickles\jaffe\\happiness\\happiness_7.jpg",
                           "D:\DAQIU\DLGP\SecretJob\\76.emonet-master\emonet-master\pickles\jaffe\\happiness\\happiness_3.jpg",
                           "D:\DAQIU\DLGP\SecretJob\\76.emonet-master\emonet-master\pickles\jaffe\\happiness\\happiness_8.jpg",
                           "D:\DAQIU\DLGP\SecretJob\\76.emonet-master\emonet-master\pickles\jaffe\\happiness\\happiness_4.jpg",
                           "D:\DAQIU\DLGP\SecretJob\\76.emonet-master\emonet-master\pickles\jaffe\\happiness\\happiness_9.jpg",
                           "D:\DAQIU\DLGP\SecretJob\\76.emonet-master\emonet-master\pickles\jaffe\\happiness\\happinessr_5.jpg",
                           "D:\DAQIU\DLGP\SecretJob\\76.emonet-master\emonet-master\pickles\jaffe\\happiness\\happiness_10.jpg",
                           ]
        self.keys_neutral = ["D:\DAQIU\DLGP\SecretJob\\76.emonet-master\emonet-master\pickles\jaffe\\neutral\\neutral_1.jpg",
                           "D:\DAQIU\DLGP\SecretJob\\76.emonet-master\emonet-master\pickles\jaffe\\neutral\\neutral_6.jpg",
                           "D:\DAQIU\DLGP\SecretJob\\76.emonet-master\emonet-master\pickles\jaffe\\neutral\\neutral_2.jpg",
                           "D:\DAQIU\DLGP\SecretJob\\76.emonet-master\emonet-master\pickles\jaffe\\neutral\\neutral_7.jpg",
                           "D:\DAQIU\DLGP\SecretJob\\76.emonet-master\emonet-master\pickles\jaffe\\neutral\\neutral_3.jpg",
                           "D:\DAQIU\DLGP\SecretJob\\76.emonet-master\emonet-master\pickles\jaffe\\neutral\\neutral_8.jpg",
                           "D:\DAQIU\DLGP\SecretJob\\76.emonet-master\emonet-master\pickles\jaffe\\neutral\\neutral_4.jpg",
                           "D:\DAQIU\DLGP\SecretJob\\76.emonet-master\emonet-master\pickles\jaffe\\neutral\\neutral_9.jpg",
                           "D:\DAQIU\DLGP\SecretJob\\76.emonet-master\emonet-master\pickles\jaffe\\neutral\\neutral_5.jpg",
                           "D:\DAQIU\DLGP\SecretJob\\76.emonet-master\emonet-master\pickles\jaffe\\neutral\\neutral_10.jpg",
                           ]
        self.keys_sadness = ["D:\DAQIU\DLGP\SecretJob\\76.emonet-master\emonet-master\pickles\jaffe\\sadness\\sadness_1.jpg",
                           "D:\DAQIU\DLGP\SecretJob\\76.emonet-master\emonet-master\pickles\jaffe\\sadness\\sadness_6.jpg",
                           "D:\DAQIU\DLGP\SecretJob\\76.emonet-master\emonet-master\pickles\jaffe\\sadness\\sadness_2.jpg",
                           "D:\DAQIU\DLGP\SecretJob\\76.emonet-master\emonet-master\pickles\jaffe\\sadness\\sadness_7.jpg",
                           "D:\DAQIU\DLGP\SecretJob\\76.emonet-master\emonet-master\pickles\jaffe\\sadness\\sadness_3.jpg",
                           "D:\DAQIU\DLGP\SecretJob\\76.emonet-master\emonet-master\pickles\jaffe\\sadness\\sadness_8.jpg",
                           "D:\DAQIU\DLGP\SecretJob\\76.emonet-master\emonet-master\pickles\jaffe\\sadness\\sadness_4.jpg",
                           "D:\DAQIU\DLGP\SecretJob\\76.emonet-master\emonet-master\pickles\jaffe\\sadness\\sadness_9.jpg",
                           "D:\DAQIU\DLGP\SecretJob\\76.emonet-master\emonet-master\pickles\jaffe\\sadness\\sadness_5.jpg",
                           "D:\DAQIU\DLGP\SecretJob\\76.emonet-master\emonet-master\pickles\jaffe\\sadness\\sadness_10.jpg",
                           ]
        self.keys_surprise = [
            "D:\DAQIU\DLGP\SecretJob\\76.emonet-master\emonet-master\pickles\jaffe\\surprise\\surprise_1.jpg",
            "D:\DAQIU\DLGP\SecretJob\\76.emonet-master\emonet-master\pickles\jaffe\\surprise\\surprise_6.jpg",
            "D:\DAQIU\DLGP\SecretJob\\76.emonet-master\emonet-master\pickles\jaffe\\surprise\\surprise_2.jpg",
            "D:\DAQIU\DLGP\SecretJob\\76.emonet-master\emonet-master\pickles\jaffe\\surprise\\surprise_7.jpg",
            "D:\DAQIU\DLGP\SecretJob\\76.emonet-master\emonet-master\pickles\jaffe\\surprise\\surprise_3.jpg",
            "D:\DAQIU\DLGP\SecretJob\\76.emonet-master\emonet-master\pickles\jaffe\\surprise\\surprise_8.jpg",
            "D:\DAQIU\DLGP\SecretJob\\76.emonet-master\emonet-master\pickles\jaffe\\surprise\\surprise_4.jpg",
            "D:\DAQIU\DLGP\SecretJob\\76.emonet-master\emonet-master\pickles\jaffe\\surprise\\surprise_9.jpg",
            "D:\DAQIU\DLGP\SecretJob\\76.emonet-master\emonet-master\pickles\jaffe\\surprise\\surprise_5.jpg",
            "D:\DAQIU\DLGP\SecretJob\\76.emonet-master\emonet-master\pickles\jaffe\\surprise\\surprise_10.jpg",
            ]
        # print("self.keys:",self.keys)
    def __len__(self):
        return len(self.keys_anger)
        # return len(self.keys)

    def __getitem__(self, index):
        key = self.keys[index]
        sample_data = self.data[key]
        key_ = self.keys_anger[index]
        image_file = self.image_path.joinpath(key).as_posix()
        image_file=key_
        print("image_file:",image_file)

        #

        valence = torch.tensor([float(sample_data['valence'])], dtype=torch.float32)
        arousal = torch.tensor([float(sample_data['arousal'])], dtype=torch.float32)
        expression = int(sample_data['expression'])
        if self.n_expression == 5 and expression == 6:
            expression = 4
        #0: 'neutral', 1:'happy', 2:'sad', 3:'surprise', 4:'fear', 5:'disgust', 6:'anger', 7:'contempt', 8:'none'
        # expression = 0   #0.2  0.0
        # expression = 1  # 0.0 0.0
        # expression = 2  # 0.0  0.1
        # expression = 3  # 0.0  0.0
        # expression = 4  # 0.7  0.7
        # expression = 5  # 0.0  0.10
        # expression = 6  # 0.1  0.1
        # expression = 7  # 0.0  0.0
        # expression = 8  # 0.0  0.0
        landmarks = sample_data['landmarks_fan']
        
        if isinstance(landmarks, list):
            landmarks = np.array(landmarks)
        image = io.imread(image_file)

        if self.transform_image_shape is not None:
            bounding_box = [landmarks.min(axis=0)[0], landmarks.min(axis=0)[1],
                            landmarks.max(axis=0)[0], landmarks.max(axis=0)[1]]
            #image, landmarks = self.transform_image_shape(image, shape=landmarks)
            image, landmarks = self.transform_image_shape(image, bb=bounding_box)
            # Fix for PyTorch currently not supporting negative stric
            image = np.ascontiguousarray(image)

        if self.transform_image is not None:
            image = self.transform_image(image)

        return dict(valence=valence, arousal=arousal, expression=expression, image=image, au=[])

