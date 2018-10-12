import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from theano import tensor as T
import numpy as np
from random import shuffle
from scipy import misc
import csv
import h5py

import neuralnet as nn

mean_bgr = np.array([103.939, 116.779, 123.68], dtype='float32')[None, None, :]
input_path_train = 'D:/1_Share/MS_COCO_train'
input_path_val = 'D:/1_Share/MS_COCO_val'
style_path = 'D:/1_Share/wikiart'
style_train_val_path = 'C:/Users/justanhduc/Downloads/ArtGAN-master/WikiArt Dataset/Style'
num_val_imgs = 240
input_size = (3, 256, 256)
bs = 8
weight = 1e-2
lr = 1e-4
n_epochs = 30
val_freq = 500


def post_process(x):
    mean_bgr = np.array([103.939, 116.779, 123.68], dtype='float32')
    return (x / 2. + .5) * 255. - mean_bgr[None, ::-1, None, None]


def unnormalize(x):
    mean_bgr = np.array([103.939, 116.779, 123.68], dtype='float32')
    return (x + mean_bgr[None, ::-1, None, None]) / 255.


def log_std(x, axis):
    return T.log(T.sqrt(T.var(x, axis=axis) + 1e-8))


class VGG19(nn.Sequential):
    def __init__(self, input_shape, name='vgg19'):
        super(VGG19, self).__init__(input_shape=input_shape, layer_name=name)
        self.append(
            nn.Conv2DLayer(self.output_shape, 64, 3, border_mode='ref', no_bias=False, layer_name=name + '/conv1_1'))
        self.append(
            nn.Conv2DLayer(self.output_shape, 64, 3, border_mode='ref', no_bias=False, layer_name=name + '/conv1_2'))
        self.append(nn.MaxPoolingLayer(self.output_shape, (2, 2), layer_name=name + '/maxpool1'))

        self.append(
            nn.Conv2DLayer(self.output_shape, 128, 3, border_mode='ref', no_bias=False, layer_name=name + '/conv2_1'))
        self.append(
            nn.Conv2DLayer(self.output_shape, 128, 3, border_mode='ref', no_bias=False, layer_name=name + '/conv2_2'))
        self.append(nn.MaxPoolingLayer(self.output_shape, (2, 2), layer_name=name + '/maxpool2'))

        self.append(
            nn.Conv2DLayer(self.output_shape, 256, 3, border_mode='ref', no_bias=False, layer_name=name + '/conv3_1'))
        self.append(
            nn.Conv2DLayer(self.output_shape, 256, 3, border_mode='ref', no_bias=False, layer_name=name + '/conv3_2'))
        self.append(
            nn.Conv2DLayer(self.output_shape, 256, 3, border_mode='ref', no_bias=False, layer_name=name + '/conv3_3'))
        self.append(
            nn.Conv2DLayer(self.output_shape, 256, 3, border_mode='ref', no_bias=False, layer_name=name + '/conv3_4'))
        self.append(nn.MaxPoolingLayer(self.output_shape, (2, 2), layer_name=name + '/maxpool3'))

        self.append(
            nn.Conv2DLayer(self.output_shape, 512, 3, border_mode='ref', no_bias=False, layer_name=name + '/conv4_1'))

    def load_params(self, param_file=None):
        f = h5py.File(param_file, mode='r')
        trained = [f['layer_%d' % idx] for idx, _ in enumerate(list(f.keys())) if
                   f['layer_%d' % idx].get('param_0', None) is not None]

        filtered_layers = []
        for layer in self:
            if 'pool' in layer.layer_name:
                continue
            filtered_layers.append(layer)

        weight_value_tuples = []
        for layer, trained_layer in zip(filtered_layers, trained):
            weight_value_tuples.append((layer.W, trained_layer['param_0']))
            weight_value_tuples.append((layer.b, trained_layer['param_1']))
        nn.utils.batch_set_value(weight_value_tuples)
        print('Pretrained weights loaded successfully!')


class Encoder(nn.Sequential):
    def __init__(self, input_shape, name='Encoder'):
        super(Encoder, self).__init__(input_shape=input_shape, layer_name=name)
        self.append(VGG19(self.output_shape, name=name+'/vgg19'))
        self.append(nn.AdaptiveInstanceNorm2DLayer(self.output_shape, layer_name=name+'/adain'))
        self[name+'/vgg19'].load_params('D:/2_Personal/Duc/neuralnet_theano/neuralnet/test_files/vgg19_weights.h5')

    def get_output(self, input):
        out = self[self.layer_name+'/vgg19'](input)
        num_ins = out.shape[0] // 2
        x, y = out[:num_ins], out[num_ins:]
        muy, sigma = T.mean(y, (2, 3)), T.std(y, (2, 3))
        out = self[self.layer_name+'/adain']((x, T.concatenate((sigma, muy), 1)))
        return out

    def vgg19_loss(self, x, y):
        out1_x, out1_y = self[self.layer_name+'/vgg19'][:1](x), self[self.layer_name+'/vgg19'][:1](y)
        out2_x, out2_y = self[self.layer_name+'/vgg19'][1:4](out1_x), self[self.layer_name+'/vgg19'][1:4](out1_y)
        out3_x, out3_y = self[self.layer_name+'/vgg19'][4:7](out2_x), self[self.layer_name+'/vgg19'][4:7](out2_y)
        out4_x, out4_y = self[self.layer_name+'/vgg19'][7:12](out3_x), self[self.layer_name+'/vgg19'][7:12](out3_y)
        return nn.norm_error(T.mean(out1_x, (2, 3)), T.mean(out1_y, (2, 3))) + nn.norm_error(T.mean(out2_x, (2, 3)), T.mean(out2_y, (2, 3))) + \
               nn.norm_error(T.mean(out3_x, (2, 3)), T.mean(out3_y, (2, 3))) + nn.norm_error(T.mean(out4_x, (2, 3)), T.mean(out4_y, (2, 3))) + \
               nn.norm_error(log_std(out1_x, (2, 3)), log_std(out1_y, (2, 3))) + nn.norm_error(log_std(out2_x, (2, 3)), log_std(out2_y, (2, 3))) + \
               nn.norm_error(log_std(out3_x, (2, 3)), log_std(out3_y, (2, 3))) + nn.norm_error(log_std(out4_x, (2, 3)), log_std(out4_y, (2, 3)))


class Decoder(nn.Sequential):
    def __init__(self, input_shape, name='Decoder'):
        super(Decoder, self).__init__(input_shape=input_shape, layer_name=name)
        self.append(
            nn.Conv2DLayer(self.output_shape, 512, 3, border_mode='ref', no_bias=False, layer_name=name + '/conv1_1'))
        self.append(nn.UpsamplingLayer(self.output_shape, 2, method='nearest', layer_name=name+'/up1'))

        self.append(
            nn.Conv2DLayer(self.output_shape, 256, 3, border_mode='ref', no_bias=False, layer_name=name + '/conv2_1'))
        self.append(
            nn.Conv2DLayer(self.output_shape, 256, 3, border_mode='ref', no_bias=False, layer_name=name + '/conv2_2'))
        self.append(
            nn.Conv2DLayer(self.output_shape, 256, 3, border_mode='ref', no_bias=False, layer_name=name + '/conv2_3'))
        self.append(
            nn.Conv2DLayer(self.output_shape, 256, 3, border_mode='ref', no_bias=False, layer_name=name + '/conv2_4'))
        self.append(nn.UpsamplingLayer(self.output_shape, 2, method='nearest', layer_name=name + '/up2'))

        self.append(
            nn.Conv2DLayer(self.output_shape, 128, 3, border_mode='ref', no_bias=False, layer_name=name + '/conv3_1'))
        self.append(
            nn.Conv2DLayer(self.output_shape, 128, 3, border_mode='ref', no_bias=False, layer_name=name + '/conv3_2'))
        self.append(nn.UpsamplingLayer(self.output_shape, 2, method='nearest', layer_name=name + '/up3'))

        self.append(
            nn.Conv2DLayer(self.output_shape, 64, 3, border_mode='ref', no_bias=False, layer_name=name + '/conv4_1'))
        self.append(
            nn.Conv2DLayer(self.output_shape, 3, 3, border_mode='ref', no_bias=False, activation='linear',
                           layer_name=name + '/output'))


def prep_image(im, size=256, color='bgr', resize=512):
    h, w, _ = im.shape
    if h < w:
        new_sh = (resize, int(w * resize / h))
    else:
        new_sh = (int(h * resize / w), resize)
    im = misc.imresize(im, new_sh, interp='bicubic')

    # Crop random
    im = nn.utils.crop_random(im, size)

    im = im.astype('float32')
    if color == 'bgr':
        im = im[:, :, ::-1] - mean_bgr
    elif color == 'rgb':
        im = im - mean_bgr[:, :, ::-1]
    else:
        raise NotImplementedError
    return np.transpose(im[None], (0, 3, 1, 2))


class DataManager(nn.DataManager):
    def __init__(self, placeholders, path, bs, n_epochs, shuffle=False):
        super(DataManager, self).__init__(None, placeholders, path, bs, n_epochs, shuffle=shuffle)
        self.load_data()

    def load_data(self):
        source = os.listdir(self.path[0])
        source = [self.path[0] + '/' + f for f in source]
        if 'val' in self.path[0]:
            source = source[:num_val_imgs]

        file = open(self.path[1], 'r')
        contents = csv.reader(file, delimiter=',')
        style = [style_path + '/' + row[0] for row in contents]

        shuffle(source)
        shuffle(style)

        self.dataset = (source, style)
        self.data_size = min(len(source), len(style))

    def generator(self):
        source, style = list(self.dataset[0]), list(self.dataset[1])

        if self.shuffle:
            shuffle(source)
            shuffle(style)

        if len(source) > len(style):
            source = source[:len(style)]
        else:
            style = style[:len(source)]

        for idx in range(0, self.data_size, self.batch_size):
            source_batch = source[idx:idx + self.batch_size]
            style_batch = style[idx:idx + self.batch_size]
            imgs, stys = [], []
            exit = False
            for sou_f, sty_f in zip(source_batch, style_batch):
                try:
                    img, sty = misc.imread(sou_f), misc.imread(sty_f)
                except FileNotFoundError:
                    exit = True
                    break

                if len(img.shape) < 3 or len(sty.shape) < 3:
                    exit = True
                    break

                img = prep_image(img, color='rgb')
                sty = prep_image(sty, color='rgb')
                imgs.append(img)
                stys.append(sty)

            if exit:
                continue

            imgs = np.concatenate(imgs)
            stys = np.concatenate(stys)
            yield imgs, stys


def train():
    enc = Encoder((None,) + input_size)
    dec = Decoder(enc.output_shape)

    X = T.tensor4('input')
    Y = T.tensor4('style')
    X_ = nn.placeholder((bs,) + input_size, name='input_plhd')
    Y_ = nn.placeholder((bs,) + input_size, name='style_plhd')

    nn.set_training_on()
    latent = enc(T.concatenate((X, Y)))
    X_styled = dec(latent)
    latent_cycle = enc[0](X_styled)

    content_loss = nn.norm_error(latent, latent_cycle)
    style_loss = enc.vgg19_loss(X_styled, Y)
    loss = content_loss + weight * style_loss
    updates = nn.adam(loss, dec.trainable, lr)
    train = nn.function([], [content_loss, style_loss], updates=updates, givens={X: X_, Y: Y_}, name='train generator')

    nn.set_training_off()
    X_styled = dec(enc(T.concatenate((X, Y))))
    test = nn.function([], X_styled, givens={X: X_, Y: Y_}, name='test generator')

    data_train = DataManager((X_, Y_), (input_path_train, style_train_val_path + '/style_train.csv'), bs, n_epochs, True)
    data_test = DataManager((X_, Y_), (input_path_val, style_train_val_path + '/style_val.csv'), bs, 1)
    mon = nn.Monitor(model_name='AdaIN style transfer', root='D:/2_Personal/Duc', valid_freq=val_freq)
    print('Training...')
    for it in data_train:
        with mon:
            c_loss_, s_loss_ = train()
            if np.isnan(c_loss_ + s_loss_) or np.isinf(c_loss_ + s_loss_):
                raise ValueError('Training failed because loss went nan!')
            mon.plot('content loss', c_loss_)
            mon.plot('style loss', s_loss_)

            if it % val_freq == 0:
                for i in data_test:
                    img_styled = test()
                    mon.imwrite('stylized image %d' % i, img_styled, callback=unnormalize)
                    mon.imwrite('input %d' % i, X_.get_value(), callback=unnormalize)
                    mon.imwrite('style %d' % i, Y_.get_value(), callback=unnormalize)
                mon.dump(enc.params, 'encoder.npz', keep=5)
                mon.dump(dec.params, 'decoder.npz', keep=5)
    mon.flush()
    print('Training finished!')


if __name__ == '__main__':
    train()
