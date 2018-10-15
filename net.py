import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from theano import tensor as T
import numpy as np
from random import shuffle
from scipy import misc
import csv
import h5py

import neuralnet as nn

mean_bgr = np.array([103.939, 116.779, 123.68], dtype='float32')
input_path_train = 'D:/1_Share/MS_COCO_train'
input_path_val = 'D:/1_Share/MS_COCO_val'
style_path = 'D:/1_Share/wikiart'
style_train_val_path = 'C:/Users/justanhduc/Downloads/ArtGAN-master/WikiArt Dataset/Style'
test_input_img_path = 'D:/2_Personal/Duc/neuralnet_theano/neuralnet/test_files/lena_small.png'
test_style_img_path = 'D:/1_Share/arts2photos/vangogh2photo/trainA/00125.jpg'
num_val_imgs = 240
input_size = (3, 256, 256)
bs = 8
weight = 1e-2
lr = 1e-4
lr_decay_rate = 5e-5
n_epochs = 30
print_freq = 100
val_freq = 500


def post_process(x):
    return (x / 2. + .5) * 255. - mean_bgr[None, ::-1, None, None]


def unnormalize(x):
    x = np.minimum(np.maximum(x, 0.), 1.)
    return x


def std(x, axis):
    return T.sqrt(T.var(x, axis=axis) + 1e-8)


def convert_kernel(kernel):
    """Converts a Numpy kernel matrix from Theano format to TensorFlow format.
    Also works reciprocally, since the transformation is its own inverse.
    # Arguments
        kernel: Numpy array (3D, 4D or 5D).
    # Returns
        The converted kernel.
    # Raises
        ValueError: in case of invalid kernel shape or invalid data_format.
    """
    kernel = np.asarray(kernel)
    if not 3 <= kernel.ndim <= 5:
        raise ValueError('Invalid kernel shape:', kernel.shape)
    slices = [slice(None, None, -1) for _ in range(kernel.ndim)]
    no_flip = (slice(None, None), slice(None, None))
    slices[-2:] = no_flip
    return np.copy(kernel[slices])


def prep(x):
    conv = nn.Conv2DLayer((1, 3, 224, 224), 3, 1, no_bias=False, activation='linear', filter_flip=False, border_mode='valid')
    kern = np.array([[0, 0, 255], [0, 255, 0], [255, 0, 0]], 'float32')[:, :, None, None]
    conv.W.set_value(kern)
    conv.b.set_value(np.array([-103.939, -116.779, -123.68], 'float32'))
    return conv(x)


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

    def get_output(self, input):
        out = prep(input)
        return super(VGG19, self).get_output(out)

    def load_params(self, param_file=None):
        f = h5py.File(param_file, mode='r')
        trained = [layer[1].value for layer in list(f.items())]
        weight_value_tuples = []
        for p, tp in zip(self.params, trained):
            if len(tp.shape) == 4:
                tp = np.transpose(convert_kernel(tp), (3, 2, 0, 1))
            weight_value_tuples.append((p, tp))
        nn.utils.batch_set_value(weight_value_tuples)
        print('Pretrained weights loaded successfully!')


class Encoder(nn.Sequential):
    def __init__(self, input_shape, name='Encoder'):
        super(Encoder, self).__init__(input_shape=input_shape, layer_name=name)
        self.append(VGG19(self.output_shape, name=name+'/vgg19'))
        self.append(nn.AdaIN2DLayer(self.output_shape, layer_name=name+'/adain'))
        self[name+'/vgg19'].load_params('C:/Users/justanhduc/Downloads/tf-adain-master\models/vgg19_weights_normalized.h5')

    def get_output(self, input):
        out = self[self.layer_name+'/vgg19'](input)
        num_ins = out.shape[0] // 2
        x, y = out[:num_ins], out[num_ins:]
        muy, sigma = T.mean(y, (2, 3)), T.std(y, (2, 3))
        out = self[self.layer_name+'/adain']((x, T.concatenate((sigma, muy), 1)))
        return out

    def vgg19_loss(self, x, y):
        input = T.concatenate((x, y))
        out1 = self[self.layer_name+'/vgg19'][:1](input)
        out2 = self[self.layer_name+'/vgg19'][1:4](out1)
        out3 = self[self.layer_name+'/vgg19'][4:7](out2)
        out4 = self[self.layer_name+'/vgg19'][7:12](out3)
        idx = x.shape[0]
        out1_x, out1_y = out1[:idx], out1[idx:]
        out2_x, out2_y = out2[:idx], out2[idx:]
        out3_x, out3_y = out3[:idx], out3[idx:]
        out4_x, out4_y = out4[:idx], out4[idx:]
        return nn.norm_error(T.mean(out1_x, (2, 3)), T.mean(out1_y, (2, 3))) + \
               nn.norm_error(T.mean(out2_x, (2, 3)), T.mean(out2_y, (2, 3))) + \
               nn.norm_error(T.mean(out3_x, (2, 3)), T.mean(out3_y, (2, 3))) + \
               nn.norm_error(T.mean(out4_x, (2, 3)), T.mean(out4_y, (2, 3))) + \
               nn.norm_error(std(out1_x, (2, 3)), std(out1_y, (2, 3))) + \
               nn.norm_error(std(out2_x, (2, 3)), std(out2_y, (2, 3))) + \
               nn.norm_error(std(out3_x, (2, 3)), std(out3_y, (2, 3))) + \
               nn.norm_error(std(out4_x, (2, 3)), std(out4_y, (2, 3)))


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


def prep_image(im, size=256, resize=512):
    h, w, _ = im.shape
    if h < w:
        new_sh = (resize, int(w * resize / h))
    else:
        new_sh = (int(h * resize / w), resize)
    im = misc.imresize(im, new_sh, interp='bicubic')

    # Crop random
    im = nn.utils.crop_random(im, size)
    im = im.astype('float32')
    return np.transpose(im[None], (0, 3, 1, 2)) / 255.


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

                imgs.append(prep_image(img))
                stys.append(prep_image(sty))

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
    lr_ = nn.placeholder(value=lr, name='learning rate')
    idx = T.scalar('iteration', 'int64')

    nn.set_training_on()
    latent = enc(T.concatenate((X, Y)))
    X_styled = dec(latent)
    latent_cycle = enc[0](X_styled)

    content_loss = nn.norm_error(latent, latent_cycle)
    style_loss = enc.vgg19_loss(X_styled, Y)
    loss = content_loss + weight * style_loss
    updates = nn.adam(loss * 1e6, dec.trainable, lr_)
    nn.anneal_learning_rate(lr_, idx, 'inverse', decay=lr_decay_rate)
    train = nn.function([idx], [content_loss, style_loss], updates=updates, givens={X: X_, Y: Y_}, name='train generator')

    nn.set_training_off()
    X_styled = dec(enc(T.concatenate((X, Y))))
    test = nn.function([], X_styled, givens={X: X_, Y: Y_}, name='test generator')

    data_train = DataManager((X_, Y_), (input_path_train, style_train_val_path + '/style_train.csv'), bs, n_epochs, True)
    data_test = DataManager((X_, Y_), (input_path_val, style_train_val_path + '/style_val.csv'), bs, 1)
    mon = nn.Monitor(model_name='AdaIN style transfer', root='D:/2_Personal/Duc', valid_freq=print_freq)
    print('Training...')
    for it in data_train:
        with mon:
            c_loss_, s_loss_ = train(it)
            if np.isnan(c_loss_ + s_loss_) or np.isinf(c_loss_ + s_loss_):
                raise ValueError('Training failed because loss went nan!')
            mon.plot('content loss', c_loss_)
            mon.plot('style loss', s_loss_)
            mon.plot('learning rate', lr_.get_value())

            if it % val_freq == 0:
                for i in data_test:
                    img_styled = test()
                    mon.hist('output histogram %d' % i, img_styled)
                    mon.imwrite('stylized image %d' % i, img_styled, callback=unnormalize)
                    mon.imwrite('input %d' % i, X_.get_value(), callback=unnormalize)
                    mon.imwrite('style %d' % i, Y_.get_value(), callback=unnormalize)
                mon.dump(enc.params, 'encoder.npz', keep=5)
                mon.dump(dec.params, 'decoder.npz', keep=5)
    mon.flush()
    print('Training finished!')


def test():
    enc = Encoder((None,) + input_size)
    dec = Decoder(enc.output_shape)
    mon = nn.Monitor(model_name='AdaIN style transfer', current_folder='D:/2_Personal/Duc/AdaIN style transfer/run1')
    trained_dec_params = [p.get_value() for p in mon.load('decoder-56360.npz')]
    nn.utils.batch_set_value(zip(dec.params, trained_dec_params))

    X = T.tensor4('input')
    Y = T.tensor4('style')

    nn.set_training_off()
    X_styled = dec(enc(T.concatenate((X, Y))))
    test = nn.function([X, Y], X_styled, name='test generator')

    input = misc.imread(test_input_img_path)
    style = misc.imread(test_style_img_path)
    input = prep_image(input, input.shape[0], resize=input.shape[0])
    style = prep_image(style, input.shape[-1], resize=input.shape[-1] + 10)

    output = test(input, style)
    # output = (output + mean_bgr[None, ::-1, None, None]) / 255.
    output = (output - np.min(output, keepdims=True)) / (np.max(output, keepdims=True) - np.min(output, keepdims=True))
    mon.imwrite('test input', input, callback=unnormalize)
    mon.imwrite('test style', style, callback=unnormalize)
    mon.imwrite('test output', output, callback=unnormalize)
    mon.flush()


if __name__ == '__main__':
    train()
