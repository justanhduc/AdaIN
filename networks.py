import numpy as np
from theano import tensor as T
import neuralnet as nn
import h5py


def unnormalize(x):
    x = np.clip(x, 0., 1.)
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


def norm_error(x, y):
    return T.sum((x - y) ** 2) / T.cast(x.shape[0], 'float32')


class Encoder(nn.Sequential):
    def __init__(self, input_shape, name='Encoder'):
        super(Encoder, self).__init__(input_shape=input_shape, layer_name=name)
        self.append(VGG19(self.output_shape, name=name+'/vgg19'))
        self.append(nn.AdaIN2DLayer(self.output_shape, layer_name=name+'/adain'))
        self[name+'/vgg19'].load_params('vgg19_weights_normalized.h5')

    def get_output(self, input):
        if isinstance(input, (tuple, list)):
            x, y = input
            x, y = self[self.layer_name+'/vgg19'](x), self[self.layer_name+'/vgg19'](y)
        else:
            out = self[self.layer_name+'/vgg19'](input)
            num_ins = out.shape[0] // 2
            x, y = out[:num_ins], out[num_ins:]
        muy, sigma = T.mean(y, (2, 3)), std(y, (2, 3))
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
        loss = norm_error(T.mean(out1_x, (2, 3)), T.mean(out1_y, (2, 3))) + \
               norm_error(T.mean(out2_x, (2, 3)), T.mean(out2_y, (2, 3))) + \
               norm_error(T.mean(out3_x, (2, 3)), T.mean(out3_y, (2, 3))) + \
               norm_error(T.mean(out4_x, (2, 3)), T.mean(out4_y, (2, 3))) + \
               norm_error(std(out1_x, (2, 3)), std(out1_y, (2, 3))) + \
               norm_error(std(out2_x, (2, 3)), std(out2_y, (2, 3))) + \
               norm_error(std(out3_x, (2, 3)), std(out3_y, (2, 3))) + \
               norm_error(std(out4_x, (2, 3)), std(out4_y, (2, 3)))
        return loss


class Decoder(nn.Sequential):
    def __init__(self, input_shape, name='Decoder'):
        super(Decoder, self).__init__(input_shape=input_shape, layer_name=name)
        self.append(
            nn.Conv2DLayer(self.output_shape, 512, 3, init=nn.GlorotUniform(), border_mode='ref', no_bias=False, layer_name=name + '/conv1_1'))
        self.append(nn.UpsamplingLayer(self.output_shape, 2, method='nearest', layer_name=name+'/up1'))

        self.append(
            nn.Conv2DLayer(self.output_shape, 256, 3, init=nn.GlorotUniform(), border_mode='ref', no_bias=False, layer_name=name + '/conv2_1'))
        self.append(
            nn.Conv2DLayer(self.output_shape, 256, 3, init=nn.GlorotUniform(), border_mode='ref', no_bias=False, layer_name=name + '/conv2_2'))
        self.append(
            nn.Conv2DLayer(self.output_shape, 256, 3, init=nn.GlorotUniform(), border_mode='ref', no_bias=False, layer_name=name + '/conv2_3'))
        self.append(
            nn.Conv2DLayer(self.output_shape, 256, 3, init=nn.GlorotUniform(), border_mode='ref', no_bias=False, layer_name=name + '/conv2_4'))
        self.append(nn.UpsamplingLayer(self.output_shape, 2, method='nearest', layer_name=name + '/up2'))

        self.append(
            nn.Conv2DLayer(self.output_shape, 128, 3, init=nn.GlorotUniform(), border_mode='ref', no_bias=False, layer_name=name + '/conv3_1'))
        self.append(
            nn.Conv2DLayer(self.output_shape, 128, 3, init=nn.GlorotUniform(), border_mode='ref', no_bias=False, layer_name=name + '/conv3_2'))
        self.append(nn.UpsamplingLayer(self.output_shape, 2, method='nearest', layer_name=name + '/up3'))

        self.append(
            nn.Conv2DLayer(self.output_shape, 64, 3, init=nn.GlorotUniform(), border_mode='ref', no_bias=False, layer_name=name + '/conv4_1'))
        self.append(
            nn.Conv2DLayer(self.output_shape, 3, 3, init=nn.GlorotUniform(), border_mode='ref', no_bias=False, activation='tanh',
                           layer_name=name + '/output'))

    def get_output(self, input):
        out = super(Decoder, self).get_output(input)
        out = out / 2. + .5
        return out
