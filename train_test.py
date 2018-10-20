import argparse

parser = argparse.ArgumentParser()
parser.add_argument('input_path', type=str, help='path to MS COCO train/test images')
parser.add_argument('style_path', type=str, help='path to the Wikiart dataset')
parser.add_argument('--test_bulk', action='store_true', help='test a trained network on the MS COCO test set')
parser.add_argument('--test_one', action='store_true', help='test a trained network on a single image pair')
parser.add_argument('--style_train_val_path', type=str, default=None, help='path to the directory containing '
                                                                           'style_train.csv and style_val.csv')
parser.add_argument('--input_path_val', type=str, default=None, help='path to MS COCO val images')
parser.add_argument('--resume', action='store_true', default=False,
                    help='whether to continue training from a checkpoint')
parser.add_argument('--checkpoint', type=int, default=0, help='the epoch from which training will continue')
parser.add_argument('--checkpoint_file', type=str, default=None, help='a pretrained weight file')
parser.add_argument('--checkpoint_folder', type=str, default=None, help='the folder containing the weight file')
parser.add_argument('--num_val_imgs', type=int, default=240, help='number of images used for validation during training')
parser.add_argument('--input_size', type=int, default=256, help='the size of input images. input images will '
                                                                'be cropped to this square size')
parser.add_argument('--bs', type=int, default=8, help='batch size')
parser.add_argument('--style_weight', type=float, default=1e2, help='weight for style loss')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--lr_decay', type=float, default=5e-5, help='decay rate for learning rate')
parser.add_argument('--n_epochs', type=int, default=20, help='number of epochs')
parser.add_argument('--print_freq', type=int, default=100, help='frequency to show losses')
parser.add_argument('--valid_freq', type=int, default=500, help='frequency to validate the network in training')
parser.add_argument('--gpu', type=int, default=0, help='GPU number to use')
args = parser.parse_args()

import os
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
from networks import *
from data_processing import *

import neuralnet as nn

if args.test_one or args.test_bulk or args.resume:
    checkpoint = args.checkpoint
    checkpoint_file = args.checkpoint_file
    checkpoint_folder = args.checkpoint_folder

input_path_train = args.input_path
input_path_test = args.input_path
style_path = args.style_path
test_style_img_path = args.style_path

if not args.test_one:
    input_path_val = args.input_path_val
    style_train_val_path = args.style_train_val_path

num_val_imgs = args.num_val_imgs
input_size = (3, args.input_size, args.input_size)
bs = args.bs
bs_test = args.bs
weight = args.style_weight
lr = args.lr
lr_decay_rate = args.lr_decay
n_epochs = args.n_epochs
print_freq = args.print_freq
val_freq = args.valid_freq


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

    data_train = DataManager((X_, Y_), (input_path_train, style_train_val_path + '/style_train.csv', style_path), bs,
                             n_epochs, True, num_val_imgs=num_val_imgs)
    data_test = DataManager((X_, Y_), (input_path_val, style_train_val_path + '/style_val.csv', style_path), bs, 1)
    mon = nn.Monitor(model_name='AdaIN style transfer', valid_freq=print_freq)
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
                mon.dump(nn.utils.shared2numpy(dec.params), 'decoder.npz', keep=5)
    mon.flush()
    mon.dump(nn.utils.shared2numpy(dec.params), 'decoder.npz', keep=5)
    print('Training finished!')


def test_bulk():
    enc = Encoder((None,) + input_size)
    dec = Decoder(enc.output_shape)
    mon = nn.Monitor(model_name='AdaIN style transfer', current_folder=checkpoint_folder)
    nn.utils.numpy2shared(mon.load(checkpoint_file), dec.params)

    X = T.tensor4('input')
    Y = T.tensor4('style')
    X_ = nn.placeholder((bs_test,) + input_size, name='input_plhd')
    Y_ = nn.placeholder((bs_test,) + (3, args.input_size//2, args.input_size//2), name='style_plhd')

    nn.set_training_off()
    X_styled = dec(enc((X, Y)))
    test = nn.function([], X_styled, givens={X: X_, Y: Y_}, name='test generator')

    data_test = DataManager((X_, Y_), (input_path_test, style_train_val_path + '/style_val.csv', style_path), bs_test, 1)
    for i in data_test:
        img_styled = test()
        mon.imwrite('test output %d' % i, img_styled, callback=unnormalize)
        mon.imwrite('test input %d' % i, X_.get_value(), callback=unnormalize)
        mon.imwrite('test style %d' % i, Y_.get_value(), callback=unnormalize)
        mon.flush()
    print('Testing finished!')


def test_one():
    enc = Encoder((None,) + input_size)
    dec = Decoder(enc.output_shape)
    mon = nn.Monitor(model_name='AdaIN style transfer', current_folder=checkpoint_folder)
    nn.utils.numpy2shared(mon.load(checkpoint_file), dec.params)

    X = T.tensor4('input')
    Y = T.tensor4('style')

    nn.set_training_off()
    X_styled = dec(enc((X, Y)))
    test = nn.function([X, Y], X_styled, name='test generator')

    x, y = load_single_sample((input_path_test, style_path), args.input_size, args.input_size+30)
    img_styled = test(x, y)
    mon.imwrite('test output', img_styled, callback=unnormalize)
    mon.imwrite('test input', x, callback=unnormalize)
    mon.imwrite('test style', y, callback=unnormalize)
    mon.flush()
    print('Testing finished!')


def resume():
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

    data_train = DataManager((X_, Y_), (input_path_train, style_train_val_path + '/style_train.csv', style_path), bs,
                             n_epochs, True, checkpoint=checkpoint, num_val_imgs=num_val_imgs)
    data_test = DataManager((X_, Y_), (input_path_val, style_train_val_path + '/style_val.csv'), bs, 1)
    mon = nn.Monitor(model_name='AdaIN style transfer', checkpoint=checkpoint * len(data_train) // bs,
                     current_folder=checkpoint_folder, valid_freq=print_freq)
    nn.utils.numpy2shared(mon.load(checkpoint_file), dec.params)
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
                mon.dump(nn.utils.shared2numpy(dec.params), 'decoder.npz', keep=5)
    mon.flush()
    mon.dump(nn.utils.shared2numpy(dec.params), 'decoder.npz', keep=5)
    print('Training finished!')


if __name__ == '__main__':
    if not args.test_one and not args.test_bulk:
        assert args.input_path_val is not None and args.style_train_val_path is not None, 'Validation paths must be provided'
        if args.resume:
            assert args.checkpoint is not None and args.checkpoint_file is not None and args.checkpoint_folder is not None, 'A pretrained model must be specified'
            resume()
        else:
            train()
    elif args.test_one and not args.test_bulk:
        assert args.checkpoint_file is not None and args.checkpoint_folder is not None, 'A pretrained model must be specified'
        test_one()
    elif args.test_bulk and not args.test_one:
        assert args.checkpoint_file is not None and args.checkpoint_folder is not None, 'A pretrained model must be specified'
        assert args.style_train_val_path is not None, 'Path to the validation style csv file must be provided'
        test_bulk()
    else:
        raise NotImplementedError
