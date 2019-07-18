import os
import argparse
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

from morph_net.network_regularizers import activation_regularizer, flop_regularizer, latency_regularizer
from morph_net.tools import structure_exporter

def mnist_model(X_ph, scope):
    bn_params = {'is_training': True, 'scale': True, 'center': False}
    with tf.variable_scope(scope, reuse=False):
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                      normalizer_fn=slim.batch_norm,
                      normalizer_params=bn_params,
                      weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                      weights_regularizer=slim.l2_regularizer(0.0005)):
            conv1 = slim.conv2d(X_ph, num_outputs=64, kernel_size=3, scope='conv_1')
            pool1 = slim.max_pool2d(conv1, kernel_size=2, scope='pool1')
            conv2 = slim.conv2d(pool1, num_outputs=64, kernel_size=3, scope='conv_2')
            pool2 = slim.max_pool2d(conv2, kernel_size=2, scope='pool2')
            conv3 = slim.conv2d(pool2, num_outputs=64, kernel_size=3, scope='conv_3')
            out = slim.conv2d(conv3, num_outputs=10, kernel_size=7, padding='VALID', normalizer_fn=None,
              normalizer_params=None, activation_fn=None, scope='conv_output')
    logits = tf.reduce_mean(out, [1, 2], keepdims=False)
    pred = tf.argmax(logits, axis=1)
    return logits, pred

def main(args):
    # Load MNIST Data
    train_data, test_data = tf.keras.datasets.mnist.load_data()
    X_train, y_train = train_data[0], train_data[1]
    X_test, y_test = test_data[0], test_data[1]
    global_step = tf.train.get_or_create_global_step()

    N, H, W = X_train.shape
    X_ph = tf.placeholder(tf.float32, [None, H, W, 1])
    y_ph = tf.placeholder(tf.int64, [None])

    # Defining Model
    logits, pred = mnist_model(X_ph, scope='base')
    loss_op = tf.losses.sparse_softmax_cross_entropy(labels=y_ph, logits=logits)
    acc_op = tf.reduce_mean(tf.cast(tf.equal(pred, y_ph), tf.float32))

    # Setting Regularizer and Loss Ops
    if args.reg_type == "activation":
        network_regularizer = activation_regularizer.GammaActivationRegularizer(
            output_boundary=[logits.op],
            input_boundary=[X_ph.op, y_ph.op],
            gamma_threshold=args.gamma_threshold)
    elif args.reg_type == "flop":
        network_regularizer = flop_regularizer.GammaFlopsRegularizer(
            output_boundary=[logits.op],
            input_boundary=[X_ph.op, y_ph.op], 
            gamma_threshold=args.gamma_threshold)
    elif args.reg_type == "latency":
        network_regularizer = latency_regularizer.GammaLatencyRegularizer(
            output_boundary=[logits.op],
            input_boundary=[X_ph.op, y_ph.op],
            hardware=args.hardware,
            gamma_threshold=args.gamma_threshold)

    reg_loss_op = network_regularizer.get_regularization_term() * args.reg_penalty  
    cost_op = network_regularizer.get_cost()
    exporter = structure_exporter.StructureExporter(network_regularizer.op_regularizer_manager)

    optimizer = tf.train.AdamOptimizer(learning_rate=args.lr)
    train_op = optimizer.minimize(loss_op + reg_loss_op, global_step=global_step)

    hooks = [
        tf.train.StopAtStepHook(last_step=args.steps + 1),
        tf.train.LoggingTensorHook(tensors={'step': global_step, 'loss': loss_op}, every_n_iter=10)
    ]
    # pin GPU to be used to process local rank (one GPU per process)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    # Training Loop
    with tf.train.MonitoredTrainingSession(checkpoint_dir=args.outdir, hooks=hooks, config=config) as mon_sess:
        while not mon_sess.should_stop():
            idx = np.random.choice(N, args.batch_size, replace=False)
            x_t, y_t = np.expand_dims(X_train[idx], axis=-1), y_train[idx]
            train_dict = {X_ph: x_t, y_ph: y_t}

            val_idx = np.random.choice(X_test.shape[0], 5000, replace=False)
            x_v, y_v = np.expand_dims(X_test[val_idx], axis=-1), y_test[val_idx]
            val_dict = {X_ph: x_v, y_ph: y_v}

            global_step_val = mon_sess.run(global_step, feed_dict=train_dict)
            structure_exporter_tensors, v_loss, v_acc, reg_cost = mon_sess.run(
                [exporter.tensors, loss_op, acc_op, cost_op], feed_dict=val_dict)
            mon_sess.run(train_op, feed_dict=train_dict)

            print("Step: ", global_step_val)
            print("Validation Loss: ", v_loss)
            print("Validation Acc: ", v_acc)
            print("Reg Cost: ", reg_cost)

            # exporting model to JSON
            if global_step_val % 1000 == 0:
                exporter.populate_tensor_values(structure_exporter_tensors)
                exporter.create_file_and_save_alive_counts(args.outdir, global_step_val)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=10000, help='total number of training steps')
    parser.add_argument('--batch_size', type=int, default=64, help='training batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='default learning rate')
    parser.add_argument('--outdir', type=str, default='mnist_debug/', help='where to save checkpoints and structure exports')

    # MorphNet specific parameters
    parser.add_argument('--gamma_threshold', type=float, default=1e-3, help='threshold below which gammas are treated as 0')
    parser.add_argument('--reg_penalty', type=float, default=1e-3, help='regularization coefficient')
    parser.add_argument('--reg_type', type=str, choices=['activation', 'flop', 'latency'], default='activation', help='type of regularizer to use')
    # Can find more harwarde options in morph_net/network_regularizers/resource_function.py
    parser.add_argument('--hardware', type=str, default='1080ti', help='hardware for latency regularizer; unused if flop or activation')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
