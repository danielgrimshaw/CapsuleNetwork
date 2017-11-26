import os
import tensorflow as tf
from tqdm import tqdm

from config import cfg
from utils import CifarDataManager
from capsuleNet import CapsNet

def main(_):
    capsNet = CapsNet(is_training=cfg.is_training)
    print("created capsNet")
    tf.logging.info('Loaded graph')
    sv = tf.train.Supervisor(graph=capsNet.graph,
            logdir=cfg.logdir,
            save_model_secs=0)
    print("created supervisor")
    path = cfg.results + '/accuracy.csv'
    if not os.path.exists(cfg.results):
        os.mkdir(cfg.results)
    elif os.path.exists(path):
        os.remove(path)
    print("made results log")

    fd_results = open(path, 'w')
    fd_results.write('step,test_acc\n')
    print("wrote column titles")
    #config = tf.ConfigProto()
    #config.gpu_options.allow_growth = True
    print("starting session")
    with sv.managed_session() as sess:
        num_batch = int(50000 / cfg.batch_size)
        num_test_batch = 10000 // cfg.batch_size
        print("calculated batches")
        cifar = CifarDataManager()
        teX, teY = cifar.test.images, cifar.test.labels
        print("starting epochs")
        for epoch in range(cfg.epochs):
            if sv.should_stop():
                break
            for step in tqdm(range(num_batch), total=num_batch, ncols=70, leave=False, unit='b'):
                global_step = sess.run(capsNet.global_step)
                sess.run(capsNet.train_op)

                if step % cfg.train_sum_freq == 0:
                    _, summary_str = sess.run([capsNet.train_op, capsNet.train_summary])
                    sv.summary_writer.add_summary(summary_str, global_step)

                if (global_step + 1) % cfg.test_sum_freq == 0:
                    test_acc = 0
                    for i in range(num_test_batch):
                        start = i * cfg.batch_size
                        end = start + cfg.batch_size
                        test_acc += sess.run(capsNet.batch_accuracy, {capsNet.X: teX[start:end], capsNet.labels: teY[start:end]})
                    test_acc = test_acc / (cfg.batch_size * num_test_batch)
                    fd_results.write(str(global_step + 1) + ',' + str(test_acc) + '\n')
                    fd_results.flush()
                    print(str(test_acc))
            if epoch % cfg.save_freq == 0:
                sv.saver.save(sess, cfg.logdir + '/model_epoch_%04d_step_%02d' % (epoch, global_step))

    fd_results.close()
    tf.logging.info("Training complete.")

if __name__ == "__main__":
    tf.app.run()

