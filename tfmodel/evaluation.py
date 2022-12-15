import tensorflow as tf

def loss_calc(logits, labels):

    class_inc_bg = 2

    labels = labels[...,0]

    class_weights = tf.constant([[10.0/90, 10.0]])

    onehot_labels = tf.one_hot(labels, class_inc_bg)

    weights = tf.reduce_sum(class_weights * onehot_labels, axis=-1)

    unweighted_losses = tf.nn.softmax_cross_entropy_with_logits(labels=onehot_labels, logits=logits)

    weighted_losses = unweighted_losses * weights

    loss = tf.reduce_mean(weighted_losses)

    tf.summary.scalar('loss', loss)
    return loss


def evaluation(logits, labels):
    labels = labels[..., 0] #[…,0]： 代表了取最里边一层的所有第0号元素
    tt=tf.argmax(logits, 3)+2*labels
    lg0=tf.zeros_like(tt)
    lg1=tf.ones_like(tt)
    lg2=tf.ones_like(tt)*2
    lg3=tf.ones_like(tt)*3

    lg0 = tf.cast(lg0, dtype=tf.int64)
    lg1 = tf.cast(lg1, dtype=tf.int64)
    lg2 = tf.cast(lg2, dtype=tf.int64)
    lg3 = tf.cast(lg3, dtype=tf.int64)

    TP0 = tf.reduce_mean(tf.cast(tf.equal(tt, lg3), tf.float32))*256*256
    TN0 = tf.reduce_mean(tf.cast(tf.equal(tt, lg0), tf.float32))*256*256
    FN0 = tf.reduce_mean(tf.cast(tf.equal(tt, lg2), tf.float32))*256*256
    FP0 = tf.reduce_mean(tf.cast(tf.equal(tt, lg1), tf.float32))*256*256
    total=TP0+TN0+FN0+FP0
    print(TP0,TN0,FP0,FN0,total)

    correct_prediction = tf.equal(tf.argmax(logits, 3), labels)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    xinzang_CPA = TP0/(TP0+FP0)
    MPA = (TP0 / (TP0 + FP0) + TN0 / (FN0 + TN0)) / 2
    IoU_zhen = TP0 / (TP0 + FP0 + FN0)
    IoU_fan = TN0 / (TN0 + FN0 +FP0)
    MIoU = (IoU_fan+IoU_zhen)/2

    #c_p是预测准确的，那预测不准确的应该是
    #tf.argmax返回最大值的索引
    #tf.equal判断相等
    #tf.cast转换数据类型
    #tf.reduce_mean计算平均值
    #这里的ac应该就是像素准确率
    tf.summary.scalar('accuracy', accuracy)
    tf.summary.scalar('xinzang_CPA', xinzang_CPA)
    tf.summary.scalar('MPA', MPA)
    tf.summary.scalar('IoU_zhen', IoU_zhen)
    tf.summary.scalar('MIoU', MIoU)
    return accuracy, xinzang_CPA, MPA, IoU_zhen, MIoU