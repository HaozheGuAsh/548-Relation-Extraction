#!/usr/bin/env python3
'''
Module Written By Haozhe Gu(Ash)
For the paper: A Soft-label Method for Noise-tolerant Distantly Supervised Relation Extraction
'''

# -*- coding: utf-8 -*-
from . import relation_extraction_3 as parent
import tensorflow as tf
import os
import time
from relation_extraction_2 import RelationExtractionModel

# Utility Section
# =================== word vector =================== #


def get_lower_sentence(sentence):
    sentence = sentence.strip()
    sentence = sentence.lower()
    return sentence

# load the same pre-trained vector from https://github.com/thunlp/NRE/tree/master/data


def word2vec():
    fvec = open("data/vector1.txt", "r")
    vec = {}
    for line in fvec:
        line = line.strip()
        line = line.split('\t')
        vec[line[0]] = line[1:51]
    fvec.close()
    return vec


def get_word_vec(vectors, one_or_att="att"):
    """
    :param vectors:
    :return: word_vocab: word -> id
             word_vectors: numpy
    """
    sorted_vectors = sorted(vectors.items(), key=lambda d: d[0])
    word_vocab = {}
    word_vectors = []
    for it in sorted_vectors:
        word_vocab[it[0]] = len(word_vocab)
        word_vectors.append(it[1])
    assert len(word_vocab) == len(word_vectors)
    word_vocab['UNK'] = len(word_vocab)
    # word_vocab['BLANK'] = len(word_vocab)
    word_vectors.append(np.random.normal(size=50, loc=0, scale=0.05))
    if one_or_att == "att":
        word_vocab['BLANK'] = len(word_vocab)
        word_vectors.append(np.random.normal(size=50, loc=0, scale=0.05))
    # word_vectors.append(np.random.normal(size=50, loc=0, scale=0.05))
    word_vectors = np.array(word_vectors, dtype=float)
    return word_vocab, word_vectors


def get_sentence_seq(sentence, word_vocab):
    vec = []
    words = sentence.split()
    for word in words:
        try:
            id = word_vocab[word]
        except:
            id = word_vocab["UNK"]
        vec.append(id)
    return vec


# =================== load data =================== #
def get_data(istrain=True, word_vocab=None):
    frel = open("data/RE/relation2id.txt")
    rel2id = {}

    for line in frel:
        line = line.strip()
        it = line.split()
        rel2id[it[0]] = it[1]

    if istrain:
        file = open("data/RE/train.txt")
    else:
        file = open("data/RE/test.txt")

    sen_id = []
    real_sen = []
    lpos = []
    rpos = []
    namepos = []
    bag_id = {}
    midrel = {}
    cnt = 0
    for line in file:
        line = line.strip()
        line = line[:-10].strip()
        it = line.split('\t')
        mid1, mid2, rel, sen = it[0], it[1], it[-2], get_lower_sentence(it[-1])

        if rel not in rel2id:
            continue
        rel = rel2id[rel]

        if len(sen.split()) > 200:
            continue

        name1, name2 = it[2], it[3]
        en1, en2, wps_left, wps_right = get_position(sen, name1, name2)
        if en1 == 0 or en2 == 0:
            cnt += 1
            continue
        name_set = key_position(sen, en1, en2)
        if istrain:
            key = mid1 + '\t' + mid2 + '\t' + rel
        else:
            key = mid1 + '\t' + mid2
            key1 = mid1 + '\t' + mid2 + '\t' + rel
            if rel != "0":
                midrel[key1] = 1

        if key not in bag_id:
            bag_id[key] = []
        bag_id[key].append(len(sen_id))
        lpos.append(wps_left.tolist())
        rpos.append(wps_right.tolist())
        sen_id.append(get_sentence_seq(sen, word_vocab))
        real_sen.append(sen)
        namepos.append(name_set)
    if istrain:
        return bag_id, sen_id, lpos, rpos, real_sen, namepos
    else:
        return bag_id, sen_id, midrel, lpos, rpos, real_sen, namepos


def key_position(sen, en1, en2):
    sen_len = len(sen.split())
    en = [en1, en2]
    ean = []
    for eni in en:
        eans = 0
        if eni == 1:
            eans = 0
        elif eni == sen_len:
            eans = eni - 3
        else:
            eans = eni - 2
        ean.append(eans)
    return ean


def get_position(sen, name1, name2):
    sentence = sen.split()
    llen = len(sentence)
    wps_left = np.array(range(1, llen + 1))
    wps_right = np.array(range(1, llen + 1))
    en1, en2 = 0, 0
    for i, it in enumerate(sentence):
        if it == name1:
            en1 = i + 1
        if it == name2:
            en2 = i + 1
    wps_left -= int(en1)
    wps_right -= int(en2)
    wps_left = np.minimum(np.maximum(wps_left, -30), 30) + 30
    wps_right = np.minimum(np.maximum(wps_right, -30), 30) + 30
    return en1, en2, wps_left, wps_right


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


# =================== evaluation =================== #
def evaluate(path, triple_list, midrel, epoch):
    frel = open("data/RE/relation2id.txt")
    fmid = open("data/RE/entity2name.txt")

    rel2id = {}
    id2rel = {}
    mid2name = {}
    triple = []
    for line in frel:
        line = line.strip()
        it = line.split()
        rel2id[it[0]] = it[1]
        id2rel[int(it[1])] = it[0]
    for line in fmid:
        line = line.strip()
        it = line.split('\t')
        mid2name[it[0]] = it[1]

    for item in triple_list:
        mid = item["mid"].split('\t')
        rel = item["rel"]
        score = item["score"]
        ent1, ent2 = mid2name[mid[0]], mid2name[mid[1]]
        rname = id2rel[rel]
        key = ent1 + '\t' + ent2 + '\t' + rname
        mid_key = mid[0] + '\t' + mid[1] + '\t' + str(rel)
        crt = "0"
        if mid_key in midrel:
            crt = "1"
        triple.append({"triple": key, "val": score, "crt": crt})
    sorted_triple = sorted(triple, key=lambda x: x["val"])
    prfile = open(path, "w")
    correct = 0
    tot_recall = len(midrel.keys())
    for i, item in enumerate(sorted_triple[::-1]):
        if str(item["crt"]) == "1":
            correct += 1
        prfile.write("{0:.5f}\t{1:.5f}\t{2:.5f}\t".format(
            float(correct) / (i + 1), float(correct) / tot_recall, float(item["val"])))
        prfile.write(str(item["triple"]) + '\n')
        if i + 1 > 2000:
            break
    prfile.close()


# Progress bar

TOTAL_BAR_LENGTH = 100.
last_time = time.time()
begin_time = last_time
# print os.popen('stty size', 'r').read()
_, term_width = os.popen('stty size', 'r').read().split()
term_width = int(term_width)


def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH * current / total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width - int(TOTAL_BAR_LENGTH) - len(msg) - 3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width - int(TOTAL_BAR_LENGTH / 2) + 2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current + 1, total))

    if current < total - 1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()


def format_time(seconds):
    days = int(seconds / 3600 / 24)
    seconds = seconds - days * 3600 * 24
    hours = int(seconds / 3600)
    seconds = seconds - hours * 3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes * 60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds * 1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f

# Model Section

# Selective Attention Model


class attention_model(object):
    def __init__(self, pad_len, num_rels, word_vectors, window_size, num_filters, embedding_size, pos_embedding, dropout, batch_num, joint_p, l2_reg=0.0):
        self.input = tf.placeholder(tf.int32, [None, pad_len], name="input")
        self.preds = tf.placeholder(tf.int32, [None, num_rels], name="preds")
        self.num_filters = num_filters
        self.mask1 = tf.placeholder(tf.float32, [
                                    None, pad_len - window_size + 1, self.num_filters], name="mask_before")
        self.mask2 = tf.placeholder(tf.float32, [
                                    None, pad_len - window_size + 1, self.num_filters], name="mask_between")
        self.mask3 = tf.placeholder(
            tf.float32, [None, pad_len - window_size + 1, self.num_filters], name="mask_after")
        self.wps1 = tf.placeholder(tf.int32, [None, pad_len], name="wps1")
        self.wps2 = tf.placeholder(tf.int32, [None, pad_len], name="wps2")
        self.pad_len = pad_len
        self.window_size = window_size
        self.num_rels = num_rels
        self.PAD = len(word_vectors) - 1
        self.bag_num = tf.placeholder(
            tf.int32, [batch_num + 1], name="bag_num")
        self.soft_label_flag = tf.placeholder(
            tf.float32, [batch_num], name="soft_label_flag")
        self.joint_p = joint_p
        total_num = self.bag_num[-1]
        self.batch_num = batch_num
        l2_loss = tf.constant(0.0)

        with tf.device('/cpu:0'):
            self.embedding = tf.Variable(word_vectors, dtype=tf.float32)
            self.inputs = tf.nn.embedding_lookup(self.embedding, self.input)
        with tf.name_scope('joint'):
            wpe1 = tf.get_variable("wpe1", shape=[
                                   62, pos_embedding], initializer=tf.contrib.layers.xavier_initializer())
            wpe2 = tf.get_variable("wpe2", shape=[
                                   62, pos_embedding], initializer=tf.contrib.layers.xavier_initializer())
            pos_left = tf.nn.embedding_lookup(wpe1, self.wps1)
            pos_right = tf.nn.embedding_lookup(wpe2, self.wps2)
            self.pos_embed = tf.concat([pos_left, pos_right], 2)
        with tf.name_scope('conv'):
            self._input = tf.concat([self.inputs, self.pos_embed], 2)
            filter_shape = [window_size, embedding_size +
                            2 * pos_embedding, 1, num_filters]
            W = tf.get_variable("conv-W", shape=filter_shape,
                                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.get_variable(
                "conv-b", shape=[num_filters], initializer=tf.contrib.layers.xavier_initializer())
            self.conv = tf.nn.conv2d(tf.expand_dims(
                self._input, -1), W, strides=[1, 1, 1, 1], padding="VALID", name="conv")
            h = tf.nn.tanh(tf.nn.bias_add(self.conv, b), name="tanh")
            self.h1 = tf.add(h, tf.expand_dims(self.mask1, 2))
            self.h2 = tf.add(h, tf.expand_dims(self.mask2, 2))
            self.h3 = tf.add(h, tf.expand_dims(self.mask3, 2))
            pooled1 = tf.nn.max_pool(self.h1, ksize=[
                                     1, self.pad_len - self.window_size + 1, 1, 1], strides=[1, 1, 1, 1], padding="VALID", name="pool")
            poolre1 = tf.reshape(pooled1, [-1, self.num_filters])
            pooled2 = tf.nn.max_pool(self.h2, ksize=[
                                     1, self.pad_len - self.window_size + 1, 1, 1], strides=[1, 1, 1, 1], padding="VALID", name="pool")
            poolre2 = tf.reshape(pooled2, [-1, self.num_filters])
            pooled3 = tf.nn.max_pool(self.h3, ksize=[
                                     1, self.pad_len - self.window_size + 1, 1, 1], strides=[1, 1, 1, 1], padding="VALID", name="pool")
            poolre3 = tf.reshape(pooled3, [-1, self.num_filters])
            poolre = tf.concat([poolre1, poolre2, poolre3], 1)
            pooled = tf.nn.dropout(poolre, dropout)
        with tf.name_scope("map"):
            W = tf.get_variable(
                "W",
                shape=[3 * self.num_filters, self.num_rels],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.get_variable(
                "b", shape=[self.num_rels], initializer=tf.contrib.layers.xavier_initializer())
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)

            # the implementation of Lin et al 2016 comes from https://github.com/thunlp/TensorFlow-NRE/blob/master/network.py
            sen_a = tf.get_variable("attention_A", [
                                    3 * self.num_filters], initializer=tf.contrib.layers.xavier_initializer())
            sen_q = tf.get_variable("query", [
                                    3 * self.num_filters, 1], initializer=tf.contrib.layers.xavier_initializer())
            sen_r = []
            sen_s = []
            sen_out = []
            sen_alpha = []
            self.bag_score = []
            self.predictions = []
            self.losses = []
            self.accuracy = []
            self.total_loss = 0.0
            # selective attention model, use the weighted sum of all related the sentence vectors as bag representation
            for i in range(batch_num):
                sen_r.append(pooled[self.bag_num[i]:self.bag_num[i + 1]])
                bag_size = self.bag_num[i + 1] - self.bag_num[i]
                sen_alpha.append(tf.reshape(tf.nn.softmax(tf.reshape(
                    tf.matmul(tf.multiply(sen_r[i], sen_a), sen_q), [bag_size])), [1, bag_size]))
                sen_s.append(tf.reshape(tf.matmul(sen_alpha[i], sen_r[i]), [
                             1, 3 * self.num_filters]))
                sen_out.append(tf.reshape(tf.nn.xw_plus_b(
                    sen_s[i], W, b), [self.num_rels]))
                self.bag_score.append(tf.nn.softmax(sen_out[i]))

                with tf.name_scope("output"):
                    self.predictions.append(
                        tf.argmax(self.bag_score[i], 0, name="predictions"))

                with tf.name_scope("loss"):

                    nscor = self.soft_label_flag[i] * self.bag_score[i] + joint_p * tf.reduce_max(
                        self.bag_score[i]) * tf.cast(self.preds[i], tf.float32)
                    self.nlabel = tf.reshape(tf.one_hot(indices=[tf.argmax(
                        nscor, 0)], depth=self.num_rels, dtype=tf.int32), [self.num_rels])
                    self.ccc = self.preds[i]
                    self.losses.append(tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                        logits=sen_out[i], labels=self.nlabel)))

                    if i == 0:
                        self.total_loss = self.losses[i]
                    else:
                        self.total_loss += self.losses[i]

                with tf.name_scope("accuracy"):
                    self.accuracy.append(tf.reduce_mean(tf.cast(tf.equal(
                        self.predictions[i], tf.argmax(self.preds[i], 0)), "float"), name="accuracy"))

        with tf.name_scope("update"):
            self.global_step = tf.Variable(
                0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
            self.train_op = optimizer.minimize(
                self.total_loss, global_step=self.global_step)

    # pad sentences for piecewise max-pooling operation described in
    # "Distant Supervision for Relation Extraction via Piecewise Convolutional Neural Networks"
    def sen_padding(self, sen_id, instance, lpos, rpos, real_sen, namepos):
        # instance :[5, 233, 3232, ...] sentences in bag
        instances = []
        mask_before = []
        mask_between = []
        mask_after = []
        lwps = []
        rwps = []
        for id in instance:
            seq = sen_id[id]
            wps_left = lpos[id]
            wps_right = rpos[id]
            en1, en2 = namepos[id]
            t1, t2 = self.get_split(en1, en2)
            seq_len = len(real_sen[id].split())
            assert seq_len <= self.pad_len
            if seq_len <= self.pad_len:
                mask1 = np.zeros(
                    [self.pad_len - self.window_size + 1, self.num_filters], dtype=float)
                mask1[t1 + 1:, :] = -100.0
                mask2 = np.zeros(
                    [self.pad_len - self.window_size + 1, self.num_filters], dtype=float)
                mask2[:t1, :] = -100.0
                mask2[t2 + 1:, :] = -100.0
                mask3 = np.zeros(
                    [self.pad_len - self.window_size + 1, self.num_filters], dtype=float)
                mask3[:t2, :] = -100.0
                mask3[seq_len - self.window_size + 1:, :] = -100.0
                # mask = [1] * (seq_len-self.window_size+1) + [0] * (self.pad_len-seq_len)
            if len(seq) < self.pad_len:
                llen = self.pad_len - len(seq)
                seq.extend([self.PAD] * (self.pad_len - len(seq)))
                wps_left.extend([61] * llen)
                wps_right.extend([61] * llen)
            mask_before.append(mask1)
            mask_between.append(mask2)
            mask_after.append(mask3)
            instances.append(seq)
            lwps.append(wps_left)
            rwps.append(wps_right)
        return instances, mask_before, mask_between, mask_after, lwps, rwps

    def get_split(self, en1, en2):
        t1, t2 = en1, en2
        if en1 > en2:
            t1 = en2
            t2 = en1
        assert t1 <= t2
        return t1, t2

    def train(self, sess, bag_key, train_bag, sen_id, lpos, rpos, real_sen, namepos, use_soft_label=False):
        # bag_key: mid1 mid2 rel
        batch = []
        pred = []
        mask_before = []
        mask_between = []
        mask_after = []
        wps_left = []
        wps_right = []
        batch_sen_num = []
        soft_label_flag = []
        cnt_sen = 0
        for key in bag_key:
            rel = int(key.split('\t')[-1])
            if use_soft_label:
                soft_label_flag.append(1)
            else:
                soft_label_flag.append(0)
            sentences = train_bag[key]
            sen_vec, mask_bef, mask_bet, mask_aft, llpos, rrpos = self.sen_padding(
                sen_id, sentences, lpos, rpos, real_sen, namepos)
            batch.extend(sen_vec)
            mask_before.extend(mask_bef)
            mask_between.extend(mask_bet)
            mask_after.extend(mask_aft)
            pred.append(rel)
            wps_left.extend(llpos)
            wps_right.extend(rrpos)
            batch_sen_num.append(cnt_sen)
            cnt_sen += len(sentences)
        batch_sen_num.append(cnt_sen)
        preds = np.zeros([len(bag_key), self.num_rels])
        preds[np.arange(len(bag_key)), pred] = 1
        _, hh, loss, acc, step = sess.run([self.train_op, self.h1, self.total_loss, self.accuracy, self.global_step], feed_dict={
            self.input: batch,
            self.mask1: mask_before,
            self.mask2: mask_between,
            self.mask3: mask_after,
            self.preds: preds,
            self.wps1: wps_left,
            self.wps2: wps_right,
            self.bag_num: batch_sen_num,
            self.soft_label_flag: soft_label_flag
        })

        assert np.min(np.max(hh, axis=1)) > -50.0
        acc = np.reshape(np.array(acc), (self.batch_num))
        acc = np.mean(acc)
        return loss

    def test(self, sess, bag_key, test_bag, sen_id, lpos, rpos, real_sen, namepos):
        # bag_key: mid1 mid2
        pair_score = []
        cnt_i = 1
        batches = util.batch_iter(bag_key, self.batch_num, 1, shuffle=True)
        for bat in batches:
            if len(bat) < self.batch_num:
                continue
            batch = []
            mask_before = []
            mask_between = []
            mask_after = []
            wps_left = []
            wps_right = []
            batch_sen_num = []
            cnt_sen = 0
            for key in bat:
                # sys.stdout.write('testing %d cases...\r' % cnt_i
                # sys.stdout.flush()
                cnt_i += 1
                sentences = test_bag[key]
                sen_vec, mask_bef, mask_bet, mask_aft, llpos, rrpos = self.sen_padding(
                    sen_id, sentences, lpos, rpos, real_sen, namepos)
                batch.extend(sen_vec)
                mask_before.extend(mask_bef)
                mask_between.extend(mask_bet)
                mask_after.extend(mask_aft)
                wps_left.extend(llpos)
                wps_right.extend(rrpos)
                batch_sen_num.append(cnt_sen)
                cnt_sen += len(sentences)
            batch_sen_num.append(cnt_sen)
            soft_label_flag = [0] * len(bat)

            scores = sess.run(self.bag_score, feed_dict={
                self.input: batch,
                self.mask1: mask_before,
                self.mask2: mask_between,
                self.mask3: mask_after,
                self.wps1: wps_left,
                self.wps2: wps_right,
                self.bag_num: batch_sen_num,
                self.soft_label_flag: soft_label_flag})
            # score = np.max(scores, axis=0)
            for k, key in enumerate(bat):
                for i, sc in enumerate(scores[k]):
                    if i == 0:
                        continue
                    pair_score.append({"mid": key, "rel": i, "score": sc})
        return pair_score

# At-Least-One Model


class one_model(object):
    def __init__(self, pad_len, num_rels, word_vectors, window_size, num_filters, embedding_size, pos_embedding, dropout, joint_p, batch_num=None, l2_reg=0.0):
        self.input = tf.placeholder(tf.int32, [None, pad_len], name="input")
        self.preds = tf.placeholder(tf.int32, [None, num_rels], name="preds")
        self.num_filters = num_filters
        self.mask1 = tf.placeholder(tf.float32, [
                                    None, pad_len - window_size + 1, self.num_filters], name="mask_before")
        self.mask2 = tf.placeholder(tf.float32, [
                                    None, pad_len - window_size + 1, self.num_filters], name="mask_between")
        self.mask3 = tf.placeholder(
            tf.float32, [None, pad_len - window_size + 1, self.num_filters], name="mask_after")
        self.wps1 = tf.placeholder(
            tf.float32, [None, pad_len, 61], name="wps1")
        self.wps2 = tf.placeholder(
            tf.float32, [None, pad_len, 61], name="wps2")
        self.pad_len = pad_len
        self.window_size = window_size
        self.num_rels = num_rels
        self.PAD = len(word_vectors) - 1
        self.joint_p = joint_p
        self.soft_label_flag = tf.placeholder(
            tf.float32, [None], name="soft_label_flag")
        l2_loss = tf.constant(0.0)

        with tf.device('/cpu:0'):
            self.embedding = tf.Variable(word_vectors, dtype=tf.float32)
            self.inputs = tf.nn.embedding_lookup(self.embedding, self.input)
        with tf.name_scope('joint'):
            wpe1 = tf.Variable(tf.truncated_normal(
                [61, pos_embedding], stddev=0.01), name="wpe1")
            wpe2 = tf.Variable(tf.truncated_normal(
                [61, pos_embedding], stddev=0.01), name="wpe2")
            pos_left = tf.reshape(tf.matmul(tf.reshape(
                self.wps1, [-1, 61]), wpe1), [-1, pad_len, pos_embedding])
            pos_right = tf.reshape(tf.matmul(tf.reshape(
                self.wps2, [-1, 61]), wpe2), [-1, pad_len, pos_embedding])
            self.pos_embed = tf.concat([pos_left, pos_right], 2)
        with tf.name_scope('conv'):
            self._input = tf.concat([self.inputs, self.pos_embed], 2)
            filter_shape = [window_size, embedding_size +
                            2 * pos_embedding, 1, num_filters]
            W = tf.Variable(tf.truncated_normal(
                filter_shape, stddev=0.1), name="W")
            b = tf.Variable(tf.constant(0.01, shape=[num_filters]), name="b")
            self.conv = tf.nn.conv2d(tf.expand_dims(
                self._input, -1), W, strides=[1, 1, 1, 1], padding="VALID", name="conv")
            h = tf.nn.tanh(tf.nn.bias_add(self.conv, b), name="tanh")
            self.h1 = tf.add(h, tf.expand_dims(self.mask1, 2))
            self.h2 = tf.add(h, tf.expand_dims(self.mask2, 2))
            self.h3 = tf.add(h, tf.expand_dims(self.mask3, 2))
            pooled1 = tf.nn.max_pool(self.h1, ksize=[
                                     1, self.pad_len - self.window_size + 1, 1, 1], strides=[1, 1, 1, 1], padding="VALID", name="pool")
            poolre1 = tf.reshape(pooled1, [-1, self.num_filters])
            pooled2 = tf.nn.max_pool(self.h2, ksize=[
                                     1, self.pad_len - self.window_size + 1, 1, 1], strides=[1, 1, 1, 1], padding="VALID", name="pool")
            poolre2 = tf.reshape(pooled2, [-1, self.num_filters])
            pooled3 = tf.nn.max_pool(self.h3, ksize=[
                                     1, self.pad_len - self.window_size + 1, 1, 1], strides=[1, 1, 1, 1], padding="VALID", name="pool")
            poolre3 = tf.reshape(pooled3, [-1, self.num_filters])
            poolre = tf.concat([poolre1, poolre2, poolre3], 1)
            self.pooled = tf.nn.dropout(poolre, dropout)
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[3 * self.num_filters, self.num_rels],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[self.num_rels]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.linear_scores = tf.nn.xw_plus_b(
                self.pooled, W, b, name="scores")
            self.scores = tf.nn.softmax(self.linear_scores)
            nscore = tf.expand_dims(self.soft_label_flag, -1) * self.scores + self.joint_p * tf.reshape(
                tf.reduce_max(self.scores, 1), [-1, 1]) * tf.cast(self.preds, tf.float32)
            self.nlabel = tf.one_hot(indices=tf.reshape(
                tf.argmax(nscore, axis=1), [-1]), depth=self.num_rels, dtype=tf.int32)
            losses = tf.nn.softmax_cross_entropy_with_logits(
                logits=self.linear_scores, labels=self.nlabel)
            self.loss = tf.reduce_mean(losses) + l2_reg * l2_loss
        with tf.name_scope("update"):
            self.global_step = tf.Variable(
                0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
            self.train_op = optimizer.minimize(
                self.loss, global_step=self.global_step)

    def sen_padding(self, sen_id, instance, lpos, rpos, real_sen, namepos):
        # instance :[5, 233, 3232, ...] sentences in bag
        instances = []
        mask_before = []
        mask_between = []
        mask_after = []
        lwps = []
        rwps = []
        for id in instance:
            seq = sen_id[id]
            wps_left = lpos[id]
            wps_right = rpos[id]
            en1, en2 = namepos[id]
            t1, t2 = self.get_split(en1, en2)
            seq_len = len(real_sen[id].split())
            llwps, rrwps = self.pos_padding(wps_left, wps_right, seq_len)
            assert seq_len <= self.pad_len
            if seq_len <= self.pad_len:
                mask1 = np.zeros(
                    [self.pad_len - self.window_size + 1, self.num_filters], dtype=float)
                mask1[t1 + 1:, :] = -100.0
                mask2 = np.zeros(
                    [self.pad_len - self.window_size + 1, self.num_filters], dtype=float)
                mask2[:t1, :] = -100.0
                mask2[t2 + 1:, :] = -100.0
                mask3 = np.zeros(
                    [self.pad_len - self.window_size + 1, self.num_filters], dtype=float)
                mask3[:t2, :] = -100.0
                mask3[seq_len - self.window_size + 1:, :] = -100.0
                # mask = [1] * (seq_len-self.window_size+1) + [0] * (self.pad_len-seq_len)
            if len(seq) < self.pad_len:
                seq.extend([self.PAD] * (self.pad_len - len(seq)))
            mask_before.append(mask1)
            mask_between.append(mask2)
            mask_after.append(mask3)
            instances.append(seq)
            lwps.append(llwps)
            rwps.append(rrwps)
        return instances, mask_before, mask_between, mask_after, lwps, rwps

    def get_split(self, en1, en2):
        t1, t2 = en1, en2
        if en1 > en2:
            t1 = en2
            t2 = en1
        assert t1 <= t2
        return t1, t2

    def pos_padding(self, wps_left, wps_right, llen):
        pos_left = np.zeros([llen, 61], dtype=int)
        pos_right = np.zeros([llen, 61], dtype=int)
        pos_left[np.arange(llen), wps_left] = 1
        pos_right[np.arange(llen), wps_right] = 1
        if llen < self.pad_len:
            pad = np.zeros([self.pad_len - llen, 61], dtype=int)
            pad[np.arange(self.pad_len - llen), [60] *
                (self.pad_len - llen)] = 1
            pos_left = np.concatenate((pos_left, pad), axis=0)
            pos_right = np.concatenate((pos_right, pad), axis=0)
        return pos_left, pos_right

    def train(self, sess, bag_key, train_bag, sen_id, lpos, rpos, real_sen, namepos, use_soft_label=False):
        # bag_key: mid1 mid2 rel
        batch = []
        pred = []
        mask_before = []
        mask_between = []
        mask_after = []
        wps_left = []
        wps_right = []
        soft_label_flag = []
        for key in bag_key:
            rel = int(key.split('\t')[-1])
            if use_soft_label:
                soft_label_flag.append(1)
            else:
                soft_label_flag.append(0)
            sentences = train_bag[key]
            sen_vec, mask_bef, mask_bet, mask_aft, llpos, rrpos = self.sen_padding(
                sen_id, sentences, lpos, rpos, real_sen, namepos)
            scores = sess.run(self.linear_scores, feed_dict={
                              self.input: sen_vec, self.mask1: mask_bef, self.mask2: mask_bet, self.mask3: mask_aft, self.wps1: llpos, self.wps2: rrpos})
            id_max = np.argmax(scores[:, rel])
            batch.append(sen_vec[id_max])
            mask_before.append(mask_bef[id_max])
            mask_between.append(mask_bet[id_max])
            mask_after.append(mask_aft[id_max])
            pred.append(rel)
            wps_left.append(llpos[id_max])
            wps_right.append(rrpos[id_max])
        preds = np.zeros([len(bag_key), self.num_rels])
        preds[np.arange(len(bag_key)), pred] = 1
        loss, step, inpp, _, ip, conv, hh, pp = sess.run([self.loss, self.global_step, self.inputs, self.train_op, self._input, self.conv, self.h1, self.pooled], feed_dict={
            self.input: batch,
            self.mask1: mask_before,
            self.mask2: mask_between,
            self.mask3: mask_after,
            self.preds: preds,
            self.wps1: wps_left,
            self.wps2: wps_right,
            self.soft_label_flag: soft_label_flag
        })
        assert np.min(np.max(hh, axis=1)) > -50.0
        return loss

    def test(self, sess, bag_key, test_bag, sen_id, lpos, rpos, real_sen, namepos):
        # bag_key: mid1 mid2
        pair_score = []
        cnt_i = 1
        for key in bag_key:
            # sys.stdout.write('testing %d cases...\r' % cnt_i
            # sys.stdout.flush()
            cnt_i += 1
            sentences = test_bag[key]
            sen_vec, mask_bef, mask_bet, mask_aft, llpos, rrpos = self.sen_padding(
                sen_id, sentences, lpos, rpos, real_sen, namepos)
            scores = sess.run(self.scores, feed_dict={self.input: sen_vec, self.mask1: mask_bef, self.mask2: mask_bet,
                                                      self.mask3: mask_aft, self.wps1: llpos, self.wps2: rrpos, self.soft_label_flag: [0]})
            score = np.max(scores, axis=0)
            for i, sc in enumerate(score):
                if i == 0:
                    continue
                pair_score.append({"mid": key, "rel": i, "score": sc})
        return pair_score

# Main class


class soft_label_RE(RelationExtractionModel):


def __init__(self):

    # Model Setting
    tf.app.flags.DEFINE_integer(
        "pad_len", 200, "Pad sentences to this length for convolution.")
    tf.app.flags.DEFINE_integer(
        "embedding_size", 50, "Size of word embedding.")
    tf.app.flags.DEFINE_integer(
        "pos_embedding", 5, "Size of position embedding.")
    tf.app.flags.DEFINE_integer(
        "batch_num", 160, "Batch size for sentence encoding.")
    tf.app.flags.DEFINE_integer(
        "num_rels", 53, "Number of pre-defined relations.")
    tf.app.flags.DEFINE_integer("window_size", 3, "Size of sliding window.")
    tf.app.flags.DEFINE_integer(
        "num_filters", 230, "Number of filters for convolution.")
    tf.app.flags.DEFINE_float("dropout", 0.7, 'dropout')

    tf.app.flags.DEFINE_string(
        "one_or_att", 'one', 'at-least-one or selective attention model')
    tf.app.flags.DEFINE_boolean("use_pre_train_model",
                                False, 'use pre-trained model or label')
    tf.app.flags.DEFINE_string("load_model_name", 'pretrain/model1.ckpt-3300',
                               'the path of pre-trained model without soft-label')
    tf.app.flags.DEFINE_boolean("save_model", True, 'save models or not')

    tf.app.flags.DEFINE_boolean(
        "use_soft_label", False, 'use soft label or not')
    tf.app.flags.DEFINE_float(
        "confidence", 0.9, 'confidence of distant-supervised label')

    tf.app.flags.DEFINE_string("dir", 'res', 'dir to store results')
    tf.app.flags.DEFINE_integer(
        "report", 100, "report loss & save models after every *** batches.")
    self.FLAGS = tf.app.flags.FLAGS

    # =================== make new dirs =================== #
    prefix = str(int(time.time() * 1000))
    # dir to save all the results in this run
    top_dir = os.path.join(FLAGS.dir, prefix)
    if not os.path.exists(FLAGS.dir):
        os.mkdir(FLAGS.dir)
    if not os.path.exists(top_dir):
        os.mkdir(top_dir)
    checkpoint_dir = os.path.join(top_dir, "checkpoint")  # dir to save models
    log_file = os.path.join(top_dir, 'log.txt')

    def write_log(s):
        print(s)
        with open(log_file, 'a') as f:
            f.write(s + '\n')


def read_dataset(self):
    # =================== load data =================== #
    print("load training and testing data ...")
    start_time = time.time()
    vect = word2vec()  # load pre-trained word vector
    # load vocabulary and pre-defined word vectors
    word_vocab, word_vector = get_word_vec(vect, one_or_att=FLAGS.one_or_att)
    '''
    bag_train: a dict , key is triple (h, r, t), related value is the list of sentence ids which contain the triple.
    sen_id: idlized sentences in the training data.
    real_sen: original sentences in the training set.
    lpos/ rpos: the distance of each token to the head/tail entities, for position embedding.
    keypos: the position of two key (head and tail) entities in the sentences.
    '''
    bag_train, sen_id, lpos, rpos, real_sen, keypos = get_data(
        istrain=True, word_vocab=word_vocab)
    bag_test, sen_id1, midrel, ltpos, rtpos, real_sen1, keypos1 = get_data(
        istrain=False, word_vocab=word_vocab)
    bag_keys = bag_train.keys()

    span = time.time() - start_time
    print("training and testing data loaded, using %.3f seconds" % span)
    write_log("training size: %d   testing size: %d" %
              (len(bag_train.keys()), len(bag_test.keys())))


def data_preprocess(self):
    pass


def tokenize(self):
    pass


def train(self):
    # =================== model initialization =================== #
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    if FLAGS.one_or_att == 'att':
        load_model = attmodel
    else:
        load_model = onemodel
    model = load_model.model(pad_len=FLAGS.pad_len,
                             num_rels=FLAGS.num_rels,
                             word_vectors=word_vector,
                             window_size=FLAGS.window_size,
                             num_filters=FLAGS.num_filters,
                             embedding_size=FLAGS.embedding_size,
                             dropout=FLAGS.dropout,
                             pos_embedding=FLAGS.pos_embedding,
                             batch_num=FLAGS.batch_num,
                             joint_p=FLAGS.confidence)

    saver = tf.train.Saver(max_to_keep=70)
    if FLAGS.use_pre_train_model:
        saver.restore(sess, FLAGS.load_model_name)
        write_log("load pre-trained model from " + FLAGS.load_model_name)
    # ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    else:
        sess.run(tf.global_variables_initializer())
        write_log("create new model")
    print "Initial model complete"


# =================== training stage =================== #
batches = batch_iter(bag_train.keys(), FLAGS.batch_num, 20)
loss, start_time = 0.0, time.time()
for batch in batches:
    if len(batch) < FLAGS.batch_num:
        continue
    loss += model.train(sess, batch, bag_train, sen_id, lpos,
                        rpos, real_sen, keypos, FLAGS.use_soft_label)
    step = tf.train.global_step(sess, model.global_step)
    progress_bar(step % FLAGS.report, FLAGS.report)
    if step % FLAGS.report == 0:  # report PR-curve results on the testing set
        cost_time = time.time() - start_time
        epoch = step // FLAGS.report
        write_log("%d : loss = %.3f, time = %.3f " %
                  (step // FLAGS.report, loss, cost_time))
        print "evaluating after epoch " + str(epoch)
        pair_score = model.test(sess, bag_test.keys(
        ), bag_test, sen_id1, ltpos, rtpos, real_sen1, keypos1)
        evaluate(top_dir + "/pr" + str(epoch) +
                 ".txt", pair_score, midrel, epoch)
        loss, start_time = 0.0, time.time()
        if FLAGS.save_model:
            checkpoint_path = os.path.join(checkpoint_dir, "model.ckpt")
            saver.save(sess, checkpoint_path, global_step=model.global_step)
            write_log("save model in " + str(sess.run(model.global_step)))


def predict(self):
    # Predict and Evaluate were done inside train, will seperate later
    pass


def evaluate(self):
    # Predict and Evaluate were done inside train, will seperate later
    pass
