import tensorflow as tf
from config import FLAGS

class CPM_Model():
    def __init__(self):
        self.stages = FLAGS.total_stages
        self.stage_maps = []
        self.stage_loss = [0] * self.stages
        self.joints = FLAGS.total_joints

        self.input_images = tf.placeholder(dtype=tf.float32,
                                           shape=(None, FLAGS.image_size, FLAGS.image_size, 3))
        
        self.hmap_placeholder = tf.placeholder(dtype=tf.float32,
                                                  shape=(None, FLAGS.hmap_size, FLAGS.hmap_size, self.joints + 1))
        self.first_stage()
        for stage in range(2, self.stages + 1):
            self.add_middle_stage(stage)

        
    def first_stage(self):
        with tf.variable_scope('stage_x'):
            Xconv1 = tf.layers.conv2d(self.input_images, 64, [3, 3], [1, 1], 'same', activation = tf.nn.relu, 
                                  kernel_initializer = tf.contrib.layers.xavier_initializer(), name='Xconv1')
            Xconv2 = tf.layers.conv2d(Xconv1, 64, [3, 3], [1, 1], 'same', activation = tf.nn.relu, 
                                  kernel_initializer = tf.contrib.layers.xavier_initializer(), name='Xconv2')
            Xpool1 = tf.layers.max_pooling2d(Xconv2, [2, 2], [2, 2], name='Xpool1')
            #######################
            Xconv3 = tf.layers.conv2d(Xpool1, 128, [3, 3], [1, 1], 'same', activation = tf.nn.relu, 
                                  kernel_initializer = tf.contrib.layers.xavier_initializer(), name='Xconv3')
            Xconv4 = tf.layers.conv2d(Xconv3, 128,[3, 3], [1, 1], 'same', activation = tf.nn.relu, 
                                  kernel_initializer = tf.contrib.layers.xavier_initializer(), name='Xconv4')
            Xpool2 = tf.layers.max_pooling2d(Xconv4,[2, 2], [2, 2], name='Xpool2')
            #######################
            Xconv5 = tf.layers.conv2d(Xpool2, 256, [3, 3], [1, 1], 'same', activation = tf.nn.relu, 
                                  kernel_initializer = tf.contrib.layers.xavier_initializer(), name='Xconv5')
            Xconv6 = tf.layers.conv2d(Xconv5, 256, [3, 3], [1, 1], 'same', activation = tf.nn.relu, 
                                  kernel_initializer = tf.contrib.layers.xavier_initializer(), name='Xconv6')
            Xconv7 = tf.layers.conv2d(Xconv6, 256, [3, 3], [1, 1], 'same', activation = tf.nn.relu, 
                                  kernel_initializer = tf.contrib.layers.xavier_initializer(), name='Xconv7')
            Xconv8 = tf.layers.conv2d(Xconv7, 256, [3, 3], [1, 1], 'same', activation = tf.nn.relu, 
                                  kernel_initializer = tf.contrib.layers.xavier_initializer(), name='Xconv8')
            Xpool3 = tf.layers.max_pooling2d(Xconv8, [2, 2], [2, 2], name='Xpool3')
            #######################
            Xconv9 = tf.layers.conv2d(Xpool3, 512, [3, 3], [1, 1], 'same', activation = tf.nn.relu, 
                                  kernel_initializer = tf.contrib.layers.xavier_initializer(), name='Xconv9')
            Xconv10 = tf.layers.conv2d(Xconv9, 512, [3, 3], [1, 1], 'same', activation = tf.nn.relu, 
                                  kernel_initializer = tf.contrib.layers.xavier_initializer(), name='Xconv10')

            Xconv11 = tf.layers.conv2d(Xconv10, 512, [3, 3], [1, 1], 'same', activation = tf.nn.relu, 
                                  kernel_initializer = tf.contrib.layers.xavier_initializer(), name='Xconv11')
            Xconv12 = tf.layers.conv2d(Xconv11, 512, [3, 3], [1, 1], 'same', activation = tf.nn.relu, 
                                  kernel_initializer = tf.contrib.layers.xavier_initializer(), name='Xconv12')
            Xconv13 = tf.layers.conv2d(Xconv12, 512, [3, 3], [1, 1], 'same', activation = tf.nn.relu, 
                                  kernel_initializer = tf.contrib.layers.xavier_initializer(), name='Xconv13')
            Xconv14 = tf.layers.conv2d(Xconv13, 512, [3, 3], [1, 1], 'same', activation = tf.nn.relu, 
                                  kernel_initializer = tf.contrib.layers.xavier_initializer(), name='Xconv14')
            self.stage_x = tf.layers.conv2d(Xconv14, 128, [3, 3], [1, 1], 'same', activation = tf.nn.relu, 
                                      kernel_initializer = tf.contrib.layers.xavier_initializer(), name='stage_x')

        with tf.variable_scope('stage_1'):
            conv1 = tf.layers.conv2d(self.stage_x, 512, [1, 1], [1, 1], activation = tf.nn.relu, 
                                kernel_initializer = tf.contrib.layers.xavier_initializer(), name='conv1')
            conv2 = tf.layers.conv2d(conv1, self.joints + 1, [1, 1], [1, 1], activation = None, kernel_initializer = tf.contrib.layers.xavier_initializer(), name='conv2')
            self.stage_maps.append(conv2)
                                               
    def add_middle_stage(self, stage):
        with tf.variable_scope('stage_' + str(stage)):
            input_layer = tf.concat([self.stage_maps[stage - 2],self.stage_x],axis=3)
            Mconv1 = tf.layers.conv2d(input_layer, 128, [7, 7], [1, 1], 'same', activation = tf.nn.relu, 
                                kernel_initializer = tf.contrib.layers.xavier_initializer(), name='Mconv1')
            Mconv2 = tf.layers.conv2d(Mconv1, 128, [7, 7], [1, 1], 'same', activation = tf.nn.relu, 
                                kernel_initializer = tf.contrib.layers.xavier_initializer(), name='Mconv2')
            Mconv3 = tf.layers.conv2d(Mconv2, 128, [7, 7], [1, 1], 'same', activation = tf.nn.relu, 
                                kernel_initializer = tf.contrib.layers.xavier_initializer(), name='Mconv3')
            Mconv4 = tf.layers.conv2d(Mconv3, 128, [7, 7], [1, 1], 'same', activation = tf.nn.relu, 
                                kernel_initializer = tf.contrib.layers.xavier_initializer(), name='Mconv4')
            Mconv5 = tf.layers.conv2d(Mconv4, 128, [7, 7], [1, 1], 'same', activation = tf.nn.relu, 
                                kernel_initializer = tf.contrib.layers.xavier_initializer(), name='Mconv5')
            Mconv6 = tf.layers.conv2d(Mconv5, 128, [7, 7], [1, 1], 'same', activation = tf.nn.relu, 
                                kernel_initializer = tf.contrib.layers.xavier_initializer(), name='Mconv6')
            Mconv7 = tf.layers.conv2d(Mconv6, self.joints + 1, [1, 1], [1, 1], activation = None, kernel_initializer = tf.contrib.layers.xavier_initializer(), name='Mconv7')

            self.stage_maps.append(Mconv7)

    def build_loss(self):
        self.batch_size = tf.cast(tf.shape(self.input_images)[0], dtype=tf.float32)
        for i in range(self.stages):
            self.stage_loss[i] = tf.nn.l2_loss(self.stage_maps[i] - self.hmap_placeholder,name='l2_loss') / self.batch_size
        
        self.total_loss = 0
        for i in range(self.stages):
            self.total_loss += self.stage_loss[i]

        self.global_step = tf.train.get_or_create_global_step()

        self.new_lr = tf.train.exponential_decay(FLAGS.lr, self.global_step, FLAGS.l_decay_step, FLAGS.l_decay_rate)

        self.train_op = tf.contrib.layers.optimize_loss(self.total_loss, self.global_step, self.new_lr, FLAGS.optimizer)
