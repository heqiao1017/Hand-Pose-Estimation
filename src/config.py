class FLAGS(object):
    '''Common settings'''
    image_size = 256 #input size for model
    hmap_size = 32 #label and predictions in the format of heatmap: 32*32*22
    total_stages = 3 #number of stages in the model
    joint_gaussian_variance = 1.0 #the gaussian distribution for heatmap
    center_radius = 21 #
    total_joints = 21 #number of joints, fixed
    use_gpu = True #GPU settings
    gpu_id = 0 #GPU settings
    box_size = 256 #tfrecord size
    optimizer = 'RMSProp' #optimizer option
    output_node_names = 'stage_3/Mconv7/BiasAdd:0' #the last layer of the model, will be the output
    DEMO_TYPE = 'test_video/sample.mp4'#'test_img' #'test_video/test.mp4' # #can be a .png/.jpg image, a folder with ONLY images inside
    #note that for a video, it's normal to have hundreds of photos or more, so it will take much longer time to produce


    '''Training settings'''
    network_def = 'cpm_model' #the name of the model, should not be changed
    train_tf_file = 'train.tfrecords' #the training tfrecord
    val_tf_file = 'test.tfrecords' #the testing tfrecord
    pretrained_model = 'cpm_model-200000' #the model to reload weights to train or predict, usually keep in format cpm_model-step
    batch_size = 5 #how many data will Ensemble Data Generator take from tfrecords for once
    lr = 0.04 #learning rate
    l_decay_rate = 0.5 #learning decay rate
    l_decay_step = 10000 #learning decay step
    training_iters = 250000 #training iterations
    validation_iters = 5000 #do validation every validation_iters iterations
    model_save_iters = 25000 #save every model_save_iters itertions
    validation_batch_per_iter = 20 #how many validation batch per iteration
    padding = 15 #bounding box with padding

    '''Drawing settings'''

    #Connections between joints
    limbs = [[0, 1],[1 , 2],[2 , 3],[3 , 4],
                [0, 5],[5 , 6],[6 , 7],[7 , 8],
                [0, 9],[9 ,10],[10,11],[11,12],
                [0,13],[13,14],[14,15],[15,16],
                [0,17],[17,18],[18,19],[19,20]]

    #Finger colors
    joint_color_code = [
        [100.,  100.,  100.], 
        [100.,    0.,    0.],
        [150.,    0.,    0.],
        [200.,    0.,    0.],
        [255.,    0.,    0.],
        [100.,  100.,    0.],
        [150.,  150.,    0.],
        [200.,  200.,    0.],
        [255.,  255.,    0.],
        [  0.,  100.,   50.],
        [  0.,  150.,   75.],
        [  0.,  200.,  100.],
        [  0.,  255.,  125.],
        [  0.,   50.,  100.],
        [  0.,   75.,  150.],
        [  0.,  100.,  200.],
        [  0.,  125.,  255.],
        [100.,    0.,  100.],
        [150.,    0.,  150.],
        [200.,    0.,  200.],
        [255.,    0.,  255.]]





















