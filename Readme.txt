The trained data and result are too large so that I only uploaded the source code.







#################### Part 1: folder introduction ####################
Path of the folder								Description
project/hand_pose								Training data
project/src										Contain all other scripts
project/src/phase1_models						Contain training, graph-forming scripts and model structure of phase 1
project/src/phase2_models						Contain model pretrained weights, model structure of phase 2

project/test_img								Test image for test bound and hand pose
project/src/predict_img							Record the prediction from project/src/phase12_prediction.ipynb

project/test_video								A test video that illustrates video prediction
project/src/video_dir							Store intermediate result for video processing and output video



#################### Part 2: file introduction #####################

*SCRIPTS BELOW CAN BE RUN DIRECTLY:please run the script in the following order, you have to run the preprocessing in order to generate tfrecords to be able to start training*

[Phase 1 Hand Detection Proprocessing] Please execute following the order of index, same index in any order is okay
Path of the scripts								Description
1.project/src/phase1_json_to_csv.ipynb			Convert Json file to CSV file
2.project/src/phase1_illumination.ipynb			Add different illumination data to csv
2.project/src/phase1_rotate_90.ipynb			Add left rotate 90 degree data to csv
2.project/src/phase1_rotate_270.ipynb			Add right rotate 90 degree data to csv
3.project/src/phase1_split_partition.ipynb		Split with 8:2 partition into train and test
4.project/src/phase2_tfrecord.ipynb				Generate train and test records

[Phase 1 Hand Detection Training]
Path of the scripts								Description
project/src/phase1_models/research/
object_detection/train.py						Load tfrecord and .config, training
project/src/phase1_models/research/
object_detection/export_inference_graph.py		Build frozen graph for trained result

[Phase 1 Hand Detection Prediction]
Path of the scripts								Description
project/src/phase1_prediction.ipynb				Predict boundary box of an image

[Phase 2 Hand Pose Estimation Preprocessing]
Path of the scripts								Description
1.project/src/phase1_tfrecord.ipynb				Convert csv to tfrecords with proprocessing
2.project/src/phase2_load_data_from_tf.ipynb	Test if Ensemble_data_generator.py works correctly

[Phase 2 Hand Pose Estimation Training]
Path of the scripts								Description
project/src/training.py							train tfrecord with cpm_model(python3 training.py)


[Phase 2 Hand Pose Estimation Prediction]
Path of the scripts								Description
project/project.ipynb							Predict hand bouond and pose for images or videos



*SCRIPTS BELOW ARE NOT TO BE RUN DIRECTLY:*

Path of the scripts								Description
project/src/Detector.py							Tool scripts for phase 2 
project/src/Ensemble_data_generator.py			Read tfrecord for phase 2 training
project/src/video2jpg.py						Tools that convert the video into series of images
project/src/jpg2video.py						Tools that convert predicted images back into video
project/src/phase2_models/nets/cpm_model.py		The deep learning cpm model
project/src/config.py							The input arguments of the whole phase2 process, determine what to train and predict
project/src/phase1_models/research/
object_detection/training/
ssd_mobilenet_v1_pets.config					Pretrained model for phase 1



#################### Part 3: Phase 1 Instruction #####################
*Important!!!*

*Before running anything*
If run phase 1 object detection training process code, in the folder of project/src/phase1_models/research, before run any source code, please run the following command:

protoc object_detection/protos/*.proto --python_out=.

export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim


*Phase 1 training process*:
———
In the folder of project/src/phase1_models/research/object_detection run: 

python3 train.py --logtostderr --train_dir=training/ --pipeline_config_path=training/ssd_mobilenet_v1_pets.config  

———
then: In the folder of project/src/phase1_models/research/object_detection, please save the hand_inference_graph to a hand_inference_graph_copy and remove it in order to run: 

python3 export_inference_graph.py \
    --input_type image_tensor \
    --pipeline_config_path training/ssd_mobilenet_v1_pets.config \
    --trained_checkpoint_prefix training/model.ckpt-48769\
    --output_directory hand_inference_graph

This will generate the frozen graph in the hand_inference_graph folder, which will be used in the project/src/phase1_prediction.ipynb for the phase 1 prediction
