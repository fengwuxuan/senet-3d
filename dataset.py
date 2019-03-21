from datasets.kinetics import Kinetics
from datasets.activitynet import ActivityNet
from datasets.ucf101 import UCF101
from datasets.hmdb51 import HMDB51
from datasets.something import Something
from datasets.fire import FIRE


def get_training_set(opt, spatial_transform, temporal_transform,
                     target_transform):
    assert opt.dataset in ['kinetics', 'activitynet', 'ucf101', 'hmdb51','something','fire']

    if opt.dataset == 'kinetics':
        training_data = Kinetics(
            opt.video_path+"/train_256",
            opt.annotation_path,
            'training',
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            target_transform=target_transform)
    elif opt.dataset == 'activitynet':
        training_data = ActivityNet(
            opt.video_path,
            opt.annotation_path,
            'training',
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            target_transform=target_transform)
    elif opt.dataset == 'ucf101':
        opt.annotation_path = opt.annotation_path + "/train_rgb_ucf101.txt"
        training_data = UCF101(
            opt.video_path,
            opt.annotation_path,
            num_segments = opt.num_segments,
            modality = opt.modality,
            transform = spatial_transform)
    elif opt.dataset == 'hmdb51':
        opt.annotation_path = opt.annotation_path + "/train_rgb_hmdb51.txt"
        training_data = HMDB51(
            opt.video_path,
            opt.annotation_path,
            num_segments = opt.num_segments,
            modality = opt.modality, 
            transform = spatial_transform)
    elif opt.dataset == 'something':
        training_data = Something(
            opt.video_path,
            opt.annotation_path,
            'training',
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,                                                            
            target_transform=target_transform)
    elif opt.dataset == 'fire':
        opt.annotation_path = opt.annotation_path + "/train_fire.txt"
        training_data = FIRE(
            opt.video_path,
            opt.annotation_path,
            num_segments = opt.num_segments,
            modality = opt.modality,
            transform = spatial_transform)


    return training_data


def get_validation_set(opt, spatial_transform, temporal_transform,
                       target_transform):
    assert opt.dataset in ['kinetics', 'activitynet', 'ucf101', 'hmdb51','something','fire']

    if opt.dataset == 'kinetics':
        validation_data = Kinetics(
            opt.video_path+"/val_256",
            opt.annotation_path,
            'validation',
            opt.n_val_samples,
            spatial_transform,
            temporal_transform,
            target_transform,
            sample_duration=opt.sample_duration)
    elif opt.dataset == 'activitynet':
        validation_data = ActivityNet(
            opt.video_path,
            opt.annotation_path,
            'validation',
            opt.n_val_samples,
            spatial_transform,
            temporal_transform,
            target_transform,
            sample_duration=opt.sample_duration)
    elif opt.dataset == 'ucf101':
        opt.annotation_path = opt.annotation_path.replace("train","test")
        #opt.annotation_path = opt.annotation_path + "/test_rgb_ucf101.txt"
        validation_data = UCF101(
            opt.video_path,
            opt.annotation_path,
            num_segments = opt.num_segments,
            modality = opt.modality,
            transform = spatial_transform,
            test_mode = True)
    elif opt.dataset == 'hmdb51':
        opt.annotation_path = opt.annotation_path.replace("train","val")
        validation_data = HMDB51(
            opt.video_path,
            opt.annotation_path,
            num_segments = opt.num_segments,
            modality = opt.modality,
            transform = spatial_transform)
    elif opt.dataset == 'fire':
        opt.annotation_path = opt.annotation_path.replace("train","test")
        validation_data = FIRE(
            opt.video_path,
            opt.annotation_path,
            num_segments = opt.num_segments,
            modality = opt.modality,
            transform = spatial_transform)
    elif opt.dataset == 'something':
        validation_data = Something(
            opt.video_path,
            opt.annotation_path,
            'validation',                                                                                                                            
            opt.n_val_samples,                                                                                                                       
            spatial_transform,                                                                                                                       
            temporal_transform,                                                                                                                      
            target_transform,                                                                                                                        
            sample_duration=opt.sample_duration)  
    return validation_data


def get_test_set(opt, spatial_transform, temporal_transform, target_transform):
    assert opt.dataset in ['kinetics', 'activitynet', 'ucf101', 'hmdb51','something','fire']
    assert opt.test_subset in ['val', 'test']

    if opt.test_subset == 'val':
        subset = 'validation'
    elif opt.test_subset == 'test':
        subset = 'testing'
    if opt.dataset == 'kinetics':
        test_data = Kinetics(
            opt.video_path,
            opt.annotation_path,
            subset,
            0,
            spatial_transform,
            temporal_transform,
            target_transform,
            sample_duration=opt.sample_duration,
            sample_stride=opt.sample_stride)
    elif opt.dataset == 'activitynet':
        test_data = ActivityNet(
            opt.video_path,
            opt.annotation_path,
            subset,
            0,
            spatial_transform,
            temporal_transform,
            target_transform,
            sample_duration=opt.sample_duration)
    elif opt.dataset == 'ucf101':
        opt.annotation_path = opt.annotation_path + "/test_rgb_ucf101.txt"
        test_data = UCF101(
            opt.video_path,
            opt.annotation_path,
            num_segments = opt.num_segments,
            modality = opt.modality,
            transform = spatial_transform,
            test_mode = True)
    elif opt.dataset == 'hmdb51':
        test_data = HMDB51(
            opt.video_path,
            opt.annotation_path,
            subset,
            0,
            spatial_transform,
            temporal_transform,
            target_transform,
            sample_duration=opt.sample_duration)
    elif opt.dataset == 'something':                                                                                                                    
        test_data = Something(                                                                                                                          
            opt.video_path,                                                                                                                          
            opt.annotation_path,                                                                                                                     
            subset,                                                                                                                                  
            0,                                                                                                                                       
            spatial_transform,                                                                                                                       
            temporal_transform,                                                                                                                      
            target_transform,                                                                                                                        
            sample_duration=opt.sample_duration)
    elif opt.dataset == 'fire':
        opt.annotation_path = "/DATACENTER2/wxy/workspace/senet-3d/datasets/txt/test_fire.txt"
        test_data = FIRE(
            opt.video_path,
            opt.annotation_path,
            num_segments = opt.num_segments,
            modality = opt.modality,
            transform = spatial_transform,
            test_mode=True,
            test_idx=opt.test_idx)

    return test_data
