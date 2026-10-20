'''
File Created: Monday, 25th November 2019 1:35:30 pm
Author: Dave Zhenyu Chen (zhenyu.chen@tum.de)
'''

import os
import time
import numpy as np
from torch.utils.data import Dataset
from lib.config import CONF
from utils.pc_utils import random_sampling, rotx, roty, rotz
from data.scannet.model_util_scannet import ScannetDatasetConfig, rotate_aligned_boxes_along_axis
import random

DC = ScannetDatasetConfig()
MAX_NUM_OBJ = 128
MEAN_COLOR_RGB = np.array([109.8, 97.2, 83.8])

SCANNET_V2_TSV = os.path.join(CONF.PATH.SCANNET_META, "scannetv2-labels.combined.tsv")

class ScannetReferenceDataset(Dataset):
       
    def __init__(self, args, scanrefer, scanrefer_new, scanrefer_all_scene,
        split="train",
        num_points=40000,
        lang_num_max=32,
        use_height=False,
        use_color=False,
        use_normal=False,
        augment=False,
        shuffle=False):

        self.args = args
        self.scanrefer = scanrefer
        self.scanrefer_new = scanrefer_new
        self.scanrefer_new_len = len(scanrefer_new)
        self.scanrefer_all_scene = scanrefer_all_scene # all scene_ids in scanrefer
        self.split = split
        self.num_points = num_points
        self.use_color = use_color
        self.use_height = use_height
        self.use_normal = use_normal
        self.augment = augment
        self.lang_num_max = lang_num_max

        self._load_data()
        self.should_shuffle = shuffle

    def __len__(self):
        return self.scanrefer_new_len

    def split_scene_new(self,  scanrefer_data):
        scanrefer_train_new = []
        scanrefer_train_new_scene, scanrefer_train_scene = [], []
        scene_id = ''
        lang_num_max = self.lang_num_max
        for data in scanrefer_data:
            if scene_id != data["scene_id"]:
                scene_id = data["scene_id"]
                if len(scanrefer_train_scene) > 0:
                    if self.should_shuffle:
                        random.shuffle(scanrefer_train_scene)
                    for new_data in scanrefer_train_scene:
                        if len(scanrefer_train_new_scene) >= lang_num_max:
                            scanrefer_train_new.append(scanrefer_train_new_scene)
                            scanrefer_train_new_scene = []
                        scanrefer_train_new_scene.append(new_data)
                    if len(scanrefer_train_new_scene) > 0:
                        scanrefer_train_new.append(scanrefer_train_new_scene)
                        scanrefer_train_new_scene = []
                    scanrefer_train_scene = []
            scanrefer_train_scene.append(data)
        if len(scanrefer_train_scene) > 0:
            if self.should_shuffle:
                random.shuffle(scanrefer_train_scene)
            for new_data in scanrefer_train_scene:
                if len(scanrefer_train_new_scene) >= lang_num_max:
                    scanrefer_train_new.append(scanrefer_train_new_scene)
                    scanrefer_train_new_scene = []
                scanrefer_train_new_scene.append(new_data)
            if len(scanrefer_train_new_scene) > 0:
                scanrefer_train_new.append(scanrefer_train_new_scene)
                scanrefer_train_new_scene = []
        return scanrefer_train_new


    def shuffle_data(self):
        print('\nshuffle dataset data(lang)', flush=True)
        self.scanrefer_new = self.split_scene_new(self.scanrefer)
        if self.should_shuffle:
            random.shuffle(self.scanrefer_new)
        assert len(self.scanrefer_new) == self.scanrefer_new_len, 'assert scanrefer length right'
        print('shuffle done\n', flush=True)


    def __getitem__(self, idx):
        start = time.time()

        lang_num = len(self.scanrefer_new[idx])
        scene_id = self.scanrefer_new[idx][0]["scene_id"]

        object_id_list = []
        object_name_list = []
        ann_id_list = []

        for i in range(self.lang_num_max):
            if i < lang_num:
                object_id = int(self.scanrefer_new[idx][i]["object_id"])
                object_name = " ".join(self.scanrefer_new[idx][i]["object_name"].split("_"))
                ann_id = self.scanrefer_new[idx][i]["ann_id"]

            # Past lang_num the parent repeats the last annotation, so the padded slots
            # carry a valid object rather than a null one; the losses mask them out.
            object_id_list.append(object_id)
            object_name_list.append(object_name)
            ann_id_list.append(ann_id)

        mesh_vertices = self.scene_data[scene_id]["mesh_vertices"]
        instance_labels = self.scene_data[scene_id]["instance_labels"]
        semantic_labels = self.scene_data[scene_id]["semantic_labels"]
        instance_bboxes = self.scene_data[scene_id]["instance_bboxes"]

        if not self.use_color:
            point_cloud = mesh_vertices[:,0:3] # do not use color for now
            pcl_color = mesh_vertices[:,3:6]
        else:
            point_cloud = mesh_vertices[:,0:6]
            point_cloud[:,3:6] = (point_cloud[:,3:6]-MEAN_COLOR_RGB)/256.0
            pcl_color = point_cloud[:,3:6]

        if self.use_normal:
            normals = mesh_vertices[:,6:9]
            point_cloud = np.concatenate([point_cloud, normals],1)


        if self.use_height:
            floor_height = np.percentile(point_cloud[:,2],0.99)
            height = point_cloud[:,2] - floor_height
            point_cloud = np.concatenate([point_cloud, np.expand_dims(height, 1)],1)

        point_cloud, choices = random_sampling(point_cloud, self.num_points, return_choices=True)
        instance_labels = instance_labels[choices]
        semantic_labels = semantic_labels[choices]
        pcl_color = pcl_color[choices]

        target_bboxes = np.zeros((MAX_NUM_OBJ, 6))
        target_bboxes_mask = np.zeros((MAX_NUM_OBJ))
        angle_classes = np.zeros((MAX_NUM_OBJ,))
        angle_residuals = np.zeros((MAX_NUM_OBJ,))
        size_classes = np.zeros((MAX_NUM_OBJ,))
        size_residuals = np.zeros((MAX_NUM_OBJ, 3))
        size_gts = np.zeros((MAX_NUM_OBJ, 3))
        
        ref_box_label_list = []
        ref_center_label_list = []
        ref_heading_class_label_list = []
        ref_heading_residual_label_list = []
        ref_size_class_label_list = []
        ref_size_residual_label_list = []
        ref_size_gt_label_list = []

        if self.split != "test":
            num_bbox = instance_bboxes.shape[0] if instance_bboxes.shape[0] < MAX_NUM_OBJ else MAX_NUM_OBJ
            target_bboxes_mask[0:num_bbox] = 1
            target_bboxes[0:num_bbox,:] = instance_bboxes[:MAX_NUM_OBJ,0:6]

            point_votes = np.zeros([self.num_points, 3])
            point_votes_mask = np.zeros(self.num_points)

            if self.augment:
                if np.random.random() > 0.7:
                    # Flipping along the YZ plane
                    point_cloud[:, 0] = -1 * point_cloud[:, 0]
                    target_bboxes[:, 0] = -1 * target_bboxes[:, 0]

                if np.random.random() > 0.7:
                    # Flipping along the XZ plane
                    point_cloud[:, 1] = -1 * point_cloud[:, 1]
                    target_bboxes[:, 1] = -1 * target_bboxes[:, 1]

                # Rotation along X-axis
                rot_angle = (np.random.random() * np.pi / 18) - np.pi / 36  # -5 ~ +5 degree
                rot_mat = rotx(rot_angle)
                point_cloud[:, 0:3] = np.dot(point_cloud[:, 0:3], np.transpose(rot_mat))
                target_bboxes = rotate_aligned_boxes_along_axis(target_bboxes, rot_mat, "x")

                # Rotation along Y-axis
                rot_angle = (np.random.random() * np.pi / 18) - np.pi / 36  # -5 ~ +5 degree
                rot_mat = roty(rot_angle)
                point_cloud[:, 0:3] = np.dot(point_cloud[:, 0:3], np.transpose(rot_mat))
                target_bboxes = rotate_aligned_boxes_along_axis(target_bboxes, rot_mat, "y")

                # Rotation along up-axis/Z-axis
                rot_angle = (np.random.random() * np.pi / 18) - np.pi / 36  # -5 ~ +5 degree
                rot_mat = rotz(rot_angle)
                point_cloud[:, 0:3] = np.dot(point_cloud[:, 0:3], np.transpose(rot_mat))
                target_bboxes = rotate_aligned_boxes_along_axis(target_bboxes, rot_mat, "z")

                scale = np.random.uniform(-0.1, 0.1, (3, 3))
                scale = np.exp(scale)
                scale = scale * np.eye(3)
                point_cloud[:, 0:3] = np.dot(point_cloud[:, 0:3], scale)
                if self.use_height:
                    point_cloud[:, 3] = point_cloud[:, 3] * float(scale[2, 2])
                target_bboxes[:, 0:3] = np.dot(target_bboxes[:, 0:3], scale)
                target_bboxes[:, 3:6] = np.dot(target_bboxes[:, 3:6], scale)

                point_cloud, target_bboxes = self._translate(point_cloud, target_bboxes)

            gt_centers = target_bboxes[:, 0:3]
            gt_centers[instance_bboxes.shape[0]:, :] += 1000.0  # padding centers with a large number so they never match a real point during nearest-center lookup below
            # Note: since there's no map between bbox instance labels and
            # pc instance_labels (it had been filtered
            # in the data preparation step) we'll compute the instance bbox
            # from the points sharing the same instance label.
            point_obj_mask = np.zeros(self.num_points)
            point_instance_label = np.zeros(self.num_points) - 1
            for i_instance in np.unique(instance_labels):
                ind = np.where(instance_labels == i_instance)[0]
                if semantic_labels[ind[0]] in DC.nyu40ids:
                    x = point_cloud[ind, :3]
                    center = 0.5 * (x.min(0) + x.max(0))
                    ilabel = np.argmin(((center - gt_centers) ** 2).sum(-1))
                    point_instance_label[ind] = ilabel
                    point_obj_mask[ind] = 1.0

            # Note: since there's no map between bbox instance labels and
            # pc instance_labels (it had been filtered
            # in the data preparation step) we'll compute the instance bbox
            # from the points sharing the same instance label.
            for i_instance in np.unique(instance_labels):
                ind = np.where(instance_labels == i_instance)[0]
                if semantic_labels[ind[0]] in DC.nyu40ids:
                    x = point_cloud[ind,:3]
                    center = 0.5*(x.min(0) + x.max(0))
                    point_votes[ind, :] = center - x
                    point_votes_mask[ind] = 1.0
            point_votes = np.tile(point_votes, (1, 3)) # make 3 votes identical

            class_ind = [DC.nyu40id2class[int(x)] for x in instance_bboxes[:num_bbox,-2]]
            # NOTE: set size class as semantic class. Consider use size2class.
            size_classes[0:num_bbox] = class_ind
            size_residuals[0:num_bbox, :] = target_bboxes[0:num_bbox, 3:6] - DC.mean_size_arr[class_ind,:]
            size_gts[0:instance_bboxes.shape[0], :] = target_bboxes[0:instance_bboxes.shape[0], 3:6]

            # construct the reference target label for each bbox
            for j in range(self.lang_num_max):
                ref_box_label = np.zeros(MAX_NUM_OBJ)
                for i, gt_id in enumerate(instance_bboxes[:num_bbox, -1]):
                    if gt_id == object_id_list[j]:
                        ref_box_label[i] = 1
                        ref_center_label = target_bboxes[i, 0:3]
                        ref_heading_class_label = angle_classes[i]
                        ref_heading_residual_label = angle_residuals[i]
                        ref_size_class_label = size_classes[i]
                        ref_size_residual_label = size_residuals[i]
                        ref_size_gt_label = size_gts[i]

                        ref_box_label_list.append(ref_box_label)
                        ref_center_label_list.append(ref_center_label)
                        ref_heading_class_label_list.append(ref_heading_class_label)
                        ref_heading_residual_label_list.append(ref_heading_residual_label)
                        ref_size_class_label_list.append(ref_size_class_label)
                        ref_size_residual_label_list.append(ref_size_residual_label)
                        ref_size_gt_label_list.append(ref_size_gt_label)
            
            if self.args.detection:
                    
                ref_box_label_list = []
                ref_center_label_list = []
                ref_heading_class_label_list = []
                ref_heading_residual_label_list = []
                ref_size_class_label_list = []
                ref_size_residual_label_list = []
                ref_size_gt_label_list = []

                i = 0
                ref_box_label[i] = 1
                ref_center_label = target_bboxes[i, 0:3]
                ref_heading_class_label = angle_classes[i]
                ref_heading_residual_label = angle_residuals[i]
                ref_size_class_label = size_classes[i]
                ref_size_residual_label = size_residuals[i]
                ref_size_gt_label = size_gts[i]

                ref_box_label_list.append(ref_box_label)
                ref_center_label_list.append(ref_center_label)
                ref_heading_class_label_list.append(ref_heading_class_label)
                ref_heading_residual_label_list.append(ref_heading_residual_label)
                ref_size_class_label_list.append(ref_size_class_label)
                ref_size_residual_label_list.append(ref_size_residual_label)
                ref_size_gt_label_list.append(ref_size_gt_label)

        else:
            num_bbox = 1
            point_votes = np.zeros([self.num_points, 9]) # make 3 votes identical
            point_votes_mask = np.zeros(self.num_points)
            point_obj_mask = np.zeros(self.num_points)
            point_instance_label = np.zeros(self.num_points) - 1

        target_bboxes_semcls = np.zeros((MAX_NUM_OBJ))
        try:
            target_bboxes_semcls[0:num_bbox] = [DC.nyu40id2class[int(x)] for x in instance_bboxes[:,-2][0:num_bbox]]
        except KeyError:
            pass

        object_cat_list = []
        for i in range(self.lang_num_max):
            object_cat = self.raw2label[object_name_list[i]] if object_name_list[i] in self.raw2label else 17
            object_cat_list.append(object_cat)

        istrain = 0
        if self.split == "train":
            istrain = 1

        data_dict = {}
        data_dict["point_clouds"] = point_cloud.astype(np.float32) # point cloud data including features

        data_dict["istrain"] = istrain
        data_dict["center_label"] = target_bboxes.astype(np.float32)[:,0:3] # (MAX_NUM_OBJ, 3) for GT box center XYZ
        data_dict["heading_class_label"] = angle_classes.astype(np.int64) # (MAX_NUM_OBJ,) with int values in 0,...,NUM_HEADING_BIN-1
        data_dict["heading_residual_label"] = angle_residuals.astype(np.float32) # (MAX_NUM_OBJ,)
        data_dict["size_class_label"] = size_classes.astype(np.int64) # (MAX_NUM_OBJ,) with int values in 0,...,NUM_SIZE_CLUSTER
        data_dict["size_residual_label"] = size_residuals.astype(np.float32) # (MAX_NUM_OBJ, 3)
        data_dict['size_gts'] = size_gts.astype(np.float32)

        data_dict["num_bbox"] = np.array(num_bbox).astype(np.int64)
        data_dict["sem_cls_label"] = target_bboxes_semcls.astype(np.int64) # (MAX_NUM_OBJ,) semantic class index
        data_dict["box_label_mask"] = target_bboxes_mask.astype(np.float32) # (MAX_NUM_OBJ) as 0/1 with 1 indicating a unique box
        data_dict["vote_label"] = point_votes.astype(np.float32)
        data_dict["vote_label_mask"] = point_votes_mask.astype(np.int64)
        data_dict['point_obj_mask'] = point_obj_mask.astype(np.int64)
        data_dict['point_instance_label'] = point_instance_label.astype(np.int64)
        data_dict["scan_idx"] = np.array(idx).astype(np.int64)
        data_dict["pcl_color"] = pcl_color

        # "lang_num" is the number of REAL annotations in this scene's chunk. The name is
        # inherited; nothing linguistic survives in this dict.
        data_dict["lang_num"] = np.array(lang_num).astype(np.int64)
        data_dict["ref_box_label_list"] = np.array(ref_box_label_list).astype(np.int64)  # 0/1 reference labels for each object bbox
        data_dict["ref_center_label_list"] = np.array(ref_center_label_list).astype(np.float32)
        data_dict["ref_heading_class_label_list"] = np.array(ref_heading_class_label_list).astype(np.int64)
        data_dict["ref_heading_residual_label_list"] = np.array(ref_heading_residual_label_list).astype(np.int64)
        data_dict["ref_size_class_label_list"] = np.array(ref_size_class_label_list).astype(np.int64)
        data_dict["ref_size_residual_label_list"] = np.array(ref_size_residual_label_list).astype(np.float32)
        data_dict["object_id_list"] = np.array(object_id_list).astype(np.int64)
        data_dict["ann_id_list"] = np.array(ann_id_list).astype(np.int64)
        data_dict["object_cat_list"] = np.array(object_cat_list).astype(np.int64)

        unique_multiple_list = []
        for i in range(self.lang_num_max):
            object_id = object_id_list[i]
            ann_id = ann_id_list[i]
            unique_multiple = self.unique_multiple_lookup[scene_id][str(object_id)][ann_id]
            unique_multiple_list.append(unique_multiple)
        data_dict["unique_multiple_list"] = np.array(unique_multiple_list).astype(np.int64)

        data_dict["load_time"] = time.time() - start

        return data_dict
    
    def _get_raw2label(self):
        scannet_labels = DC.type2class.keys()
        scannet2label = {label: i for i, label in enumerate(scannet_labels)}

        lines = [line.rstrip() for line in open(SCANNET_V2_TSV)]
        lines = lines[1:]
        raw2label = {}
        for i in range(len(lines)):
            label_classes_set = set(scannet_labels)
            elements = lines[i].split('\t')
            raw_name = elements[1]
            nyu40_name = elements[7]
            if nyu40_name not in label_classes_set:
                raw2label[raw_name] = scannet2label['others']
            else:
                raw2label[raw_name] = scannet2label[nyu40_name]
        raw2label["shower_curtain"] = 13

        return raw2label

    def _get_unique_multiple_lookup(self):
        all_sem_labels = {}
        cache = {}
        for data in self.scanrefer:
            scene_id = data["scene_id"]
            object_id = data["object_id"]
            object_name = " ".join(data["object_name"].split("_"))
            ann_id = data["ann_id"]

            if scene_id not in all_sem_labels:
                all_sem_labels[scene_id] = []

            if scene_id not in cache:
                cache[scene_id] = {}

            if object_id not in cache[scene_id]:
                cache[scene_id][object_id] = {}
                try:
                    all_sem_labels[scene_id].append(self.raw2label[object_name])
                except KeyError:
                    all_sem_labels[scene_id].append(17)

        # convert to numpy array
        all_sem_labels = {scene_id: np.array(all_sem_labels[scene_id]) for scene_id in all_sem_labels.keys()}

        unique_multiple_lookup = {}
        for data in self.scanrefer:
            scene_id = data["scene_id"]
            object_id = data["object_id"]
            object_name = " ".join(data["object_name"].split("_"))
            ann_id = data["ann_id"]

            try:
                sem_label = self.raw2label[object_name]
            except KeyError:
                sem_label = 17

            unique_multiple = 0 if (all_sem_labels[scene_id] == sem_label).sum() == 1 else 1

            if scene_id not in unique_multiple_lookup:
                unique_multiple_lookup[scene_id] = {}

            if object_id not in unique_multiple_lookup[scene_id]:
                unique_multiple_lookup[scene_id][object_id] = {}

            if ann_id not in unique_multiple_lookup[scene_id][object_id]:
                unique_multiple_lookup[scene_id][object_id][ann_id] = None

            unique_multiple_lookup[scene_id][object_id][ann_id] = unique_multiple

        return unique_multiple_lookup

    def _load_data(self):
        self.scene_list = sorted(list(set([data["scene_id"] for data in self.scanrefer])))

        self.scene_data = {}
        for scene_id in self.scene_list:
            self.scene_data[scene_id] = {}
            self.scene_data[scene_id]["mesh_vertices"] = np.load(os.path.join(CONF.PATH.SCANNET_DATA, scene_id)+"_aligned_vert.npy") # axis-aligned
            self.scene_data[scene_id]["instance_labels"] = np.load(os.path.join(CONF.PATH.SCANNET_DATA, scene_id)+"_ins_label.npy")
            self.scene_data[scene_id]["semantic_labels"] = np.load(os.path.join(CONF.PATH.SCANNET_DATA, scene_id)+"_sem_label.npy")
            self.scene_data[scene_id]["instance_bboxes"] = np.load(os.path.join(CONF.PATH.SCANNET_DATA, scene_id)+"_aligned_bbox.npy")

        lines = [line.rstrip() for line in open(SCANNET_V2_TSV)]
        lines = lines[1:]
        raw2nyuid = {}
        for i in range(len(lines)):
            elements = lines[i].split('\t')
            raw_name = elements[1]
            nyu40_name = int(elements[4])
            raw2nyuid[raw_name] = nyu40_name

        self.raw2nyuid = raw2nyuid
        self.raw2label = self._get_raw2label()
        self.unique_multiple_lookup = self._get_unique_multiple_lookup()

    def _translate(self, point_set, bbox):
        coords = point_set[:, :3]

        x_factor = np.random.choice(np.arange(-0.5, 0.501, 0.001), size=1)[0]
        y_factor = np.random.choice(np.arange(-0.5, 0.501, 0.001), size=1)[0]
        z_factor = np.random.choice(np.arange(-0.5, 0.501, 0.001), size=1)[0]
        factor = [x_factor, y_factor, z_factor]

        coords += factor
        point_set[:, :3] = coords
        bbox[:, :3] += factor

        return point_set, bbox
