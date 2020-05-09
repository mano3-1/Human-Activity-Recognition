from posenet.base_model import BaseModel
import posenet


class PoseNet:

    def __init__(self, model: BaseModel, min_score=0.25):
        self.model = model
        self.min_score = min_score
    
    def get_heatmaps(self ,image):
        heat_maps ,_ ,__ ,___ ,______ = self.model.predict(image)
        return heat_maps 
    

    def estimate_multiple_poses(self, image, max_pose_detections=10):
        heatmap_result, offsets_result, displacement_fwd_result, displacement_bwd_result, image_scale = \
            self.model.predict(image)

        pose_scores, keypoint_scores, keypoint_coords = posenet.decode_multiple_poses(
            heatmap_result.numpy().squeeze(axis=0),
            offsets_result.numpy().squeeze(axis=0),
            displacement_fwd_result.numpy().squeeze(axis=0),
            displacement_bwd_result.numpy().squeeze(axis=0),
            output_stride=self.model.output_stride,
            max_pose_detections=max_pose_detections,
            min_pose_score=self.min_score)

        keypoint_coords *= image_scale

        return pose_scores, keypoint_scores, keypoint_coords

    def estimate_single_pose(self, image):
        return self.estimate_multiple_poses(image, max_pose_detections=1)

    def draw_poses(self, image, pose_scores, keypoint_scores, keypoint_coords):
        draw_image = posenet.draw_skel_and_kp(
            image, pose_scores, keypoint_scores, keypoint_coords,
            min_pose_score=self.min_score, min_part_score=self.min_score)

        return draw_image

    def print_scores(self, image_name, pose_scores, keypoint_scores, keypoint_coords):
        print()
        print("Results for image: %s" % image_name)
        for pi in range(len(pose_scores)):
            if pose_scores[pi] == 0.:
                break
            print('Pose #%d, score = %f' % (pi, pose_scores[pi]))
            for ki, (s, c) in enumerate(zip(keypoint_scores[pi, :], keypoint_coords[pi, :, :])):
                print('Keypoint %s, score = %f, coord = %s' % (posenet.PART_NAMES[ki], s, c))
