import os

# location of boggart repository
BOGGART_REPO_PATH = "/home/kth/rva/boggart"
main_dir = f"{BOGGART_REPO_PATH}/data/"

assert os.path.exists(BOGGART_REPO_PATH), "Update Boggart Repository Path in configs.py"
assert main_dir is not None, "Set main_dir to <REPO PATH>/boggart/data/"

video_directory = f"{main_dir}{{vid_label}}{{hour}}/"

frames_dir = f"{{video_dir}}frames/"

trajectories_dir = f"{{video_dir}}trajectories/"
background_dir = f"{{video_dir}}backgrounds/"

kps_loc_dir = f"{{video_dir}}kps_locs/{{traj_info}}/"
kps_matches_dir = f"{{video_dir}}kps_matches/{{traj_info}}/"
kps_raw_dir = f"{{video_dir}}raw_kps/"

obj_dets_dir = f"{{video_dir}}object_detection_results/{{model}}/"
obj_dets_csv = f"{{obj_dets_dir}}{{name}}{{hour}}.csv"

query_results_dir = f"{{video_dir}}query_results/{{query_type}}/"
boggart_results_dir = f"{{video_dir}}boggart_results/{{query_type}}/"

pipeline_results_dir = f"{main_dir}pipeline_results/"

video_files_dir = f"{{video_dir}}video/"

crops = {
    # this is what you DON'T WANT!
    # vname : {class_id : [x1, y1, x2, y2]} where x1, y1 top left
    "auburn_first_angle" : {
        "person" : [0, 0, 1920, 400],
        "car" : [0, 0, 1920, 500],
        "bicycle" : [0, 0, 1920, 500],
        "truck" : [0, 0, 1920, 500]
    },
    "auburn_first_angle_base" : {
        "person" : [0, 0, 1920, 400],
        "car" : [0, 0, 1920, 500],
        "bicycle" : [0, 0, 1920, 500],
        "truck" : [0, 0, 1920, 500]
    },
    "auburn_crf23_live" : {
        "person" : [0, 0, 1920, 400],
        "car" : [0, 0, 1920, 500],
        "bicycle" : [0, 0, 1920, 500],
        "truck" : [0, 0, 1920, 500]
    },
    "auburn_crf32_live" : {
        "person" : [0, 0, 1920, 400],
        "car" : [0, 0, 1920, 500],
        "bicycle" : [0, 0, 1920, 500],
        "truck" : [0, 0, 1920, 500]
    },
    "auburn_crf37_live" : {
        "person" : [0, 0, 1920, 400],
        "car" : [0, 0, 1920, 500],
        "bicycle" : [0, 0, 1920, 500],
        "truck" : [0, 0, 1920, 500]
    },
    "auburn_crf42_live" : {
        "person" : [0, 0, 1920, 400],
        "car" : [0, 0, 1920, 500],
        "bicycle" : [0, 0, 1920, 500],
        "truck" : [0, 0, 1920, 500]
    },
    "auburn_crf47_live" : {
        "person" : [0, 0, 1920, 400],
        "car" : [0, 0, 1920, 500],
        "bicycle" : [0, 0, 1920, 500],
        "truck" : [0, 0, 1920, 500]
    },
    "auburn_1000k" : {
        "person" : [0, 0, 1920, 400],
        "car" : [0, 0, 1920, 500],
        "bicycle" : [0, 0, 1920, 500],
        "truck" : [0, 0, 1920, 500]
    },
    "auburn_1500k" : {
        "person" : [0, 0, 1920, 400],
        "car" : [0, 0, 1920, 500],
        "bicycle" : [0, 0, 1920, 500],
        "truck" : [0, 0, 1920, 500]
    },
    "auburn_500k" : {
        "person" : [0, 0, 1920, 400],
        "car" : [0, 0, 1920, 500],
        "bicycle" : [0, 0, 1920, 500],
        "truck" : [0, 0, 1920, 500]
    },
    "auburn_120k" : {
        "person" : [0, 0, 1920, 400],
        "car" : [0, 0, 1920, 500],
        "bicycle" : [0, 0, 1920, 500],
        "truck" : [0, 0, 1920, 500]
    },
    "auburn_90k" : {
        "person" : [0, 0, 1920, 400],
        "car" : [0, 0, 1920, 500],
        "bicycle" : [0, 0, 1920, 500],
        "truck" : [0, 0, 1920, 500]
    },
    "auburn_60k" : {
        "person" : [0, 0, 1920, 400],
        "car" : [0, 0, 1920, 500],
        "bicycle" : [0, 0, 1920, 500],
        "truck" : [0, 0, 1920, 500]
    },
    "auburn_100k" : {
        "person" : [0, 0, 1920, 400],
        "car" : [0, 0, 1920, 500],
        "bicycle" : [0, 0, 1920, 500],
        "truck" : [0, 0, 1920, 500]
    },
    "auburn_34k" : {
        "person" : [0, 0, 1920, 400],
        "car" : [0, 0, 1920, 500],
        "bicycle" : [0, 0, 1920, 500],
        "truck" : [0, 0, 1920, 500]
    },
    "auburn_200k" : {
        "person" : [0, 0, 1920, 400],
        "car" : [0, 0, 1920, 500],
        "bicycle" : [0, 0, 1920, 500],
        "truck" : [0, 0, 1920, 500]
    },
    "auburn_272k" : {
        "person" : [0, 0, 1920, 400],
        "car" : [0, 0, 1920, 500],
        "bicycle" : [0, 0, 1920, 500],
        "truck" : [0, 0, 1920, 500]
    },
    "auburn_5k" : {
        "person" : [0, 0, 1920, 400],
        "car" : [0, 0, 1920, 500],
        "bicycle" : [0, 0, 1920, 500],
        "truck" : [0, 0, 1920, 500]
    },
    "auburn_150k" : {
        "person" : [0, 0, 1920, 400],
        "car" : [0, 0, 1920, 500],
        "bicycle" : [0, 0, 1920, 500],
        "truck" : [0, 0, 1920, 500]
    },
    "mfs250k" : {
        "person" : [0, 0, 1920, 400],
        "car" : [0, 0, 1920, 500],
        "bicycle" : [0, 0, 1920, 500],
        "truck" : [0, 0, 1920, 500]
    },
    "mfs144k" : {
        "person" : [0, 0, 1920, 400],
        "car" : [0, 0, 1920, 500],
        "bicycle" : [0, 0, 1920, 500],
        "truck" : [0, 0, 1920, 500]
    },
    "mfs144k_ultra_max" : {
        "person" : [0, 0, 1920, 400],
        "car" : [0, 0, 1920, 500],
        "bicycle" : [0, 0, 1920, 500],
        "truck" : [0, 0, 1920, 500]
    },
    "mfs108k" : {
        "person" : [0, 0, 1920, 400],
        "car" : [0, 0, 1920, 500],
        "bicycle" : [0, 0, 1920, 500],
        "truck" : [0, 0, 1920, 500]
    },
    "mfs108k_ultra_max" : {
        "person" : [0, 0, 1920, 400],
        "car" : [0, 0, 1920, 500],
        "bicycle" : [0, 0, 1920, 500],
        "truck" : [0, 0, 1920, 500]
    },
    "mfs314k_ultra_max" : {
        "person" : [0, 0, 1920, 400],
        "car" : [0, 0, 1920, 500],
        "bicycle" : [0, 0, 1920, 500],
        "truck" : [0, 0, 1920, 500]
    },
    "mfs250k_ultra_max" : {
        "person" : [0, 0, 1920, 400],
        "car" : [0, 0, 1920, 500],
        "bicycle" : [0, 0, 1920, 500],
        "truck" : [0, 0, 1920, 500]
    },
    "mfs500k_ultra_max" : {
        "person" : [0, 0, 1920, 400],
        "car" : [0, 0, 1920, 500],
        "bicycle" : [0, 0, 1920, 500],
        "truck" : [0, 0, 1920, 500]
    },
    "mfs1620k_ultra_max" : {
        "person" : [0, 0, 1920, 400],
        "car" : [0, 0, 1920, 500],
        "bicycle" : [0, 0, 1920, 500],
        "truck" : [0, 0, 1920, 500]
    },
    "mfs74k" : {
        "person" : [0, 0, 1920, 400],
        "car" : [0, 0, 1920, 500],
        "bicycle" : [0, 0, 1920, 500],
        "truck" : [0, 0, 1920, 500]
    },
    "mfs74k_ultra_max" : {
        "person" : [0, 0, 1920, 400],
        "car" : [0, 0, 1920, 500],
        "bicycle" : [0, 0, 1920, 500],
        "truck" : [0, 0, 1920, 500]
    },
    "mfs800k_ultra_max" : {
        "person" : [0, 0, 1920, 400],
        "car" : [0, 0, 1920, 500],
        "bicycle" : [0, 0, 1920, 500],
        "truck" : [0, 0, 1920, 500]
    },
    "mfs74k_ultra" : {
        "person" : [0, 0, 1920, 400],
        "car" : [0, 0, 1920, 500],
        "bicycle" : [0, 0, 1920, 500],
        "truck" : [0, 0, 1920, 500]
    },

    "jackson_hole_wy" : {
        "person" : [0, 0, 1920, 500],
        "car" : [0, 0, 1920, 500],
        "bicycle" : [0, 0, 1920, 500],
        "truck" : [0, 0, 1920, 500]
    },
    "lausanne_pont_bassieres" : {
        "person" : [0, 0, 0, 0],
        "car" : [0, 0, 0, 0],
        "bicycle" : [0, 0, 0, 0],
        "truck" : [0, 0, 0, 0]
    },
    "lausanne_crf23_pont_bassieres" : {
        "person" : [0, 0, 0, 0],
        "car" : [0, 0, 0, 0],
        "bicycle" : [0, 0, 0, 0],
        "truck" : [0, 0, 0, 0]
    },
    "lausanne_crf37_pont_bassieres" : {
        "person" : [0, 0, 0, 0],
        "car" : [0, 0, 0, 0],
        "bicycle" : [0, 0, 0, 0],
        "truck" : [0, 0, 0, 0]
    },
}

frame_bounds = {
    "auburn_first_angle": [1080, 1920],
    "auburn_first_angle_base": [1080, 1920],
    "auburn_crf23_live": [1080, 1920],
    "auburn_crf32_live": [1080, 1920],
    "auburn_crf37_live": [1080, 1920],
    "auburn_crf42_live": [1080, 1920],
    "auburn_crf47_live": [1080, 1920],
    "auburn_1000k": [1080, 1920],
    "auburn_500k": [1080, 1920],
    "auburn_100k": [1080, 1920],
    "auburn_200k": [1080, 1920],
    "auburn_120k": [1080, 1920],
    "auburn_272k": [1080, 1920],
    "auburn_90k": [1080, 1920],
    "auburn_60k": [1080, 1920],
    "auburn_150k": [1080, 1920],
    "auburn_34k": [1080, 1920],
    "mfs74k_ultra": [1080, 1920],
    "mfs108k_ultra_max": [1080, 1920],
    "mfs74k_ultra_max": [1080, 1920],
    "mfs144k_ultra_max": [1080, 1920],
    "mfs250k_ultra_max": [1080, 1920],
    "mfs314k_ultra_max": [1080, 1920],
    "mfs500k_ultra_max": [1080, 1920],
    "mfs800k_ultra_max": [1080, 1920],
    "mfs1620k_ultra_max": [1080, 1920],
    "auburn_5k": [1080, 1920],
    "auburn_1500k": [1080, 1920],
    "mfs250k": [1080, 1920],
    "mfs144k": [1080, 1920],
    "mfs108k": [1080, 1920],
    "mfs74k": [1080, 1920],
    "jackson_hole_wy": [1080, 1920],
    "lausanne_pont_bassieres": [720, 1280],
    "lausanne_crf23_pont_bassieres": [720, 1280],
}

class BackgroundConfig:
    def __init__(self, peak_thresh):
        self.peak_thresh : int = peak_thresh
        self.sample_rate : int = 30
        self.box_length : int = 2
        self.quant : int = 16
        self.bg_dur : int = 1800

    def get_bg_dir(self, video_dir):
        return background_dir.format(video_dir=video_dir)

    def get_bg_start(self, chunk_start):
        return chunk_start//self.bg_dur * self.bg_dur

    def get_base_bg_fname(self, video_dir, bg_start):
        return f"{self.get_bg_dir(video_dir)}bg_{bg_start}.pkl"

    def get_proper_bg_fname(self, video_dir, bg_start):
        return f"{self.get_bg_dir(video_dir)}bg_{bg_start}_{self.peak_thresh}.pkl"

class TrajectoryConfig:
    def __init__(self, diff_thresh, chunk_size, fps=30):
        self.fps : int = fps
        self.diff_thresh : int = diff_thresh
        self.blur_amt = 15
        self.chunk_size = chunk_size

        assert 0 < self.chunk_size <= 1800
        assert 0 < self.fps <= 30

        self.kps_loc_template = None
        self.kps_matches_template = None
        # self.kps_raw_template = None

    def get_traj_dir(self, video_dir):
        return trajectories_dir.format(video_dir=video_dir)

    def get_traj_fname(self, video_dir, chunk_start, bg_config):
        return f"{trajectories_dir.format(video_dir=video_dir)}{chunk_start}_{bg_config.peak_thresh}_{self.fps}_{self.diff_thresh}_{self.chunk_size}.csv"

    def get_kps_loc_template(self, video_dir, bg_peak_thresh):
        if self.kps_loc_template is None:
            local_dir = kps_loc_dir.format(video_dir=video_dir, traj_info=f"{bg_peak_thresh}_{self.fps}_{self.diff_thresh}_{self.chunk_size}")
            os.makedirs(local_dir, exist_ok=True)
            self.kps_loc_template = f"{local_dir}{{frame_no}}.pkl"
        return self.kps_loc_template

    def get_kps_matches_template(self, video_dir, bg_peak_thresh):
        if self.kps_matches_template is None:
            local_dir = kps_matches_dir.format(video_dir=video_dir, traj_info=f"{bg_peak_thresh}_{self.fps}_{self.diff_thresh}_{self.chunk_size}")
            os.makedirs(local_dir, exist_ok=True)
            self.kps_matches_template = f"{local_dir}{{frame_no}}.pkl"
        return self.kps_matches_template
