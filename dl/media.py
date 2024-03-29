import cv2


class Video():
    def __init__(self, video_path):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        assert self.cap is not None, f'Cannot open video: {video_path}'
        self.frame_nums = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.duration = self.frame_nums / self.fps

    def get_frame(self, index):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, index)  #设置要获取的帧号
        ret, frame = self.cap.read()  #read方法返回一个布尔值和一个视频帧。若帧读取成功，则返回True
        assert ret, f'can not read frame of {index} in {self.video_path}'
        return frame


def get_frame(video_path, index):
    cap = cv2.VideoCapture(video_path)  #返回一个capture对象
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)  #设置要获取的帧号
    ret, frame = cap.read()  #read方法返回一个布尔值和一个视频帧。若帧读取成功，则返回True
    assert ret, f'无法读取视频{video_path}的第{index}帧'
    return frame


def resize_imgs(base_img, other_imgs: list, method=cv2.INTER_AREA):
    """缩放图像列表至基准图像大小

    Args:
        base_img : 基准图像
        other_imgs : 待比较的图像列表
        method (optional): 缩放算法. Defaults to cv2.INTER_AREA.

    Returns:
        img_list: 处理后的图像列表，顺序不变
    """
    for i, img in enumerate(other_imgs):
        if not img.shape == base_img.shape:
            other_imgs[i] = cv2.resize(img,
                                       (base_img.shape[1], base_img.shape[0]),
                                       interpolation=method)
    return other_imgs


def get_imgs_diff_score(base_img, other_imgs: list):
    """获取图像列表和基准图片的相似度

    Args:
        base_img : 基准图像
        other_imgs : 待比较的图像列表

    Returns:
        similarity_list : 按顺序的图像相似度list
    """
    other_imgs = resize_imgs(base_img, other_imgs)
    similarity_list = []
    for img in other_imgs:
        errorL2 = cv2.norm(base_img, img, cv2.NORM_L2)
        similarity = 1 - errorL2 / (base_img.shape[0] * base_img.shape[1])
        similarity_list.append(similarity)
    return similarity_list  #从0到1，相似度升高，1代表完全相同


def get_video_frame_nums(video_path) -> int:
    cap = cv2.VideoCapture(video_path)
    return int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
