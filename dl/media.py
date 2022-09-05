import cv2

def get_frame(video_path,index):
    cap = cv2.VideoCapture(video_path)  #返回一个capture对象
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)  #设置要获取的帧号
    ret, frame = cap.read()  #read方法返回一个布尔值和一个视频帧。若帧读取成功，则返回True
    assert ret,f'无法读取视频{video_path}的第{index}帧'
    return frame