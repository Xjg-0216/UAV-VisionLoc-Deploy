import socket
import time
from ctypes import Structure, c_ubyte, c_uint, c_float, sizeof, memmove, addressof
import threading
import cv2

class UAVSystem:
    def __init__(self, ip, port, camera_index):
        # 初始化网络和摄像头
        self.jbSocket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.jbSocket.bind((ip, port))

        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise ValueError("无法打开摄像头")

        self.g_air = AAIR()

    def process_attitude_data(self, air):
        """处理并打印姿态数据"""
        print("接收到的姿态数据:")
        print(f"  start0: {hex(air.start0)}")
        print(f"  start1: {hex(air.start1)}")
        print(f"  length: {air.length}")
        print(f"  id: {air.id}")
        print(f"  time: {air.time}")
        print(f"  actime: {air.actime}")
        print(f"  lat: {air.lat}")
        print(f"  lng: {air.lng}")
        print(f"  height: {air.height}")
        print(f"  yaw: {air.yaw}")
        print(f"  pitch: {air.pitch}")
        print(f"  roll: {air.roll}")
        print(f"  angle: {air.angle}")
        print(f"  crc: {hex(air.crc)}")

    def capture_image(self):
        """捕获图像，保存并返回图像数据"""
        ret, frame = self.cap.read()
        if ret:
            timestamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
            filename = f"capture_{timestamp}.jpg"
            cv2.imwrite(filename, frame)
            print(f"图像已保存为 {filename}")
            return frame  # 返回捕获的图像数据
        else:
            print("无法从摄像头读取数据")
            return None

    def udp_fun(self):
        while True:
            try:
                # 接收姿态数据
                recv_data, _ = self.jbSocket.recvfrom(sizeof(AAIR))
                air = AAIR()
                memmove(addressof(air), recv_data, sizeof(AAIR))

                # 校验数据包
                if air.start0 != 0x55 or air.start1 != 0xAA or air.crc != 0xFF:
                    print("接收到的数据包错误:", air.start0, air.start1, air.length, air.id, air.time, air.crc)
                    continue

                self.g_air = air
                self.process_attitude_data(self.g_air)
                captured_frame = self.capture_image()

                # 如果需要，可以在此处对返回的图像进行进一步处理
                if captured_frame is not None:
                    # 进一步处理图像的代码
                    pass

            except Exception as e:
                print(f"数据处理时出错: {e}")

    def start(self):
        # 启动接收姿态数据的线程
        rec_trd = threading.Thread(target=self.udp_fun)
        rec_trd.start()

class AAIR(Structure):
    _pack_ = 1  # 让结构体内存连续
    _fields_ = [
        ("start0", c_ubyte),  # 0x55
        ("start1", c_ubyte),  # 0xAA
        ("length", c_ubyte),  # 数据长度,41个字节
        ("id", c_ubyte),  # 报文ID,151
        ("time", c_uint),  # 位置采样时间
        ("actime", c_uint),  # 飞机相机同步时间
        ("lat", c_float),  # 目标纬度
        ("lng", c_float),  # 目标经度
        ("height", c_float),  # 目标高度
        ("yaw", c_float),  # Yaw
        ("pitch", c_float),  # Pitch
        ("roll", c_float),  # Roll
        ("angle", c_float),  # 航向角
        ("crc", c_ubyte)  # 包校验值固定为0xFF
    ]

if __name__ == "__main__":
    # 替换为实际的IP、端口和摄像头设备索引
    uav_system = UAVSystem(ip="192.168.1.19", port=16300, camera_index=81)
    uav_system.start()
