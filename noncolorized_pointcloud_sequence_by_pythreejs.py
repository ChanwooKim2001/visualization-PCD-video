import numpy as np
from pythreejs import *
import time
import copy
import math
from PIL import Image
import time
import cv2
from collections import namedtuple
from IPython.display import display

# File paths
calib_cam_to_cam_path="2011_09_26/calib_cam_to_cam.txt"
calib_velo_to_cam_path="2011_09_26/calib_velo_to_cam.txt"

# # File paths
base_lidar_path = "2011_09_26/2011_09_26_drive_0001_extract/velodyne_points/data/"

#Point cloud list
lidar_data_paths = [f"{base_lidar_path}{i:010d}.txt" for i in range(0, 111)]
lidar_data_list = []
for path in lidar_data_paths:
    lidar_data = np.loadtxt(path, dtype=np.float32).reshape(-1, 4)
    lidar_data=lidar_data[:,:3]
    lidar_data_list.append(lidar_data)

lidar_timestamp_path="2011_09_26/2011_09_26_drive_0001_extract/velodyne_points/timestamps.txt"
lidar_timestamp=np.loadtxt(lidar_timestamp_path, dtype=np.string_)
lidar_timestamp=lidar_timestamp.T[1]
time_seconds = [float(time.decode('utf-8').split(':')[-1]) for time in lidar_timestamp]
time_seconds=np.array(time_seconds)-time_seconds[0]

# Read calibration data
with open(calib_cam_to_cam_path, 'r') as fid:
    lines = fid.readlines()
    for i in range(len(lines)):
        lines[i] = lines[i].strip().split(' ')
        del lines[i][0]
        lines[i] = ' '.join(lines[i])
    S_00=np.fromstring(lines[0+2], dtype=float, sep=' ')
    K_00=np.fromstring(lines[1+2], dtype=float, sep=' ')
    D_00=np.fromstring(lines[2+2], dtype=float, sep=' ')
    R_00=np.fromstring(lines[3+2], dtype=float, sep=' ')
    T_00=np.fromstring(lines[4+2], dtype=float, sep=' ')
    S_rect_00=np.fromstring(lines[5+2], dtype=float, sep=' ')
    R_rect_00=np.fromstring(lines[6+2], dtype=float, sep=' ')
    P_rect_00=np.fromstring(lines[7+2], dtype=float, sep=' ')

    S_02=np.fromstring(lines[16+2], dtype=float, sep=' ')
    K_02=np.fromstring(lines[17+2], dtype=float, sep=' ')
    D_02=np.fromstring(lines[18+2], dtype=float, sep=' ')
    R_02=np.fromstring(lines[19+2], dtype=float, sep=' ')
    T_02=np.fromstring(lines[20+2], dtype=float, sep=' ')
    S_rect_02=np.fromstring(lines[21+2], dtype=float, sep=' ')
    R_rect_02=np.fromstring(lines[22+2], dtype=float, sep=' ')
    P_rect_02=np.fromstring(lines[23+2], dtype=float, sep=' ')

with open(calib_velo_to_cam_path, 'r') as f:
    lines=f.readlines()
    for i in range(len(lines)):
        lines[i]=lines[i].strip().split(' ')
        del lines[i][0]
        lines[i]=' '.join(lines[i])
    R_velo_to_cam=np.fromstring(lines[0+1], dtype=float, sep=' ')
    T_velo_to_cam=np.fromstring(lines[1+1], dtype=float, sep=' ')

# Perform LiDAR-to-Image mapping
R0_rect=R_rect_00.reshape(3,3)
R_velo_to_cam=R_velo_to_cam.reshape(3,3)
t_velo_to_cam=T_velo_to_cam.reshape(3,1)
Rt_velo_to_cam=np.hstack((R_velo_to_cam,t_velo_to_cam))
Tr_velo_to_cam=Rt_velo_to_cam.reshape(3,4)
p2_rect=P_rect_02.reshape(3,4)
k2,r2,t2,_,_,_,_= cv2.decomposeProjectionMatrix(p2_rect)
t2=t2[:3,0]

def lidar_to_cam2(lidar_data):
    #delete unuseful reflactance information
    lidar_xyz=lidar_data.T

    #from lidar coordinate to cam2 coordinate
    lidar_xyz1 = np.vstack((lidar_xyz, np.ones(lidar_data.T[0].shape)))
    world_xyz1_unrect=np.dot(Tr_velo_to_cam, lidar_xyz1)
    world_xyz1_rect=np.dot(R0_rect,world_xyz1_unrect)
    world_xyz1_rect=np.vstack((world_xyz1_rect, np.ones(world_xyz1_rect[0].shape)))
    cam2_xy1=np.dot(p2_rect,world_xyz1_rect)
    return cam2_xy1

def cam2_to_world(cam2_xy1):
    world_xy1=np.dot(np.linalg.inv(k2),cam2_xy1.T)
    world_xy1=world_xy1.T-t2
    world_xy1=np.dot(np.linalg.inv(r2),world_xy1.T)
    world_xy1=world_xy1.T
    return world_xy1

def world_to_lidar(world_xy1):
    lidar_xyz1=np.dot(np.linalg.inv(R0_rect),world_xy1.T)
    lidar_xyz1=lidar_xyz1-t_velo_to_cam
    lidar_xyz1=np.dot(np.linalg.inv(R_velo_to_cam),lidar_xyz1)
    return lidar_xyz1

def get_xy_index(image_color, lidar_data):
    xy=get_xy(lidar_data)
    xy_index_max_1=np.where(xy[:,0]<image_color.shape[1]-1)
    xy_index_min_1=np.where(xy[:,0]>=0)
    xy_index_max_2=np.where(xy[:,1]<image_color.shape[0]-1)
    xy_index_min_2=np.where(xy[:,1]>=0)
    xy_index1=np.intersect1d(xy_index_max_1,xy_index_min_1)
    xy_index2=np.intersect1d(xy_index_max_2,xy_index_min_2)

    #xy_index: points rocated in the range of camera's FOV
    xy_index=np.intersect1d(xy_index1,xy_index2)
    return xy_index

def resizing_xy(image_color,lidar_data):
    xy=get_xy(lidar_data)
    reshaped_xy=xy[get_xy_index(image_color,lidar_data)]
    return reshaped_xy

def get_valid_xy1_index(lidar_data):
    cam2_xy1=lidar_to_cam2(lidar_data)
    weight=cam2_xy1[2,:]
    #x=cam2_xy1[0,:]/weight
    #y=cam2_xy1[1,:]/weight
    k = np.where(weight > 0)[0]
    return k

def get_xy(lidar_data):
    cam2_xy1=lidar_to_cam2(lidar_data)
    k=get_valid_xy1_index(lidar_data)
    weight=cam2_xy1[2,:]
    x=cam2_xy1[0,:]/weight
    y=cam2_xy1[1,:]/weight
    x=x[k].reshape(-1,1)
    y=y[k].reshape(-1,1)
    w=weight[k].reshape(-1,1)
    xy=np.hstack((x,y))
    return xy

def get_rgb_color(image_color,lidar_data):
    #concatenate rgb indices and values
    rgb=image_color.reshape(-1,3)
    rgb_index=[]
    for index, value in np.ndenumerate(rgb.T[0]):
        rgb_index.append(index)
    rgb=np.hstack((rgb_index,rgb))

    # clip rgb array according to condition which is contained in the range of camera's FOV.
    reshaped_xy=resizing_xy(image_color,lidar_data)
    int_xy=reshaped_xy.astype(int)
    rgb_color=np.zeros((len(int_xy.T[0]),3))
    for i in range(len(int_xy.T[0])):
        index=int_xy[i,1]*1392+int_xy[i,0]
        rgb_color[i]=rgb[index,1:4]

    return rgb_color

def build_PointCloud(xyz, rgb=None, show_distance=None, size=0.03, height=960, width=960):
    pts = copy.deepcopy(xyz)
    if rgb is None:
        rgb = 0.5 * np.ones_like(pts)

    if show_distance is not None:
        dist = [dist] if not isinstance(show_distance, (tuple, list)) else show_distance

        angles = np.linspace(1, 360, 360 * 10)
        xx = np.array([d * np.cos(angles) for d in dist]).flatten()
        yy = np.array([d * np.sin(angles) for d in dist]).flatten()
        zz = np.zeros_like(xx)

        cc = np.zeros((len(xx), 3))
        cc[:, 0] = 1.0

        pts = np.concatenate([pts, np.stack([xx, yy, zz], axis=1)], axis=0)
        rgb = np.concatenate([rgb, cc], axis=0)

    points_buf = BufferAttribute(array=pts)
    points_buf.needsUpdate=True
    colors_buf = BufferAttribute(array=rgb)
    geometryAttrs = {'position': points_buf, 'color': colors_buf}
    geometry = BufferGeometry(attributes=geometryAttrs)

    material = PointsMaterial(vertexColors='VertexColors', size=size)
    pointCloud = Points(geometry=geometry, material=material)
    return pointCloud

def build_point_geometry(lidar_data):
    rgb = np.ones_like(lidar_data)
    rgb[:,0]=rgb[:,0]*0
    rgb[:,1]=rgb[:,1]*0
    points_buf = BufferAttribute(array=lidar_data)
    colors_buf = BufferAttribute(array=rgb)
    geometryAttrs = {'position': points_buf, 'color': colors_buf}
    geometry = BufferGeometry(attributes=geometryAttrs)
    return geometry

#you can use this function when you only want to display just a frame of pointCloud data
def visualize_3d(xyz, rgb=None, show_distance=None, size=0.03, height=960, width=960):
    pts = copy.deepcopy(xyz)
    if rgb is None:
        rgb = 0.5 * np.ones_like(pts)

    if show_distance is not None:
        dist = [dist] if not isinstance(show_distance, (tuple, list)) else show_distance

        angles = np.linspace(1, 360, 360 * 10)
        xx = np.array([d * np.cos(angles) for d in dist]).flatten()
        yy = np.array([d * np.sin(angles) for d in dist]).flatten()
        zz = np.zeros_like(xx)

        cc = np.zeros((len(xx), 3))
        cc[:, 0] = 1.0

        pts = np.concatenate([pts, np.stack([xx, yy, zz], axis=1)], axis=0)
        rgb = np.concatenate([rgb, cc], axis=0)

    points_buf = BufferAttribute(array=pts)
    colors_buf = BufferAttribute(array=rgb)
    geometryAttrs = {'position': points_buf, 'color': colors_buf}
    geometry = BufferGeometry(attributes=geometryAttrs)

    material = PointsMaterial(vertexColors='VertexColors', size=size)
    pointCloud = Points(geometry=geometry, material=material)

    pythreejs_camera = PerspectiveCamera(
        up=[0, 0, 1],
        children=[DirectionalLight(color='white', intensity=0.5)])

    pythreejs_camera.position = (-15., 0., 3.)

    scene = Scene(children=[
                    pointCloud,
                    pythreejs_camera,
                    AmbientLight(color='#777777')], background=None)


    cam_position=[0,0,0]
    cam_position=np.dot(np.linalg.inv(k2),cam_position)
    cam_position=cam_position-t2
    cam_position=np.dot(np.linalg.inv(r2),cam_position)

    near_condition=np.where(pts[:,0]>=0)
    near=np.min(depth_camera02[near_condition,0])
    far=np.max(pts[:,0])

    Fov= 2*math.atan(p2_rect[1,1]/p2_rect[0,0])*(180/math.pi)
    cam=PerspectiveCamera(up=[0,0,-1], position=(cam_position[0],cam_position[1],cam_position[2]), aspect=370/1224, fov=Fov, near=near, far=far)
    camera_helper = CameraHelper(cam)
    camera_helper.rotateY(-math.pi/2)

    axes = AxesHelper(size=3)
    scene.add(axes)
    scene.add(camera_helper)
    control = OrbitControls(controlling=pythreejs_camera)
    renderer = Renderer(camera=pythreejs_camera,
                        scene=scene,
                        width=width,
                        height=height,
                        preserveDrawingBuffer=True,
                        controls=[control])

    return renderer

# Per dataformat.txt
OxtsPacket = namedtuple('OxtsPacket',
                        'lat, lon, alt, ' +
                        'roll, pitch, yaw, ' +
                        'vn, ve, vf, vl, vu, ' +
                        'ax, ay, az, af, al, au, ' +
                        'wx, wy, wz, wf, wl, wu, ' +
                        'pos_accuracy, vel_accuracy, ' +
                        'navstat, numsats, ' +
                        'posmode, velmode, orimode')

# Bundle into an easy-to-access structure
OxtsData = namedtuple('OxtsData', 'packet, T_w_imu')


def subselect_files(files, indices):
    try:
        files = [files[i] for i in indices]
    except:
        pass
    return files


def rotx(t):
    """Rotation about the x-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[1,  0,  0],
                     [0,  c, -s],
                     [0,  s,  c]])


def roty(t):
    """Rotation about the y-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c,  0,  s],
                     [0,  1,  0],
                     [-s, 0,  c]])


def rotz(t):
    """Rotation about the z-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s,  0],
                     [s,  c,  0],
                     [0,  0,  1]])


def transform_from_rot_trans(R, t):
    """Transforation matrix from rotation matrix and translation vector."""
    R = R.reshape(3, 3)
    t = t.reshape(3, 1)
    return np.vstack((np.hstack([R, t]), [0, 0, 0, 1]))


def read_calib_file(filepath):
    """Read in a calibration file and parse into a dictionary."""
    data = {}

    with open(filepath, 'r') as f:
        for line in f.readlines():
            try:
                key, value = line.split(':', 1)
            except ValueError:
                key, value = line.split(' ', 1)
            # The only non-float values in these files are dates, which
            # we don't care about anyway
            try:
                data[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                pass

    return data


def pose_from_oxts_packet(packet, scale):
    """Helper method to compute a SE(3) pose matrix from an OXTS packet.
    """
    er = 6378137.  # earth radius (approx.) in meters

    # Use a Mercator projection to get the translation vector
    tx = scale * packet.lon * np.pi * er / 180.
    ty = scale * er * \
        np.log(np.tan((90. + packet.lat) * np.pi / 360.))
    tz = packet.alt
    t = np.array([tx, ty, tz])

    # Use the Euler angles to get the rotation matrix
    Rx = rotx(packet.roll)
    Ry = roty(packet.pitch)
    Rz = rotz(packet.yaw)
    R = Rz.dot(Ry.dot(Rx))

    # Combine the translation and rotation into a homogeneous transform
    return R, t


def load_oxts_packets_and_poses(oxts_files):
    """Generator to read OXTS ground truth data.

       Poses are given in an East-North-Up coordinate system
       whose origin is the first GPS position.
    """
    # Scale for Mercator projection (from first lat value)
    scale = None
    # Origin of the global coordinate system (first GPS position)
    origin = None

    oxts = []

    for filename in oxts_files:
        with open(filename, 'r') as f:
            for line in f.readlines():
                line = line.split()
                # Last five entries are flags and counts
                line[:-5] = [float(x) for x in line[:-5]]
                line[-5:] = [int(float(x)) for x in line[-5:]]

                packet = OxtsPacket(*line)

                if scale is None:
                    scale = np.cos(packet.lat * np.pi / 180.)

                R, t = pose_from_oxts_packet(packet, scale)

                if origin is None:
                    origin = t

                T_w_imu = transform_from_rot_trans(R, t - origin)

                oxts.append(OxtsData(packet, T_w_imu))

    return oxts


def load_image(file, mode):
    """Load an image from file."""
    return Image.open(file).convert(mode)


def yield_images(imfiles, mode):
    """Generator to read image files."""
    for file in imfiles:
        yield load_image(file, mode)


def load_velo_scan(file):
    """Load and parse a velodyne binary file."""
    scan = np.fromfile(file, dtype=np.float32)
    return scan.reshape((-1, 4))


def yield_velo_scans(velo_files):
    """Generator to parse velodyne binary files into arrays."""
    for file in velo_files:
        yield load_velo_scan(file)

def animate_frame(frame,pointCloud):
    # 새로운 위치 데이터 생성
    new_position = lidar_data_list[frame]

    # 새로운 위치 데이터를 기존 PointCloud 객체에 할당
    pointCloud.geometry.attributes['position'].array =new_position
    pointCloud.geometry.attributes['position'].needsUpdate = True
    renderer.render(scene, renderer.camera)

    # 잠시 대기 (프레임 간격 조절)
    if frame != 0:
        time.sleep(time_seconds[frame] - time_seconds[frame-1])
    return pointCloud

def animate_pointCloud_frame_tr(frame, pointCloud):
    # generate new position data
    if frame==0:
        new_position = lidar_data_list[frame]
        pointCloud.geometry.attributes['position'].array =new_position
        pointCloud.geometry.attributes['position'].needsUpdate = True
        renderer.render(scene, renderer.camera)

    else:
        new_position = lidar_data_list[frame]
        new_position=np.vstack((new_position.T, np.ones(new_position.shape[0])))
        for i in range(frame):
            new_position=np.dot(transformation_list[i],new_position)
        new_position=new_position.T[:,:3]

        pointCloud.geometry.attributes['position'].array =new_position
        pointCloud.geometry.attributes['position'].needsUpdate = True
        renderer.render(scene, renderer.camera)

    #frame term
    if frame != 0:
        time.sleep(time_seconds[frame] - time_seconds[frame-1])

    return pointCloud

material = PointsMaterial(vertexColors='VertexColors', size=0.01)

pythreejs_camera = PerspectiveCamera(
        up=[0, 0, 1],
        children=[DirectionalLight(color='white', intensity=0.5)])
pythreejs_camera.rotateX(np.pi/4)
pythreejs_camera.position = (-15., 0., 50.)
scene = Scene(children=[
                    pythreejs_camera,
                    AmbientLight(color='#777777')], background=None)

axes = AxesHelper(size=3)
scene.add(axes)

control = OrbitControls(controlling=pythreejs_camera)
renderer = Renderer(camera=pythreejs_camera,
                        scene=scene,
                        width=800,
                        height=600,
                        preserveDrawingBuffer=True,
                        controls=[control])
display(renderer)

# 애니메이션 함수
pointCloud=build_PointCloud(lidar_data_list[0])
scene.add(pointCloud)

# 애니메이션 루프
for frame in range(100):
    pointCloud=animate_frame(frame,pointCloud)
scene.remove(pointCloud)

# File paths
base_oxts_path = "2011_09_26/2011_09_26_drive_0001_extract/oxts/data/"

#Point cloud list
oxts_data_paths = [f"{base_oxts_path}{i:010d}.txt" for i in range(0, 1165)]
oxts_data_list = []
for path in oxts_data_paths:
    oxts_data = np.loadtxt(path, dtype=np.float32)
    oxts_data_list.append(oxts_data)

metadata = load_oxts_packets_and_poses(oxts_data_paths)
T_w_imu_list=[]
for i in range(0,1165,10):
    T_w_imu=metadata[i][1]
    T_w_imu_list.append(T_w_imu)

transformation_list=[]
for i in range(115):
    tr=np.linalg.inv(T_w_imu_list[i]).dot(T_w_imu_list[i+1])
    transformation_list.append(tr)
transformation_list=np.array(transformation_list)

pointCloud=build_PointCloud(lidar_data_list[0])
scene.add(pointCloud)
display(renderer)

for frame in range(1,110):
    pointCloud=animate_pointCloud_frame_tr(frame, pointCloud)
scene.remove(pointCloud)