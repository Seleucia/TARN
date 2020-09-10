import os
import socket

def get_data_folder():
    data_folder=get_wd()+'/data'
    return data_folder


def get_wd():
    if socket.gethostname() == 'coskunh':
      data_folder='/media/hc/Data/all/PycharmProjects/TARN'
    elif socket.gethostname() == 'wscskn':
        data_folder = '/home/cskn/PycharmProjects/TARN/'
    return data_folder

def get_c3d_feats_hdf5(ds):

    if socket.gethostname() == 'coskunh':
        if ds =='epic':
            hdf_file = '/media/hc/Data/all/PycharmProjects/TARN/data/{0}/cnt-2000_c3d_features.hdf5'.format(ds)
        elif ds =='gaze':
            hdf_file = '/media/hc/Data/all/PycharmProjects/TARN/data/{0}/cnt-999_c3d_features.hdf5'.format(ds)
        # hdf_file = '/media/hc/Data/all/PycharmProjects/TARN/data/cnt-28472_c3d_features.hdf5'
    elif socket.gethostname() == 'wscskn':
        if ds == 'epic':
            hdf_file = '/mnt/4tb/tpami/{0}/c3d_feats_hlyr-7_sb-2/cnt-28472_c3d_features.hdf5'.format(ds)
        elif ds == 'gaze':
            hdf_file = '/mnt/4tb/tpami/{0}/c3d_feats_hlyr-7_sb-2/cnt-10321_c3d_features.hdf5'.format(ds)
    return hdf_file

