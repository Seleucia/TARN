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

def get_c3d_feats_hdf5():
    if socket.gethostname() == 'coskunh':
        hdf_file = '/media/hc/Data/all/PycharmProjects/TARN/data/cnt-2000_c3d_features.hdf5'
        # hdf_file = '/media/hc/Data/all/PycharmProjects/TARN/data/cnt-28472_c3d_features.hdf5'
    elif socket.gethostname() == 'wscskn':
        hdf_file = '/mnt/4tb/tpami/c3d_feats_hlyr-7_sb-2/cnt-28472_c3d_features.hdf5'
    return hdf_file

def get_hdf5_prot_vector():
    if socket.gethostname() == 'coskunh':
        hdf_file = get_wd()+'/data/feats_class_uidx-5000.pkl'
    elif socket.gethostname() == 'wscskn':
        hdf_file = get_wd()+'/data/feats_class_uidx-80000.pkl'
    return hdf_file
