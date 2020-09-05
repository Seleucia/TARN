import sys
# reload(sys)
# sys.setdefaultencoding('utf-8')
import h5py

#checking saved files.
hdf_file='/media/hc/Data/all/PycharmProjects/TARN/data/c3d_features_19.hdf5'


with h5py.File(hdf_file, "r") as f:
    # List all groups
    print("Keys: %s" % f.keys())
    a_group_key_lst = list(f.keys())



    # Get the data
    for a_group_key in a_group_key_lst:
        c3d_features= f[a_group_key]['c3d_features']
        print(c3d_features.value.shape)
        print ( 'Done.')
