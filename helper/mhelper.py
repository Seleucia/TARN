import torch
import helper.io_utils as iot

def save_model(uidx,prms,mmodule,mm_opt,show_txt=True):
    # save the model
    mpath = iot.get_wd()+'/models/model_{0}.pth'.format(uidx)
    if show_txt==True:
        print('Saving model {0}'.format(mpath))
    save_dic={}
    save_dic['prms']=prms
    save_dic['model'] = mmodule
    save_dic['optim'] = mm_opt
    torch.save(save_dic,mpath)
    return mpath



def load_model(mpath):
    save_dic=torch.load(mpath)
    prms=    save_dic['prms']
    mmodule=save_dic['model']
    mm_opt=save_dic['optim']
    return prms,mmodule,mm_opt

