import torch
import helper.io_utils as iot

def save_model(uidx,prms,mmodule,mm_opt):
    # save the model
    mpath = iot.get_wd()+'/models/model_{0}.pth'.format(uidx)
    print('Saving model {0}'.format(mpath))
    save_dic={}
    save_dic['prms']=prms
    save_dic['model'] = mmodule
    save_dic['optim'] = mm_opt
    torch.save(save_dic,mpath)