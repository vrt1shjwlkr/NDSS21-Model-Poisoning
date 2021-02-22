from femnist_models import *

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.xavier_uniform_(m.weight)
    elif classname.find('Linear') != -1:
        torch.nn.init.xavier_uniform_(m.weight)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.fill_(0)
        m.bias.data.fill_(0)

def save_checkpoint_user_(user_num, state, is_best, checkpoint=None, filename='checkpoint.pth.tar', best_filename=None):
    assert (checkpoint != None), 'Error: No checkpoint path provided!'

    if not os.path.isdir(checkpoint):
        mkdir_p(checkpoint)

    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    
    if not best_filename:
        best_filename='user_%d_model_best.pth.tar'%(user_num)
    
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, best_filename))

        
def save_checkpoint_user(user_num, state, is_best, checkpoint=None, filename='checkpoint.pth.tar', best_filename=None):
    assert (checkpoint != None), 'Error: No checkpoint path provided!'

    if not os.path.isdir(checkpoint):
        mkdir_p(checkpoint)

    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    
    if not best_filename:
        best_filename='user_%d_model_best.pth.tar'%(user_num)
    else:
        if best_filename=='user_%d_distil_best_model.pth.tar'%user_num:
            shutil.copyfile(os.path.join(checkpoint, 'user_%d_model_best.pth.tar'%(user_num)), os.path.join(checkpoint, best_filename))

    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, best_filename))
        

def save_checkpoint_global(state, is_best, checkpoint=None, filename='checkpoint.pth.tar', best_filename='model_best.pth.tar'):
    assert (checkpoint != None), 'Error: No checkpoint path provided!'

    if not os.path.isdir(checkpoint):
        mkdir_p(checkpoint)

    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, best_filename))


def write_csv(csv_file_path,row,header=False):
    if header:
        with open(csv_file_path, 'w') as f:
            f_w = csv.writer(f)
            f_w.writerow(row)
    else:
        with open(csv_file_path, 'a') as f:
            f_w = csv.writer(f)
            f_w.writerow(row)

def update_model(agg_grads, fed_model, optimizer_fed, epoch_num, schedule, gamma):
    start_idx=0
    if epoch_num in schedule:
        for param_group in optimizer_fed.param_groups:
            param_group['lr'] *= gamma
            print('epoch %d new fed lr %.4f' % (epoch_num, param_group['lr']))

    optimizer_fed.zero_grad()

    model_grads=[]

    for i, param in enumerate(fed_model.parameters()):
        param_=agg_grads[start_idx:start_idx+len(param.data.view(-1))].reshape(param.data.shape)
        start_idx=start_idx+len(param.data.view(-1))
        param_=param_.cuda()
        model_grads.append(param_)

    optimizer_fed.step(model_grads)
    
    return fed_model, optimizer_fed

def get_model(fed_model, optimizer_fed, chkpt, fed_file, epoch_num):
    fed_checkpoint=chkpt+'/'+fed_file
    assert os.path.isfile(fed_checkpoint), 'Error: no user checkpoint at %s'%(fed_checkpoint)
    checkpoint = torch.load(fed_checkpoint, map_location='cuda:%d'%torch.cuda.current_device())
    fed_model.load_state_dict(checkpoint['state_dict'])
    optimizer_fed.load_state_dict(checkpoint['optimizer'])
    epoch_num+=checkpoint['epoch']
    best_global_acc = checkpoint['best_global_acc']
    best_global_te_acc = checkpoint['best_global_te_acc']
    
    return fed_model, optimizer_fed, epoch_num, best_global_acc, best_global_te_acc
