import os
import shutil
import glob

def make_dir(dirpath):
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)


def remove_dir(dirpath):
    if os.path.exists(dirpath):
        shutil.rmtree(dirpath)



def get_weight_list(ckpt_path,choice=None):
    path_list = []
    for fold in os.scandir(ckpt_path):
        if choice is not None and eval(str(fold.name)[-1]) not in choice:
            continue
        if fold.is_dir():
            weight_path = os.listdir(fold.path)
            # print(weight_path)
            weight_path.sort(key=lambda x:int(x.split('-')[0].split(':')[-1]))
            path_list.append(os.path.join(fold.path,weight_path[-1]))
            # print(os.path.join(fold.path,weight_path[-1]))
    path_list.sort(key=lambda x:x.split('/')[-2])
    return path_list



def get_weight_path(ckpt_path):

    if os.path.isdir(ckpt_path):
        pth_list = os.listdir(ckpt_path)
        if len(pth_list) != 0:
            pth_list.sort(key=lambda x:int(x.split('-')[0].split(':')[-1]))
            return os.path.join(ckpt_path,pth_list[-1])
        else:
            return None
    else:
        return None



def remove_weight_path(ckpt_path,retain=1):

    if os.path.isdir(ckpt_path):
        pth_list = os.listdir(ckpt_path)
        if len(pth_list) >= retain:
            pth_list.sort(key=lambda x:int(x.split('-')[0].split(':')[-1]))
            for pth_item in pth_list[:-retain]:
                os.remove(os.path.join(ckpt_path,pth_item))


def dfs_remove_weight(ckpt_path,retain=1):
    for sub_path in os.scandir(ckpt_path):
        if sub_path.is_dir():
            dfs_remove_weight(sub_path.path,retain=retain)
        else:
            remove_weight_path(ckpt_path,retain=retain)
            break  


def calculate_flops(net,shape):
    """
    calculate the floating point operations of the model
    :param module: predict model
    :param sample: image tensor that needs to be predicted
    """
    import sys
    import time 
    from thop import profile
    import torch
    
    sample = torch.randn(shape)
    # print(sample.size())
    sys.stdout = open(os.devnull, 'w')
    runtime = time.clock()
    _ = net(sample)
    runtime = time.clock() - runtime
    flops, params = profile(net, inputs=sample.unsqueeze(0))
    sys.stdout = sys.__stdout__
    print(f'CALCULATION RESULT :\n\tFLOPs     : {int(flops)}\n\tPARAM NUM : {int(params)}\n\tRUN TIME  : {runtime} s')




# def output_model(net, shape, PATH, NAME):

#     from torch.utils.mobile_optimizer import optimize_for_mobile
#     import torch

#     class PredictModule(torch.nn.Module):
#         def __init__(self,net):
#             super(PredictModule, self).__init__()
#             # TODO import your final model, here take Mobile Net V3 Small as an example
#             self.final_model = net
#             self.final_model.eval()

#         def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
#             """
#             predict function
#             :param input_tensor: unprocessed picture tensor
#             :return: predicted classification number
#             """
#             x = input_tensor
#             y = self.final_model.forward(x)
#             return y.argmax()

#     sample = torch.randn(shape)

#     predict_module = PredictModule(net)

#     if not os.path.exists(PATH):
#         os.makedirs(PATH)
#     torch.save(predict_module, os.path.join(PATH, NAME+'.ptl'))
#     traced_script_module = torch.jit.trace(predict_module, (sample,))
#     traced_script_module_optimized = optimize_for_mobile(traced_script_module)
#     traced_script_module_optimized._save_for_lite_interpreter(os.path.join(PATH, NAME+'_mobile.ptl'))


if __name__ == "__main__":

    ckpt_path = './ckpt/'
    dfs_remove_weight(ckpt_path)