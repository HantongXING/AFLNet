# Note: The model and training settings do not follow the reference settings
# from the paper. The settings are chosen such that the example can easily be
# run on a small dataset with a single GPU.

import torch
from torch import nn
import time
from selfsup.loss import DCLLoss, DCLWLoss
from selfsup.heads import SimCLRProjectionHead
from selfsup.optimizer_choice import load_optimizer

class DCL(nn.Module):
    def __init__(self, backbone,out_feature):
        super().__init__()
        self.backbone = backbone
        self.projection_head = SimCLRProjectionHead(out_feature, 512, 128)

    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(x)
        return x, z

def Dcl_train(args,backbone,out_feature,dataloader):
    ##########-------------------- 写一个记录文件 -------------------##########
    # log = open('log/log.txt', 'a') # 写一个记录文件
    # log.write('-'*30+time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))+'-'*30+'\n') # 写入运行时间
    # log.write('method:{}\ntrain_path:{}\nnum_epoch:{}\nlearning_rate:{}\noptimizer:{}\n'.format(
    #         args.method, args.train_path, args.max_epochs, args.lr, args.optimizer)) # 记录参数 
    # 模型
    model = DCL(backbone,out_feature)
    # model.load_state_dict(torch.load('/media/hp3090/HDD-2T/WX/RML_selfsup/checkpoints/dcl/classifier_cnn_final_SGD_RML2016_10a_110_6dB_SNR.pth'))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    criterion = DCLLoss()
    # or use the weighted DCLW loss:
    # criterion = DCLWLoss()

    #optimizer = torch.optim.SGD(model.parameters(), lr=0.06)
    optimizer, scheduler = load_optimizer(args, model)
    print("Starting Training")

    for epoch in range(args.max_epochs):
        total_loss = 0
        for i, ((x0, x1), _) in enumerate(dataloader):
            x0 = x0.to(device)
            x1 = x1.to(device)
            _, z0 = model(x0)
            _, z1 = model(x1)
            loss = criterion(z0, z1)
            total_loss += loss.detach()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            print('Epoch [%d/%d], iter %d, Loss: %.4f' %(epoch+1, args.max_epochs,  i+1, loss))
        avg_loss = total_loss / len(dataloader)
        print('Epoch [%d/%d], Val_Loss: %.4f' %(epoch+1, args.max_epochs, avg_loss))
        # log.write('Epoch [%d/%d], Val_Loss: %.4f\n' %(epoch+1, args.max_epochs, avg_loss))
        # 保存模型， 设置每隔100次就保存一次
        if (epoch+1) % 1 == 0 :
            # 聚类/分类 + 模型名称 + 最大训练次数 + 优化器名称 + 数据集名称 + bs + 信噪比 + .pth
            checkpoint_name = '{}_{}_{}_{}_{}.pth'.format(args.model, str(epoch+1), args.optimizer,args.dataset_name,args.batch_size)
            torch.save(model.state_dict(), '{}/{}'.format('checkpoints/dcl', checkpoint_name))
        if scheduler:
            scheduler.step()  

    # 保存最终的模型 
    checkpoint_name = '{}_{}_{}_{}_{}_final.pth'.format(args.model, str(epoch+1), args.optimizer,args.dataset_name,args.batch_size)
    print('参数保存在：',checkpoint_name)
    torch.save(model.state_dict(), '{}/{}'.format('checkpoints/dcl', checkpoint_name))
    
    # log.write('-'*40+"End of Train"+'-'*40+'\n')
    # log.close()

