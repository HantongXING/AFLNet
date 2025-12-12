# Note: The model and training settings do not follow the reference settings
# from the paper. The settings are chosen such that the example can easily be
# run on a small dataset with a single GPU.

import torch
from torch import nn
import time
import copy
from selfsup.loss import NTXentLoss
from selfsup.heads import MoCoProjectionHead
from selfsup.model.utils import deactivate_requires_grad
from selfsup.model.utils import update_momentum
from selfsup.optimizer_choice import load_optimizer

class MoCo(nn.Module):
    def __init__(self, backbone,out_feature):
        super().__init__()

        self.backbone = backbone
        self.projection_head = MoCoProjectionHead(out_feature, 512, 128) #考虑改成128，64

        self.backbone_momentum = copy.deepcopy(self.backbone)
        self.projection_head_momentum = copy.deepcopy(self.projection_head)

        deactivate_requires_grad(self.backbone_momentum)
        deactivate_requires_grad(self.projection_head_momentum)

    def forward(self, x):
        query_f = self.backbone(x).flatten(start_dim=1)
        query = self.projection_head(query_f)
        return query_f, query

    def forward_momentum(self, x):
        key = self.backbone_momentum(x).flatten(start_dim=1)
        key = self.projection_head_momentum(key).detach()
        return key


def Moco_train(args,backbone,out_feature,dataloader):

    ##########-------------------- 写一个记录文件 -------------------##########
    # log = open('log/log.txt', 'a') # 写一个记录文件
    # log.write('-'*30+time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))+'-'*30+'\n') # 写入运行时间
    # log.write('method:{}\ntrain_path:{}\nnum_epoch:{}\nlearning_rate:{}\noptimizer:{}\n'.format(
    #         args.method, args.train_path, args.max_epochs, args.lr, args.optimizer)) # 记录参数 
    # # 模型
    model = MoCo(backbone, out_feature)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    criterion = NTXentLoss(memory_bank_size=4096) #损失函数
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.06)
    optimizer, scheduler = load_optimizer(args, model)

    print("Moco Starting Training")
    for epoch in range(args.max_epochs):
        total_loss = 0
        for i,((x_query, x_key), _) in enumerate(dataloader):
            update_momentum(model.backbone, model.backbone_momentum, m=0.99)
            update_momentum(model.projection_head, model.projection_head_momentum, m=0.99)
            x_query = x_query.to(device)
            x_key = x_key.to(device)
            _, query = model(x_query)
            key = model.forward_momentum(x_key)
            loss = criterion(query, key)
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
            torch.save(model.state_dict(), '{}/{}'.format('checkpoints/moco', checkpoint_name))
        if scheduler:
            scheduler.step() 

    # 保存最终的模型 
    checkpoint_name = '{}_{}_{}_{}_{}_final.pth'.format(args.model, str(epoch+1), args.optimizer,args.dataset_name,args.batch_size)
    print('参数保存在：',checkpoint_name)
    torch.save(model.state_dict(), '{}/{}'.format('checkpoints/moco', checkpoint_name))
    
    # log.write('-'*40+"End of Train"+'-'*40+'\n')
    # log.close()