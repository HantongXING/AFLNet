import torch
from torch import nn
# from selfsup.models.modules import SimCLRProjectionHead
from selfsup.heads import SimCLRProjectionHead
import time
from selfsup.optimizer_choice import load_optimizer
from selfsup.ntx_ent_loss import NTXentLoss

class SimCLR(nn.Module):
    def __init__(self, backbone,out_feature):
        super().__init__()
        self.backbone = backbone
        self.projection_head = SimCLRProjectionHead(out_feature, 512, 128)

    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(x)
        return x,z

def Simclr_train(args,backbone,out_feature, dataloader):

    model = SimCLR(backbone,out_feature)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model.to(device)

    criterion = NTXentLoss() # 损失函数，每种方法对应不同的损失函数

    optimizer, scheduler = load_optimizer(args, model)
    print("Starting Training")
    for epoch in range(args.max_epochs):
        total_loss = 0
        for i, ((x0, x1), _)  in enumerate(dataloader):
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
        # 保存模型， 设置每隔100次就保存一次
        if (epoch+1) % 1 == 0 :
            # 聚类/分类 + 模型名称 + 最大训练次数 + 优化器名称 + 数据集名称 + bs + 信噪比 + .pth
            checkpoint_name = '{}_{}_{}_{}_{}.pth'.format(args.model, str(epoch+1), args.optimizer, args.dataset_name,args.batch_size)
            torch.save(model.state_dict(), '{}/{}'.format('checkpoints/simclr', checkpoint_name))
        if scheduler:
            scheduler.step()  

    # 保存最终的模型 
    checkpoint_name = '{}_{}_{}_{}_{}_final.pth'.format(args.model, str(epoch+1), args.optimizer,args.dataset_name,args.batch_size)
    print('参数保存在：',checkpoint_name)
    torch.save(model.state_dict(), '{}/{}'.format('checkpoints/simclr', checkpoint_name))
    