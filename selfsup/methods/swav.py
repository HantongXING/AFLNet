# Note: The model and training settings do not follow the reference settings
# from the paper. The settings are chosen such that the example can easily be
# run on a small dataset with a single GPU.

import torch
from torch import nn
import time
from selfsup.loss import SwaVLoss
from selfsup.heads import SwaVProjectionHead
from selfsup.heads import SwaVPrototypes
from selfsup.optimizer_choice import load_optimizer

class SwaV(nn.Module):
    def __init__(self, backbone,out_feature):
        super().__init__()
        self.backbone = backbone
        self.projection_head = SwaVProjectionHead(out_feature, 512, 128)
        self.prototypes = SwaVPrototypes(128, n_prototypes=512)

    def forward(self, x):
        x_f = self.backbone(x).flatten(start_dim=1)
        x = self.projection_head(x_f)
        x = nn.functional.normalize(x, dim=1, p=2)
        p = self.prototypes(x)
        # return x_f,p
        return p

def Swav_train(args,backbone,out_feature, dataloader):

    model = SwaV(backbone,out_feature)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    model.to(device)

    criterion = SwaVLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    optimizer, scheduler = load_optimizer(args, model)
    print("Starting Training")
    for epoch in range(args.max_epochs):
        total_loss = 0
        for i,(batch, _) in enumerate(dataloader):
            model.prototypes.normalize()
            multi_crop_features = [model(x.to(device)) for x in batch]
            # _, multi_crop_features = [model(x.to(device)) for x in batch]
            high_resolution = multi_crop_features[:2]
            low_resolution = multi_crop_features[2:]
            loss = criterion(high_resolution, low_resolution)
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
            checkpoint_name = '{}_{}_{}_{}_{}.pth'.format(args.model, str(epoch+1), args.optimizer,args.dataset_name,args.batch_size)
            torch.save(model.state_dict(), '{}/{}'.format('checkpoints/swav', checkpoint_name))
        if scheduler:
            scheduler.step()  

    # 保存最终的模型 
    checkpoint_name = '{}_{}_{}_{}_{}_final.pth'.format(args.model, str(epoch+1), args.optimizer,args.dataset_name,args.batch_size)
    print('参数保存在：',checkpoint_name)
    torch.save(model.state_dict(), '{}/{}'.format('checkpoints/swav', checkpoint_name))
    