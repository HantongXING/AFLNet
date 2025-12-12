import copy
from torch import nn
import torch
import time

from selfsup.loss import NegativeCosineSimilarity
from selfsup.heads import BYOLProjectionHead, BYOLPredictionHead
from selfsup.model.utils import deactivate_requires_grad
from selfsup.model.utils import update_momentum
from selfsup.optimizer_choice import load_optimizer

class BYOL(nn.Module):
    def __init__(self, backbone,out_feature):
        super().__init__()

        self.backbone = backbone
        self.projection_head = BYOLProjectionHead(out_feature, 512, 256)
        self.prediction_head = BYOLPredictionHead(256, 1024, 256)

        self.backbone_momentum = copy.deepcopy(self.backbone)
        self.projection_head_momentum = copy.deepcopy(self.projection_head)

        deactivate_requires_grad(self.backbone_momentum)
        deactivate_requires_grad(self.projection_head_momentum)

    def forward(self, x):
        y = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(y)
        p = self.prediction_head(z)
        return y,p

    def forward_momentum(self, x):
        y = self.backbone_momentum(x).flatten(start_dim=1)
        z = self.projection_head_momentum(y)
        z = z.detach()
        return z

def Byol_train(args, backbone, out_feature, dataloader):

    # 模型
    model = BYOL(backbone,out_feature)
    # 在训练基础上继续训练
    # model.load_state_dict(torch.load('/media/hp3090/HDD-2T/WX/RML_selfsup/checkpoints/byol/classifier_cnn_final_SGD_RML2016_10a_110_6dB_SNR.pth'))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(model)
    ##########---------------------- 训练 -------------------##########
    criterion = NegativeCosineSimilarity() #损失计算
    #optimizer = torch.optim.SGD(model.parameters(), lr=args.lr) # 这里重新设置，暂时先不用修改
    optimizer, scheduler = load_optimizer(args, model)
    # 修改训练的次数
    print("Starting Training")
    for epoch in range(args.max_epochs):
        total_loss = 0
        for i, ((x0, x1), _) in enumerate(dataloader):
            update_momentum(model.backbone, model.backbone_momentum, m=0.99)
            update_momentum(model.projection_head, model.projection_head_momentum, m=0.99)
            x0 = x0.to(device)
            x1 = x1.to(device)
            _, p0 = model(x0)
            z0 = model.forward_momentum(x0)
            _, p1 = model(x1)
            z1 = model.forward_momentum(x1)
            loss = 0.5 * (criterion(p0, z1) + criterion(p1, z0)) # 他返回的就是负数，这个注意一下
            total_loss += loss.detach()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            print('Epoch [%d/%d], iter %d, Loss: %.4f' %(epoch+1, args.max_epochs,  i+1, loss))
        avg_loss = total_loss / len(dataloader)
        print(f"epoch: {epoch:>02}, loss: {avg_loss:.5f}")

        # 保存模型， 设置每隔100次就保存一次
        if (epoch+1) % 1 == 0 :
            # 聚类/分类 + 模型名称 + 最大训练次数 + 优化器名称 + 数据集名称 + bs + 信噪比 + .pth
            checkpoint_name = '{}_{}_{}_{}_{}.pth'.format(args.model, str(epoch+1), args.optimizer,args.dataset_name,args.batch_size)
            torch.save(model.state_dict(), '{}/{}'.format('checkpoints/byol', checkpoint_name))
        if scheduler:
            scheduler.step()  
    # 保存最终的模型 
    checkpoint_name = '{}_{}_{}_{}_{}_final.pth'.format(args.model, str(epoch+1), args.optimizer,args.dataset_name,args.batch_size)
    print('参数保存在：',checkpoint_name)
    torch.save(model.state_dict(), '{}/{}'.format('checkpoints/byol', checkpoint_name))
    