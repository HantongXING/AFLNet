# Note: The model and training settings do not follow the reference settings
# from the paper. The settings are chosen such that the example can easily be
# run on a small dataset with a single GPU.

import torch
from torch import nn

from selfsup.loss import NTXentLoss
from selfsup.heads import NNCLRProjectionHead
from selfsup.heads import NNCLRPredictionHead
from selfsup.nn_memory_bank import NNMemoryBankModule
from selfsup.optimizer_choice import load_optimizer

class NNCLR(nn.Module):
    def __init__(self, backbone,out_feature):
        super().__init__()

        self.backbone = backbone
        self.projection_head = NNCLRProjectionHead(out_feature, 512, 128)
        self.prediction_head = NNCLRPredictionHead(128, 512, 128)

    def forward(self, x):
        y = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(y)
        p = self.prediction_head(z)
        z = z.detach()
        return y, z, p

def Nnclr_train(args,backbone,out_feature,dataloader):

    model = NNCLR(backbone,out_feature)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    memory_bank = NNMemoryBankModule(size=4096)
    memory_bank.to(device)

    criterion = NTXentLoss()
    #optimizer = torch.optim.SGD(model.parameters(), lr=0.06)
    optimizer, scheduler = load_optimizer(args, model)

    print("Starting Training")
    for epoch in range(args.max_epochs):
        total_loss = 0
        for i, ((x0, x1), _) in enumerate(dataloader):
            x0 = x0.to(device)
            x1 = x1.to(device)
            _, z0, p0 = model(x0)
            _, z1, p1 = model(x1)
            z0 = memory_bank(z0, update=False)
            z1 = memory_bank(z1, update=True)
            loss = 0.5 * (criterion(z0, p1) + criterion(z1, p0))
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
            torch.save(model.state_dict(), '{}/{}'.format('checkpoints/nnclr', checkpoint_name))
        if scheduler:
            scheduler.step()  

    # 保存最终的模型 
    checkpoint_name = '{}_{}_{}_{}_{}_final.pth'.format(args.model, str(epoch+1), args.optimizer,args.dataset_name,args.batch_size)
    print('参数保存在：',checkpoint_name)
    torch.save(model.state_dict(), '{}/{}'.format('checkpoints/nnclr', checkpoint_name))
    