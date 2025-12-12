# Note: The model and training settings do not follow the reference settings
# from the paper. The settings are chosen such that the example can easily be
# run on a small dataset with a single GPU.

import torch
from torch import nn
import time
from lightly.models.modules import BarlowTwinsProjectionHead
from lightly.loss import BarlowTwinsLoss
from selfsup.optimizer_choice import load_optimizer

class BarlowTwins(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.projection_head = BarlowTwinsProjectionHead(75, 2048, 2048)

    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(x)
        return x, z

def BarlowTwins_train(args,backbone,dataloader):
    ##########-------------------- 写一个记录文件 -------------------##########
    log = open('log/log.txt', 'a') # 写一个记录文件
    log.write('-'*30+time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))+'-'*30+'\n') # 写入运行时间
    log.write('method:{}\ntrain_path:{}\nnum_epoch:{}\nlearning_rate:{}\noptimizer:{}\n'.format(
            args.method, args.train_path, args.max_epochs, args.lr, args.optimizer)) # 记录参数 
    # 模型
    model = BarlowTwins(backbone)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    criterion = BarlowTwinsLoss()
    #optimizer = torch.optim.SGD(model.parameters(), lr=0.06)
    optimizer, scheduler = load_optimizer(args, model)

    print("Starting Training")
    for epoch in range(10):
        total_loss = 0
        for (x0, x1), _ in dataloader:
            x0 = x0.to(device)
            x1 = x1.to(device)
            _, z0 = model(x0)
            _, z1 = model(x1)
            loss = criterion(z0, z1)
            total_loss += loss.detach()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        avg_loss = total_loss / len(dataloader)
        print('Epoch [%d/%d], Val_Loss: %.4f' %(epoch+1, args.max_epochs, avg_loss))
        log.write('Epoch [%d/%d], Val_Loss: %.4f\n' %(epoch+1, args.max_epochs, avg_loss))
        # 保存模型， 设置每隔100次就保存一次
        if (epoch+1) % args.log_freq == 0 :
            # 聚类/分类 + 模型名称 + 最大训练次数 + 优化器名称 + 数据集名称 + bs + 信噪比 + .pth
            checkpoint_name = '{}_{}_{}_{}_{}_{}_{}.pth'.format(args.fine_method, args.model, str(epoch+1), args.optimizer,args.dataset_name,args.batch_size,args.db)
            torch.save(model.state_dict(), '{}/{}'.format('checkpoints/balowtwins', checkpoint_name))
        if scheduler:
            scheduler.step()  

    # 保存最终的模型 
    checkpoint_name = '{}_{}_{}_{}_{}_{}_{}.pth'.format(args.fine_method, args.model, 'final', args.optimizer,args.dataset_name,args.batch_size,args.db)
    print('参数保存在：',checkpoint_name)
    torch.save(model.state_dict(), '{}/{}'.format('checkpoints/balowtwins', checkpoint_name))
    
    log.write('-'*40+"End of Train"+'-'*40+'\n')
    log.close()