import torch
import torch.nn as nn
from torch.autograd import Variable
import argparse
from dataloader import get_loader
from model_distill import EncoderNet, DecoderNet, ClassNet, EPELoss
from torch.utils.tensorboard import SummaryWriter
import time
from torch.optim.lr_scheduler import ReduceLROnPlateau

parser = argparse.ArgumentParser(description='GeoNetM')
parser.add_argument('--epochs', type=int, default=50, metavar='N')
parser.add_argument('--reg', type=float, default=0.1, metavar='REG')
parser.add_argument('--lr', type=float, default=0.0001, metavar='LR')
parser.add_argument('--data_num', type=int, default=60000, metavar='N')
parser.add_argument('--batch_size', type=int, default=12, metavar='N')
parser.add_argument("--dataset_dir", type=str, default="E:\pycharm\pythonproject\learnpytorch\distortion correction\distortion_data")
parser.add_argument("--distortion_type", type=list,
                    default=['barrel','pincushion','shear','rotate','perspective','wave'])
parser.add_argument('--patience', type=int, default=7, help='早停法等待轮数')
parser.add_argument('--min_lr', type=float, default=1e-6, help='最小学习率')
parser.add_argument('--factor', type=float, default=0.5, help='学习率衰减因子')
args = parser.parse_args()

use_GPU = torch.cuda.is_available()

#训练函数
def train(train_loader,val_loader,epochs,optimizer,criterion,criterion_clas):
    # 初始化学习率调度器和早停相关变量
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=args.factor,
        patience=args.patience // 2,  # 调度器耐心值为早停的一半
        min_lr=args.min_lr
    )
    no_improve_counter = 0
    best_val_loss = float('inf')
    step = 0
    for epoch in range(epochs):
        start_time = time.time()
        previous_step_time = start_time
        # 训练模式
        model_en.train()
        model_de.train()
        model_class.train()
        print("--------第{}轮训练--------".format(epoch+1))
        for i, (disimgs, disx, disy, labels) in enumerate(train_loader):
            if use_GPU:
                disimgs = disimgs.to(device)
                disx = disx.to(device)
                disy = disy.to(device)
                labels = labels.to(device)

            disimgs = Variable(disimgs)
            labels_x = Variable(disx)
            labels_y = Variable(disy)
            labels_clas = Variable(labels)
            flow_truth = torch.cat([labels_x, labels_y], dim=1)

            # Forward + Backward + Optimize
            optimizer.zero_grad()

            middle = model_en(disimgs)
            flow_output = model_de(middle)
            clas = model_class(middle)

            loss1 = criterion(flow_output, flow_truth)
            loss2 = criterion_clas(clas, labels_clas) * reg

            loss = loss1 + loss2

            loss.backward()
            optimizer.step()
            step += 1
            if step % 50 == 0:
                current_time = time.time()
                step_interval = current_time - (start_time if step == 50 else previous_step_time)
                previous_step_time = current_time

                print(f"Epoch [{epoch + 1}], Step [{step}], Loss: {loss.item():.4f}, "
                      f"Loss1: {loss1.item():.4f}, Loss2: {loss2.item():.4f}")
                print(f"最近50步训练花费时间: {step_interval/60:.2f}分钟")

                writer.add_scalar("train_loss1", loss1.item(), step)
                writer.add_scalar("train_loss2", loss2.item(), step)
                writer.add_scalar("train_loss", loss.item(), step)


        #测试模式
        # 验证阶段
        model_en.eval()
        model_de.eval()
        model_class.eval()
        total_loss = 0.0
        total_loss1 = 0.0
        total_loss2 = 0.0
        correct = 0
        total_samples = 0
        with torch.no_grad():
            for i, (disimgs, disx, disy, labels) in enumerate(val_loader):

                if use_GPU:
                    disimgs = disimgs.to(device)
                    disx = disx.to(device)
                    disy = disy.to(device)
                    labels = labels.to(device)

                disimgs = Variable(disimgs)
                labels_x = Variable(disx)
                labels_y = Variable(disy)
                labels_clas = Variable(labels)
                flow_truth = torch.cat([labels_x, labels_y], dim=1)

                middle = model_en(disimgs)
                flow_output = model_de(middle)
                clas = model_class(middle)

                loss1 = criterion(flow_output, flow_truth)
                loss2 = criterion_clas(clas, labels_clas) * args.reg

                loss = loss1 + loss2
                # 计算统计量
                total_loss += loss.item() * disimgs.size(0)
                total_loss1 += loss1.item() * disimgs.size(0)
                total_loss2 += loss2.item() * disimgs.size(0)
                correct += (clas.argmax(1) == labels).sum().item()
                total_samples += disimgs.size(0)
                if (i + 1) % 50 == 0:  # 更频繁的记录
                    avg_loss = total_loss / total_samples
                    avg_loss1 = total_loss1 / total_samples
                    avg_loss2 = total_loss2 / total_samples
                    accuracy = correct / total_samples

                    print(f"Val Batch [{i + 1}], Loss: {avg_loss:.4f}, "
                          f"Loss1: {avg_loss1:.4f}, Loss2: {avg_loss2:.4f}, "
                          f"Acc: {accuracy:.4f}")

                    writer.add_scalar("val_loss", avg_loss, epoch * len(val_loader) + i)
                    writer.add_scalar("val_loss1", avg_loss1, epoch * len(val_loader) + i)
                    writer.add_scalar("val_loss2", avg_loss2, epoch * len(val_loader) + i)
                    writer.add_scalar("val_acc", accuracy, epoch * len(val_loader) + i)

            # 计算整个验证集的平均指标
            avg_loss = total_loss / total_samples
            avg_loss1 = total_loss1 / total_samples
            avg_loss2 = total_loss2 / total_samples
            accuracy = correct / total_samples
            print(f'Val Epoch [{epoch + 1}], Avg Loss: {avg_loss:.4f}, '
                  f'Avg Loss1: {avg_loss1:.4f}, Avg Loss2: {avg_loss2:.4f}, '
                  f'Accuracy: {accuracy:.4f}')

            writer.add_scalar("val_epoch_loss", avg_loss, epoch)
            writer.add_scalar("val_epoch_loss1", avg_loss1, epoch)
            writer.add_scalar("val_epoch_loss2", avg_loss2, epoch)
            writer.add_scalar("val_epoch_acc", accuracy, epoch)

            # 更新学习率调度器
            scheduler.step(avg_loss)

            # 早停法判断
            if avg_loss < best_val_loss:
                best_val_loss = avg_loss
                no_improve_counter = 0
                torch.save(model_en.state_dict(), 'best_model_distill_en.pkl')
                torch.save(model_de.state_dict(), 'best_model_distill_de.pkl')
                torch.save(model_class.state_dict(), 'best_model_distill_class.pkl')
                print("发现新最佳模型，已保存!")
            else:
                no_improve_counter += 1
                print(f"验证损失连续 {no_improve_counter}/{args.patience} 次未提升")

        # 获取当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        stop_training = False
        writer.add_scalar("lr", current_lr, epoch)

        # 检查是否达到最小学习率
        if current_lr <= args.min_lr:
            print(f"\n! 学习率已达到最小值 {args.min_lr}，终止训练")
            stop_training = True
        # 检查是否达到早停条件
        if no_improve_counter >= args.patience:
            print(f"\n! 早停触发！连续 {args.patience} 个epoch验证损失未提升")
            print(f"当前最佳验证损失: {best_val_loss:.4f}")
            stop_training = True

        if stop_training:
            # 最终保存逻辑
            torch.save(model_en.state_dict(), f'model_stop_en_{epoch + 1}.pkl')
            torch.save(model_de.state_dict(), f'model_stop_de_{epoch + 1}.pkl')
            torch.save(model_class.state_dict(), f'model_stop_class_{epoch + 1}.pkl')
            print("提前终止训练！")
            break

        # 保存模型
        torch.save(model_en.state_dict(), f'model_distill_en_{epoch + 1}.pkl')
        torch.save(model_de.state_dict(), f'model_distill_de_{epoch + 1}.pkl')
        torch.save(model_class.state_dict(), f'model_distill_class_{epoch + 1}.pkl')

        epoch_time = time.time() - start_time
        print(f"本轮训练耗时: {epoch_time/60:.2f}分钟")



    torch.save(model_en.state_dict(), 'model_distill_en_last.pkl')
    torch.save(model_de.state_dict(), 'model_distill_de_last.pkl')
    torch.save(model_class.state_dict(), 'model_distill_class_last.pkl')
    writer.close()

train_loader = get_loader(distortedImgDir=r"E:\pycharm\pythonproject\learnpytorch\distortion correction\distortion_data\train",
                          flowDir=r"E:\pycharm\pythonproject\learnpytorch\distortion correction\distortion_data\train\displacement_field",
                          batch_size=args.batch_size,
                          distortion_type=args.distortion_type,
                          data_num=args.data_num)

val_loader = get_loader(distortedImgDir=r"E:\pycharm\pythonproject\learnpytorch\distortion correction\distortion_data\test",
                        flowDir=r"E:\pycharm\pythonproject\learnpytorch\distortion correction\distortion_data\test\displacement_field",
                        batch_size=args.batch_size,
                        distortion_type=args.distortion_type,
                        data_num=int(args.data_num * 0.1) + 60000)

#加载tensorboard
writer = SummaryWriter("logs_distill")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_en = EncoderNet([1,1,1,1,2]).to(device)
model_de = DecoderNet([1,1,1,1,2]).to(device)
model_class = ClassNet().to(device)
criterion = EPELoss().to(device)
criterion_clas = nn.CrossEntropyLoss().to(device)

print('dataset type:', args.distortion_type)
print('dataset number:', args.data_num)
print('batch size:', args.batch_size)
print('epochs:', args.epochs)
print('lr:', args.lr)
print('reg:', args.reg)
print('train_loader', len(train_loader), 'train_num', args.batch_size * len(train_loader))
print('val_loader', len(val_loader), 'test_num', args.batch_size * len(val_loader))
# print(model_en, model_de, model_class, criterion)

reg = args.reg
lr = args.lr

# #调试
# test_input = torch.randn(1, 3, 512, 512).to(device)
# middle = model_en(test_input)
# print("输出中间层尺寸:", middle.shape)
# flow_output = model_de(middle)
# print("输出位移场尺寸:", flow_output.shape)
# clas = model_class(middle)
# print("输出分类尺寸:", clas.shape)


# 初始化优化器时建议添加权重衰减
optimizer = torch.optim.Adam([
    {'params': model_en.parameters(), 'lr': args.lr},
    {'params': model_de.parameters(), 'lr': args.lr},
    {'params': model_class.parameters(), 'lr': args.lr}
], weight_decay=1e-5)  # 新增权重衰减
train(train_loader,val_loader,args.epochs,optimizer, criterion, criterion_clas)