import argparse

from torch.utils.data import DataLoader

from models import *
from model.layers import *
from dataset.BBBC005.datasets import *

import torch
from torchvision.utils import save_image
from util.misc import *
from util import pytorch_ssim
from util import util_image as ui
from tensorboardX import SummaryWriter
from sklearn.model_selection import KFold

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=30, help="number of epochs of training")
parser.add_argument("--dataset_name", type=str, default=r'D:\cc\pix2pix\dataset\BBBC005', help="name of the dataset")
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0001, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
parser.add_argument("--n_cpu", type=int, default=1, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_height", type=int, default=128, help="size of image height")
parser.add_argument("--img_width", type=int, default=128, help="size of image width")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")

parser.add_argument("--level_class", type=int, default=1, help="number of count level class")
parser.add_argument("--lambda_pixel", type=float, default=10, help="number of lambda_pixel loss")
parser.add_argument("--lambda_count", type=float, default=1, help="number of lambda_count loss")
parser.add_argument("--lambda_level", type=float, default=1, help="number of lambda_level loss")

parser.add_argument("--idx", type=int, default=0, help="idx of test data part")

parser.add_argument(
    "--sample_interval", type=int, default=5, help="interval between sampling of images from generators"
)
parser.add_argument("--checkpoint_interval", type=int, default=5, help="interval between model checkpoints")

parser.add_argument('--output_dir', default='./log/SSRNet_FBU_X2_bc5_Hloss_10pixel_1count1',
                    help='path where to save, empty for no saving')
parser.add_argument('--checkpoints_dir', default='./ckpt/SSRNet_FBU_X2_bc5_Hloss_10pixel_1count1',
                    help='path where to save checkpoints, empty for no saving')
parser.add_argument('--tensorboard_dir', default='./runs/SSRNet_FBU_X2_bc5_Hloss_10pixel_1count1',
                    help='path where to save, empty for no saving')
# * BackboneW
parser.add_argument('--backbone', default='vgg16_bn', type=str,
                    help="Name of the convolutional backbone to use")
parser.add_argument('--gpu_id', default=1, type=int, help='the gpu used for training')

opt = parser.parse_args()
print(opt)

os.makedirs(opt.output_dir, exist_ok=True)
os.makedirs(opt.checkpoints_dir, exist_ok=True)
os.makedirs(opt.tensorboard_dir, exist_ok=True)

os.environ["CUDA_VISIBLE_DEVICES"] = '{}'.format(opt.gpu_id)
cuda = True if torch.cuda.is_available() else False
# Loss functions
criterion_GAN = torch.nn.MSELoss()
# criterion_GAN = torch.nn.L1Loss()#torch.nn.MSELoss()
# criterion_pixelwise = torch.nn.MSELoss()
criterion_pixelwise = torch.nn.L1Loss()
# criterion_SSIM = pytorch_ssim.SSIM(window_size=11)
# auxiliary_loss = torch.nn.CrossEntropyLoss()

# Loss weight of L1 pixel-wise loss between translated image and real image
lambda_pixel = opt.lambda_pixel
lambda_count = opt.lambda_count
lambda_level = opt.lambda_level

# Calculate output of image discriminator (PatchGAN)
patch = (1, opt.img_height // 2 ** 4, opt.img_width // 2 ** 4)
#####################################
######################################
class GeneratorUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(GeneratorUNet, self).__init__()

        self.down1 = UNetDown(in_channels, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512)
        self.down5 = UNetDown(512, 512, normalize=True, dropout=0.5)
        self.down6 = UNetDown(512, 512, normalize=False, dropout=0.5)

        self.level_class = levelclass(input_channel=128, pred_shapes=[1, 1, 32, 32])


        self.up5 = UNetcat(256, 64)
        self.up1 = nn.Sequential(
            # nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0, bias=False),
            # nn.Upsample(scale_factor=16, mode='nearest'),
            # nn.ConvTranspose2d(512, 256, 4, 16,output_padding=12, bias=False),
            SSRupsampling2(inplanes=512, outplanes=256, scale=2,k_size=1,  pad=0),
            nn.ReLU(inplace=True),
            SSRupsampling2(inplanes=256, outplanes=256, scale=2, k_size=1, pad=0),
            nn.ReLU(inplace=True),
            SSRupsampling2(inplanes=256, outplanes=256, scale=2, k_size=1, pad=0),
            nn.ReLU(inplace=True),
            SSRupsampling2(inplanes=256, outplanes=256, scale=2, k_size=1, pad=0),
            nn.ReLU(inplace=True),
            SSRupsampling2(inplanes=256, outplanes=256, scale=2, k_size=1, pad=0),
            nn.ReLU(inplace=True),

        )
        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(128, 3, 4, padding=1),
            # nn.Tanh(),

            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # U-Net generator with skip connections from encoder to decoder
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)

        u1 = self.up1(d6)

        u5 = self.up5(u1, d1)
        # count = self.count(u5)
        count = self.level_class(u5)
        return self.final(u5),count

class levelclass(nn.Module):
    def __init__(self,input_channel=512,pred_shapes=[1, 128, 1, 1]):
        super(levelclass, self).__init__()

        self.pred_shape = pred_shapes

        self.pred_vectors = nn.Parameter(torch.rand(self.pred_shape),
                                          requires_grad=True)

        self.inputlayer=nn.Sequential(nn.AdaptiveAvgPool2d((32,32)), #fax feature size
                                nn.Conv2d(input_channel, 1, 1, stride=1, padding=0),#down channel
                                # nn.InstanceNorm2d(128),
                                nn.ReLU(),
                                )


        self.count_layer = nn.Conv2d(32, 1, 32, stride=1, padding=0)

    def forward(self, input):
        x = self.inputlayer(input)
        count = F.conv2d(input=x, weight=self.pred_vectors)

        return count

class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalization=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels * 2, 64, normalization=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),#left, right, up, down
            nn.Conv2d(512, 1, 4, padding=1, bias=False)
        )


    def forward(self, img_A, img_B):
        # Concatenate image and condition image by channels to produce input
        img_input = torch.cat((img_A, img_B), 1)
        out = self.model(img_input)

        return out
########################################
######################################
# Initialize generator and discriminator
# backbone=build_backbone(opt)
# backbone=build_backbone(opt)
# generator = Model(backbone)
# generator =Res50()
generator = GeneratorUNet()
# discriminator = Discriminator()

if cuda:
    generator = generator.cuda()
    # discriminator = discriminator.cuda()
    criterion_GAN.cuda()
    criterion_pixelwise.cuda()
    # criterion_SSIM.cuda()
    # auxiliary_loss.cuda()

if opt.epoch != 0:
    # Load pretrained model
    generator.load_state_dict(torch.load(os.path.join(opt.checkpoints_dir,"generator_%d.pth" % (opt.epoch))))
    # discriminator.load_state_dict(torch.load(os.path.join(opt.checkpoints_dir,"discriminator_%d.pth" % (opt.epoch))))
else:
    pass
    # Initialize weights
    # generator.apply(weights_init_normal)
    # discriminator.apply(weights_init_normal)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
# optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

################################
########   load data   #########
################################
def loading_data(data_root):
    # the pre-proccssing transform
    transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(mean=[0.387, 0.387, 0.387],
                                    std=[0.278, 0.278, 0.278]),
    ])
    img_paths=[]
    img_dir1 = os.path.join(data_root, 'BBBC005_v1_images')
    path_set = [img_dir1]
    for path in path_set:
        for img_path in glob.glob(os.path.join(path, '*.TIF')):
            img_paths.append(img_path)

    kf = KFold(n_splits=5, shuffle=False)
    idx=opt.idx
    print(kf)
    img_train=[]
    img_test=[]
    Train_idx_set=[]
    Test_idx_set =[]
    # 做split时只需传入数据，不需要传入标签
    for train_index, test_index in kf.split(img_paths):
        Train_idx_set.append(train_index)
        Test_idx_set.append(test_index)
    print("TRAIN:", Train_idx_set[idx], "TEST:", Test_idx_set[idx])
    for i in Train_idx_set[idx]:
            img_train.append(img_paths[i])
    for j in Test_idx_set[idx]:
            img_test.append(img_paths[j])


    # create the training dataset
    train_set = ImageDataset(img_train, train=True, transform=transform, patch=False, flip=False,resize=False)
    # create the validation dataset
    val_set = ImageDataset(img_test, train=False, transform=transform)

    return train_set, val_set

###### Load data########
########################
dataset_name =opt.dataset_name
train_set, val_set = loading_data(dataset_name)
sampler_train = torch.utils.data.RandomSampler(train_set)
sampler_val = torch.utils.data.SequentialSampler(val_set)
batch_sampler_train = torch.utils.data.BatchSampler(
    sampler_train, opt.batch_size, drop_last=True)

data_loader_train = DataLoader(train_set, batch_sampler=batch_sampler_train,
                               collate_fn=collate_fn_bc05, num_workers=0)

data_loader_val = DataLoader(val_set, 1, sampler=sampler_val,
                             drop_last=False, collate_fn=collate_fn_bc05, num_workers=0)

# create the logging file
run_log_name = os.path.join(opt.output_dir, 'run_log.txt')
with open(run_log_name, "w") as log_file:
    log_file.write('Eval Log %s\n' % time.strftime("%c"))

# Tensor type
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

#########################################
#########      save_image     ###########
def save_images(fake_B,imgdir,w_border=960,h_border=960):

    fake_B = fake_B[0][0].data.cpu().numpy()
    fake_B=fake_B / (np.max(fake_B) + 5e-4) * 255
    fake_B[fake_B<0]=0
    fake_B = fake_B.astype(np.uint8)

    if fake_B.ndim == 2:
        h, w = fake_B.shape[:]
        # img1=img1[border:h-border,border:w-border]
        fake_B = fake_B[0:h_border, 0:w_border]
        fake_B = fake_B.transpose([0, 1])
    elif fake_B.ndim == 3:
        fake_B = fake_B[:, 0:h_border, 0:w_border]
        fake_B = fake_B.transpose([1, 2, 0])


    # fake_B = cv2.cvtColor(fake_B, cv2.COLOR_BGR2RGB)
    # fake_B = cv2.applyColorMap(fake_B, cv2.COLORMAP_JET)
    imgdir = os.path.join(opt.output_dir, "images%s.png" % (i))
    cv2.imwrite(imgdir, fake_B)
    imgdir = os.path.join(opt.output_dir, "boneimages%s.png" % (i))
    fake_B = cv2.applyColorMap(fake_B, cv2.COLORMAP_BONE)
    cv2.imwrite(imgdir, fake_B)
    return fake_B

#########################################
############       test      ############
def test_images():
    """Saves a generated sample from the validation set"""
    maes = []
    mses = []

    img_maes = []
    img_mses = []
    psnrs= []
    ssims=[]
    t_avg=[]

    for i, (samples, targets,gt_count) in enumerate(data_loader_val):
        generator.eval()
        real_img = samples.cuda()
        real_gt = targets.cuda()
        # 生成计数gt，该数据集gt为传入的count
        real_count=Variable(Tensor(gt_count),requires_grad=False)
        # real_level=Variable(Tensor(gt_level),requires_grad=False)

        t1=time.time()
        fake_B,count = generator(real_img)
        t2=time.time()
        time_left=t2-t1
        t_avg.append(time_left)

        imgdir = os.path.join(opt.output_dir,"newimages%s.png" % (i))
        save_images(fake_B, imgdir, w_border=696, h_border=520)
        # img_sample = torch.cat((real_img.data, fake_B.data, real_gt.data), -2)
        save_image(fake_B[:,:,0:520, 0:696].data, os.path.join(opt.output_dir,"newimages%s.png" % (i)), nrow=5, normalize=True)
        loss_pixel = criterion_pixelwise(fake_B, real_gt)

        mae = abs(count.squeeze()-real_count.squeeze())
        mse = (count.squeeze() - real_count.squeeze()) * (count.squeeze() - real_count.squeeze())
        maes.append(float(mae))
        mses.append(float(mse))

        im_gt = real_gt[:,:,0:520, 0:696].data.cpu().numpy().astype(np.uint8)
        im_fakeb = fake_B[:,:,0:520, 0:696].data.cpu().numpy().astype(np.uint8)

        img_mae = np.mean(abs(im_fakeb - im_gt))
        img_mse = np.mean((im_fakeb - im_gt)**2)
        img_maes.append(float(img_mae))
        img_mses.append(float(img_mse))

        psnr = ui.calculate_psnr(fake_B.data.cpu().numpy().astype(np.uint8),real_gt.data.cpu().numpy().astype(np.uint8),border=696)
        psnrs.append(psnr)
        ssim = pytorch_ssim.ssim(fake_B, real_gt, border=696)
        ssim = ssim.data.cpu().numpy().astype(np.float32)
        ssims.append(ssim)
    psnr = np.mean(psnrs)
    ssim=np.mean(ssims)
    mae = np.mean(maes)
    mse = np.sqrt(np.mean(mses))
    img_mae = np.mean(img_maes)
    img_mse = np.sqrt(np.mean(img_mses))
    t_avg=np.mean(t_avg)

    return loss_pixel, mae,mse, psnr,ssim,img_mae, img_mse, fake_B[:,:,0:520, 0:696].data,t_avg

    # ------------------
    #  Train Generators
    # ------------------

# ----------######################
##########    Training        ####
# ----------######################
writer = SummaryWriter(opt.tensorboard_dir)
prev_time = time.time()
MAE =[]
MSE =[]
PSNR =[]
SSIM=[]
IMG_MAE=[]
IMG_MSE=[]
ACC=[]
step = 0
acc=0
best_mae_epoch = 0
best_ssim_epoch=0
best_img_mae_epoch=0
best_epoch=0
for epoch in range(opt.epoch, opt.n_epochs):
    for i, (samples,targets,gt_count) in enumerate(data_loader_train):
        real_img = samples.cuda()
        real_gt = targets.cuda()
        # Adversarial ground truths
        valid = Variable(Tensor(np.ones((real_img.size(0),real_img.size(1),real_img.size(2),real_img.size(3)))), requires_grad=False)
        fake = Variable(Tensor(np.zeros((real_img.size(0), *patch))), requires_grad=False)
        real_count = Variable(Tensor(gt_count),requires_grad=False)

        # real_level = Variable(Tensor(gt_level), requires_grad=False).long()

        generator.train()
        optimizer_G.zero_grad()

        t1 = time.time()
        # GAN loss
        fake_B, fake_count = generator(real_img)
        t4 = time.time()

        ############## count loss####################
        loss_pixel = criterion_pixelwise(fake_B, real_gt)
        loss_count = criterion_GAN(fake_count.squeeze(), real_count.squeeze())

        loss_G =  lambda_pixel * loss_pixel + lambda_count * loss_count

        loss_G.backward()
        optimizer_G.step()

        # --------------
        #  Log Progress
        # --------------

        time_left = t4 - t1

        prev_time = time.time()

        # Print log
        if i % 50 == 0:
            print(
                "\r[Epoch %d/%d] [Batch %d/%d] [G loss: %f, pixel: %f, count: %f] ETA: %s"
                % (
                    epoch,
                    opt.n_epochs,
                    i,
                    len(data_loader_train),
                    loss_G.item(),
                    loss_pixel.item(),
                    loss_count.item(),
                    time_left,
                )
            )
        if writer is not None:
            with open(run_log_name, "a") as log_file:

                log_file.write("loss/loss_G@{}: {}".format(epoch, loss_G.item()))
                log_file.write("loss/loss_pixel@{}: {}".format(epoch, loss_pixel.item()))
                log_file.write("loss/loss_count@{}: {}".format(epoch, loss_count.item()))


            writer.add_scalar('loss/loss_G', loss_G, epoch)
            writer.add_scalar('loss/loss_count', loss_count, epoch)
            writer.add_scalar('loss/loss_pixel', loss_pixel, epoch)


            ############################   test #################################################
    if opt.checkpoint_interval != -1 and (epoch + 1) % opt.checkpoint_interval == 0:
        t1 = time.time()
        loss, mae, mse, psnr, ssim,img_mae,img_mse, img_sample, time_left = test_images()
        t2 = time.time()
        # #########################统计测试结果
        MAE.append(mae)
        MSE.append(mse)
        SSIM.append(ssim)
        PSNR.append(psnr)
        IMG_MAE.append(img_mae)
        IMG_MSE.append(img_mse)

        best_mae = np.min(MAE)
        best_mse = np.min(MSE)
        best_ssim = np.max(SSIM)
        best_psnr = np.max(PSNR)
        best_img_mae = np.min(IMG_MAE)
        best_img_mse = np.min(IMG_MSE)

        if mae <= best_mae and epoch !=0 and ssim>=best_ssim:
            best_epoch=epoch

        if mae <= best_mae and epoch != 0:
            best_mae_epoch = epoch
            # imgdir = os.path.join(opt.output_dir, "bestimage.png")
            # save_image(img_sample, imgdir, w_border=696, h_border=520)
            save_image(img_sample, os.path.join(opt.output_dir, "bestimage.png"), nrow=5, normalize=True)
            # Save model checkpoints
            torch.save(generator.state_dict(), os.path.join(opt.checkpoints_dir, "generator_%d.pth" % (epoch)))
        if ssim >= best_ssim and epoch != 0:
            best_ssim_epoch = epoch
            # imgdir = os.path.join(opt.output_dir, "bestimage_ssim.png")
            # save_image(img_sample, imgdir, w_border=696, h_border=520)
            save_image(img_sample, os.path.join(opt.output_dir, "bestimage_ssim.png"), nrow=5, normalize=True)
            # Save model checkpoints
            torch.save(generator.state_dict(),
                        os.path.join(opt.checkpoints_dir, "generator_ssim_%d.pth" % (epoch)))
        if img_mae <= best_img_mae and epoch != 0:
            best_img_mae_epoch = epoch

            save_image(img_sample, os.path.join(opt.output_dir, "bestimage_mae.png"), nrow=5, normalize=True)

        # Print log
        print("_____________________________________")
        print("\r[valEpoch %d/%d] [loss=%f] [mae:%f] [mse:%f] [psnr: %f] [ssim: %f] [img_mae: %f] [img_mse: %f]  ETA: %s"
                % (
                epoch,
                opt.n_epochs,
                loss,
                mae,
                mse,
                psnr,
                ssim,
                img_mae,
                img_mse,
                time_left
                )
                )
        print("_____________________________________")
        print("##################################################")
        print(
            "[best_mae %f] [best_mse %f] [best_mae_epoch %d] [best_psnr %f] [best_ssim %f] [best_ssim_epoch %d] [best_img_mae %f]"
            " [best_img_mse %f] [best_img_mae_epoch %d] [best_epoch %d]" % (
            best_mae, best_mse, best_mae_epoch,
            best_psnr, best_ssim, best_ssim_epoch,
            best_img_mae,best_img_mse,best_img_mae_epoch, best_epoch))

        with open(run_log_name, "a") as log_file:
            log_file.write("mae:{}, mse:{},psnr:{},ssim:{}, time:{}, best mae:{}, best mse:{}, best mae epoch:{},"
                           " best psnr:{},best ssim:{}, best ssim epoch{},"
                           " best img_mae:{},best img_mse:{},best img_mae epoch{}".format(mae, mse,ssim,psnr, time_left,
                                        best_mae, best_mse, best_mae_epoch,
                                        best_psnr, best_ssim, best_ssim_epoch,
                                        best_img_mae,best_img_mse,best_img_mae_epoch))
            log_file.write("time@{}: {}".format(epoch, time_left))
        print("##################################################")
        if writer is not None:
            with open(run_log_name, "a") as log_file:
                log_file.write("metric/mae@{}: {}".format(step, mae))
                log_file.write("metric/mse@{}: {}".format(step, mse))
                log_file.write("metric/psnr@{}: {}".format(step, psnr))
                log_file.write("metric/ssim@{}: {}".format(step, ssim))
            writer.add_scalar('metric/mae', mae, step)
            writer.add_scalar('metric/mse', mse, step)
            writer.add_scalar('metric/psnr', psnr, step)
            writer.add_scalar('metric/ssim', ssim, step)
            writer.add_scalar('metric/im_mae', img_mae, step)
            writer.add_scalar('metric/im_mse', img_mse, step)
            step += 1