# SSRnet

# abstract
(Background) Cell counting and segmentation are critical tasks in the fields of biology and medicine. The traditional methods for cell counting are labor-intensive, time-consuming, and prone to human errors. Recently, deep learning-based cell counting methods become a trend, including point-based counting methods, such as cell detection and cell density prediction, and non-point-based counting, such as cell number regression prediction. However, the point-based counting method heavily relies on well-annotated datasets, which are scarce and difficult to obtain. On the other hand, non-point-based counting is less interpretable. (Method) We have approached the task of cell counting by dividing it into two sub-tasks: cell number prediction and cell distribution prediction. To accomplish this, we propose a deep learning network for spatial-based super-resolution reconstruction (SSRNet) that predicts the cell count and segments the cell distribution contour. To effectively train the model, we propose an Optimized Multitask Loss function (OM loss) that coordinates the training of multiple tasks. In SSRNet, we propose a Spatial-based Super-Resolution Fast Upsampling Module (SSR-upsampling) for feature map enhancement and one-step upsampling, which can enlarge the deep feature map by 32 times without blurring and achieves fine-grained detail and fast processing. (Result) Our SSRNet uses an optimized encoder network. Compared with the classic U-Net, our SSRNet's running memory read and write consumption is only 1/10 of that of U-Net, and the total number of multiply and add calculations is 1/20 of that of U-Net. Compared with the traditional sampling method, our SSR-upsampling can complete the upsampling of the entire decoder stage at one time, reducing the complexity of the network and achieving better performance. Our experiments demonstrate that our method achieves state-of-the-art performance in cell counting and segmentation tasks. (Conclusion) Our method has achieved non-point-based counting, eliminating the need for exact position annotation of each cell in the image during training. As a result, it has demonstrated excellent performance on cell counting and segmentation tasks.

# structure of SSRNet
![image](https://github.com/Roin626/SSRnet/assets/44090641/a07ab49b-9bf1-4aa9-a38d-b7911ee72d4e)



# Citation

If you find this code useful for your research, please consider citing:
```bibtex
@article{https://doi.org/10.1002/aisy.202300185,
author = {Deng, Lijia and Zhou, Qinghua and Wang, Shuihua and Zhang, Yudong},
title = {SSRNet: A Deep Learning Network via Spatial-Based Super-resolution Reconstruction for Cell Counting and Segmentation},
journal = {Advanced Intelligent Systems},
volume = {5},
number = {10},
pages = {2300185},
keywords = {artificial intelligence, automated counting, cell counting, cell segmentation, convocational neural network, deep learning, machine learning},
doi = {https://doi.org/10.1002/aisy.202300185},
url = {https://onlinelibrary.wiley.com/doi/abs/10.1002/aisy.202300185},
eprint = {https://onlinelibrary.wiley.com/doi/pdf/10.1002/aisy.202300185},
abstract = {Cell counting and segmentation are critical tasks in biology and medicine. The traditional methods for cell counting are labor-intensive, time-consuming, and prone to human errors. Recently, deep learning-based cell counting methods have become a trend, including point-based counting methods, such as cell detection and cell density prediction, and non-point-based counting, such as cell number regression prediction. However, the point-based counting method heavily relies on well-annotated datasets, which are scarce and difficult to obtain. On the other hand, nonpoint-based counting is less interpretable. The task of cell counting by dividing it into two subtasks is approached: cell number prediction and cell distribution prediction. To accomplish this, a deep learning network for spatial-based super-resolution reconstruction (SSRNet) is proposed that predicts the cell count and segments the cell distribution contour. To effectively train the model, an optimized multitask loss function (OM loss) is proposed that coordinates the training of multiple tasks. In SSRNet, a spatial-based super-resolution fast upsampling module (SSR-upsampling) is proposed for feature map enhancement and one-step upsampling, which can enlarge the deep feature map by 32 times without blurring and achieves fine-grained detail and fast processing. SSRNet uses an optimized encoder network. Compared with the classic U-Net, SSRNet's running memory read and write consumption is only 1/10 of that of U-Net, and the total number of multiply and add calculations is 1/20 of that of U-Net. Compared with the traditional sampling method, SSR-upsampling can complete the upsampling of the entire decoder stage at one time, reducing the complexity of the network and achieving better performance. Experiments demonstrate that the method achieves state-of-the-art performance in cell counting and segmentation tasks. The method achieves nonpoint-based counting, eliminating the need for exact position annotation of each cell in the image during training. As a result, it has demonstrated excellent performance on cell counting and segmentation tasks. The code is public on GitHub (https://github.com/Roin626/SSRnet).},
year = {2023}
}
```
