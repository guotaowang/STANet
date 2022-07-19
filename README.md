# Pytorch Implementation for **(STANet+ and STANet)** 
          
### *V<sub>2</sub>*-Weakly Supervised Visual-Auditory Human-eye Fixation Prediction with Multigranularity Perception ([arxiv](https://arxiv.org/abs/2112.13697)), pdf:[V<sub>2</sub>](https://arxiv.org/pdf/2112.13697.pdf)
### *V<sub>1</sub>*-From Semantic Categories to Fixations: A Novel Weakly-supervised Visual-auditory Saliency Detection Approach ([CVPR2021](https://openaccess.thecvf.com/CVPR2021)), pdf:[V<sub>1</sub>](https://openaccess.thecvf.com/content/CVPR2021/papers/Wang_From_Semantic_Categories_to_Fixations_A_Novel_Weakly-Supervised_Visual-Auditory_Saliency_CVPR_2021_paper.pdf)
****
### Introduction
*  This repository contains the source code, results, and evaluation toolbox of **STANet+** (V2), which are the journal **extension version** of our paper **STANet** (V<sub>1</sub>) published at **CVPR-2021**.    
*  Compared our conference version STANet (V<sub>2</sub>), which  has been extended in **two distinct aspects**.     
**First** on the basis of multisource and multiscale perspectives which have been adopted by the CVPR version (V1), we have provided a deep insight into the relationship between **multigranularity perception** (***Fig.2***) and real human attention behaved in visual-auditory environment.     
**Second** without using any complex networks, we have provided an elegant framework to complementary integrate **multisource, multiscale, and multigranular information** (***Fig.1***) to formulate pseudofixations which are very consistent with the real ones. Apart from achieving significant performance gain, this work also provides a comprehensive solution for mimicking multimodality attention.    

<div align=center><img width="800" height="380" src="https://github.com/guotaowang/STANet/blob/main/fig/STANet2.gif"/></div>
<p align="center">
Figure 1: STANet+ mainly focuses on devising a weakly supervised approach for the spatial-temporal-audio (STA) fixation prediction task, where     
the key innovation is that, as one of the first attempts, we automatically convert semantic category tags to pseudofixations via the      
newly proposed selective class activation mapping (SCAM) and the upgraded version SCAM+ that has been additionally       
equipped with the multigranularity perception ability. The obtained pseudofixations can be used as the learning objective      
to guide knowledge distillation to teach two individual fixation prediction networks (i.e., STA and STA+), which      
jointly enable generic video fixation prediction without requiring any video tags. </p>     

<div align=center><img width="600" height="700" 
src="https://github.com/guotaowang/STANet/blob/main/fig/MG.gif"/></div>
<p align="center">
Figure 2: Some representative ’fixation shifting’ cases, additional multigranularity information (i.e., long/crossterm information) has been shown before collecting fixations in A_SRC. Clearly, by comparing A_FIX0, A_FIX1, and A _FIX2, we can easily notice that the multigranularity information could draw human attention to the most meaningful objects and make the fixations to be more focused. </p>     

### Dependencies  
* Windows10    
* NVIDIA GeForce RTX 2070 SUPER & NVIDIA GeForce RTX 1080Ti  
* python 3.6.4    
* Matlab R2016b    
* pytorch 1.8.0    
* soundmodel       

### Preparation
##### Downloading the official pretrained visual and audio model    
**Visual**:[resnext101_32x8d](https://pytorch.org/hub/facebookresearch_WSL-Images_resnext/), [vgg16](https://download.pytorch.org/models/vgg16-397923af.pth)     
**Audio**: [vggsound](https://github.com/hche11/VGGSound), `net = torch.load('vggsound_netvlad')`.    
##### Downloading the training dataset and testing dataset:    
**Training dataset**: [AVE](https://drive.google.com/file/d/1FjKwe79e0u96vdjIVwfRQ1V6SoDHe7kK/view)(Audio Visual Event Location).    
**Testing dataset**: [AVAD](https://sites.google.com/site/minxiongkuo/home), [DIEM](https://thediemproject.wordpress.com/videos-and%c2%a0data/), [SumMe](https://gyglim.github.io/me/vsum/index.html#benchmark), [ETMD](http://cvsp.cs.ntua.gr/research/aveyetracking/), [Coutrot](http://antoinecoutrot.magix.net/public/databases.html).  

### Training     
> **Note**    
We use Fourier-transform to transform audio features as audio stream input, therefore, you firstly need to use the function audiostft.py to convert the audio files (.wav) to get the audio features(.h5).       
  
##### *Step 1.    SCAM training*     
**Coarse:** Separately training branches of **S<sub>coarse</sub>**,  **SA<sub>coarse</sub>**,   **ST<sub>coarse</sub>**  ，it should be noted that the coarse stage is coarse location, so the size is set to 256 to ensure object-wise location accuracy.      
**Fine:** Separately re-training branches of **S<sub>fine</sub>**, **SA<sub>fine</sub>**,    **ST<sub>fine</sub>**，it should be noted that the fine stage is a fine location, so the size is set to 356 to ensure regional location exactness.      
#####    *Step2.    SCAM+ training*           
**S+:** Separately training branches of  **S+<sub>short</sub>**,  **S+<sub>long</sub>**,  **S+<sub>cross</sub>**,  because it is frame-wise relational reasoning network, the network is the same, so we only need to change the source of the input data.      
**SA+:** Separately training branches of **SA+<sub>long</sub>**,  **SA+<sub>cross</sub>**.     
**ST+:**  Separately training branches of **ST+<sub>short</sub>**,  **ST+<sub>long</sub>**,  **ST+<sub>cross</sub>**.      
#####  *Step 3.   pseudoGT generation*     
In order to facilitate the display of matrix data processing, **Matlab2016b**  was performed in coarse location of inter-frame smoothing and pseudo GT data post-processing.    
#####  *Step 4.  STA and STA+ training*       
Training the model of **STA** and **STA+** using the AVE video frames with the generated **pseudoGT**.    

### Testing   
##### ***Step 1.*** Using the function **audiostft.py** to convert the audio files (.wav) to get the audio features (.h5).  
##### ***Step 2.*** Testing STA, STA+ network, fusing the test results to generate final saliency results.(STANet+)  
The model **weight** file STANet+, STANet, AudioSwitch:  
([Baidu Netdisk](https://pan.baidu.com/s/1nvtJm1Z6-sHBaLPsEHhw4Q), code:6afo).    
### Evaluation  
##### We use the evaluation code in the paper of [**STAVIS**](https://github.com/atsiami/STAViS) for fair  comparisons.     
##### You may need to revise the algorithms, data_root, and maps_root defined in the **main.m**.     
##### We provide the saliency maps of the **SOTA**:  
(**STANet+**, **STANet**, ITTI, GBVS, SCLI, AWS-D, SBF, CAM, GradCAM, GradCAMpp, SGradCAMpp, xGradCAM, SSCAM, ScoCAM, LCAM, ISCAM, ACAM, EGradCAM, ECAM, SPG, VUNP, WSS, MWS, WSSA).  
([Baidu Netdisk](https://pan.baidu.com/s/1nvtJm1Z6-sHBaLPsEHhw4Q), code:6afo).  

#### Quantitative comparisons:    
<div align=center><img width="650" height="850" src="https://github.com/guotaowang/STANet/blob/main/fig/Compare2.gif"/></div>
<p align="center">  Qualitative results of our method and eight representative saliency models: ITTI, GBVS, SCLI, SBF, AWS-D, WSS, MWS, WSSA. It can be observed that our method is able to handle various challenging scenes well and produces more accurate results than other competitors.   </p>     

#### Qualitative comparisons:
<p align="center">  Quantitative comparisons between our method with other fully-/weakly-/un-supervised methods on 6 datasets. Bold means the best result, "
denotes the higher the score, the better the performance.  </p>     
<div align=center><img width="700" height="800" src="https://github.com/guotaowang/STANet/blob/main/fig/compare3.gif"/></div>  

### References  
[1][Tsiami, A., Koutras, P., Maragos, P.STAViS: Spatio-Temporal AudioVisual Saliency Network. (CVPR 2020).]  (https://openaccess.thecvf.com/content_CVPR_2020/papers/Tsiami_STAViS_Spatio-Temporal_AudioVisual_Saliency_Network_CVPR_2020_paper.pdf)  
[2][Tian, Y., Shi, J., Li, B., Duan, Z., Xu, C. Audio-Visual Event Localization in Unconstrained Videos. (ECCV 2018)] (https://openaccess.thecvf.com/content_ECCV_2018/papers/Yapeng_Tian_Audio-Visual_Event_Localization_ECCV_2018_paper.pdf)  
[3][Chen, H., Xie, W., Vedaldi, A., & Zisserman, A. Vggsound: A Large-Scale Audio-Visual Dataset. (ICASSP 2020)]   (https://www.robots.ox.ac.uk/~vgg/publications/2020/Chen20/chen20.pdf)   

### Citation  
If you find this work useful for your research, please consider citing the following paper:   

    @InProceedings{Wang_2021_CVPR,  
        author    = {Wang, Guotao and Chen, Chenglizhao and Fan, Deng-Ping and Hao, Aimin and Qin, Hong},
        title     = {From Semantic Categories to Fixations: A Novel Weakly-Supervised Visual-Auditory Saliency Detection Approach},  
        booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},  
        month     = {June},  
        year      = {2021},  
        pages     = {15119-15128}  
    }  
    
    
    @article{wang2021weakly,
       title={Weakly Supervised Visual-Auditory Human-eye Fixation Prediction with Multigranularity Perception},
       author={Wang, Guotao and Chen, Chenglizhao and Fan, Deng-ping and Hao, Aimin and Qin, Hong},
       journal={arXiv preprint arXiv:2112.13697},
       year={2021}
    }
