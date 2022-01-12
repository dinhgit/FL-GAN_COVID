# FL-GAN_COVID
This is source code for the paper: "Federated Learning for COVID-19 Detection with Generative Adversarial Networks in Edge Cloud Computing", published at the IEEE Internet of Things Journal, Nov. 2021 (https://ieeexplore.ieee.org/abstract/document/9580478)
# Requirements:
python >=3.5

tensorflow >= 2.6

pytorch >= 0.4
# How to run this code: 
Install all required libraries and now ready to run the code. It is recommended to run the standalone GAN code "COVID_GAN3.py" first, then run the FL-GAN code "Server_COVID.py". Here, we set 1 server and random 50 hospitals as hospital clients for training the COVID X-ray data. Then, using the CNN classifer with the code "CNN_COVID_Classification.py" for COVID detection.
# COVID-19 datasets: 
This paper has used two datasets called ChestCOVID dataset (https://github.com/ieee8023/COVID-chestxray-dataset) and DarkCOVID dataset (https://github.com/muhammedtalo/COVID-19). It is recommended to download these datasets and save in the folder "CovidDataset" for running the codes. 
# Citation: 
The paper is available at https://arxiv.org/abs/2110.07136 and the authors using this code should cite as: 

--------------------------
@article{nguyen2021federatedcovid,

  title={Federated learning for covid-19 detection with generative adversarial networks in edge cloud computing},
  
  author={Nguyen, Dinh C and Ding, Ming and Pathirana, Pubudu N and Seneviratne, Aruna and Zomaya, Albert Y},
  
  journal={IEEE Internet of Things Journal},
  
  year={2021},
}
