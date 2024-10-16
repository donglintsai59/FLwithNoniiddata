# FL with non-iid data 

## Introduction
This program aims to simulate real-world FL System, so we use non-IID data as referenced in the paper to achieve this.
We need to run two different programs to meet the server and device requirements. You'll need to run client.py on the Jetson Nano and serverwandb.py on your computer.

## Dataset

MNIST
We use Dirichlet distribution to convert the data into a non-IID. The Dirichlet distribution is a continuous multivariate probability distribution, which is an extension of the Beta distribution to higher dimensions. It is primarily used for modeling proportions of random variables whose sum is constrained to be one.
## Reslut
The program includes the Weights and Biases (wandb) API to easily monitor and visualize the experiment data.
![image](https://github.com/donglintsai59/FLwithNoniiddata/blob/main/reslut.jpg)
