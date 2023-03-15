# Resolution_Pumper
Pumps up the Image resolution by x4 factor  



## Problem Statement 

To solve the degradation of resolution of images on social media and messaging platforms like whatsapp and instagram by pumping up its resolution

<br>

## Dataset

The Dataset used is the Housing dataset which contains information about different houses in Boston. This data was originally a part of UCI Machine Learning Repository and has been removed now. We can also access this data from the scikit-learn library and kaggle. There are 506 samples and 13 feature variables in this dataset. The objective is to predict the value of prices of the house using the given features.


<br>


## Web Interface

![pump](https://user-images.githubusercontent.com/114252357/225192633-f62abffc-89e2-44ed-80e2-d813b40c25b9.png)



<br>
<br>


##  Steps to Deploy the Application

## 1)  Local Deployment

#### 1) Clone the repostory 

```
https://github.com/Pratik-Prakash-Sannakki/WhatsApp-Insta_Resolution_Pumper.git

```
#### 2) Create a conda Environment 

```
conda create -n {Environment name} python==3.7 

```

#### 3) Start the Environment 

```
conda activate {Environment name} 

```

#### 4) Install requirements

```
pip install requirements.txt

```

#### 5) Run the Application 
```
python app.py

```

#### 6)  Access the local Web Application  

   copy the url eg. 127.0.0.1/5000 from the out put and paste the URL in your browser 

<br>


## 2)  Cloud Deployment


#### 1) Clone the repostory 

```
git clone https://github.com/Pratik-Prakash-Sannakki/MLOps_E2E_Workflow_LinearRegression.git

```
#### 2) Create a git repository with all the cloned file  
#### 3) Create a Render Account 
  link - https://render.com/
#### 4) link you github account and create a new Web service
#### 5) connect your git Repository 
#### 6) Change name of API to your desired name and change start command to
```
gunicorn --timeout 300 {flask file name}:app

```
#### 7) Deploy the application 

#### 8) Go to the URL provided on deployment 

#### 9) Now you can access the application remotely






