Traffic Flow Predection using LSTM Networks
===========================================


#### Hello!
#### This is Jungseok's personal project git.
#### I am feel free to comment


### Data Source
The full data is available in <http://data.ex.co.kr/portal/fdwn/view?type=TCS&num=35&requestfrom=dataset>. <br />
It is too big to upload in hear. 
<br /><br />
To run this code is only need total_seoul_to_gangwon_xxx.csv. <br />
It is uploaded in "/data"

### Data description
Data contatning 22 columns.
date code 17 + weather 4 + traffic 1
5 Years traffic data
total 1515 records
train/validation/test: 0.7/0.15/0.15

### algorithms decription
__LSTM model:__ (hyperparamters optimized empiricalls
number of cells: 4
number of hid. : 30
iteration      : 400
*Fully connected* layer in a last layer
loss function: rmse + l2 norm
__Optimization methods__
Dropout(prop=0.5)
L2 norm
Validation set
xavier initialization
adamoptimizer

### [scr] file description. 
clone the git and run lst_traffic_prediction.py

