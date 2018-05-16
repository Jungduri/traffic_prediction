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

### Data description <br />
Data contatning 22 columns. <br />
date code 17 + weather 4 + traffic 1 <br />
5 Years traffic data <br />
total 1515 records <br />
train/validation/test: 0.7/0.15/0.15 <br />

### algorithms decription <br />
__LSTM model:__ (hyperparamters optimized empiricalls <br />
number of cells: 4 <br />
number of hid. : 30 <br />
iteration      : 400 <br />
*Fully connected* layer in a last layer <br />
loss function: rmse + l2 norm <br />

__Optimization methods__ <br />
Dropout(prop=0.5) <br />
L2 norm <br />
Validation set <br />
xavier initialization <br />
adamoptimizer <br />

### [scr] file description.  <br />
clone the git and run lst_traffic_prediction.py <br />

