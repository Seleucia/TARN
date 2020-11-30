##TARN
TARN: Temporal Attentive Relation Network for few-shot and zero-shot action recognition

Unofficial implement of "TARN: Temporal Attentive Relation Network for few-shot and zero-shot action recognition"  
Implemented by: Huseyin Coskun

##Dataset:
- EPIC: https://epic-kitchens.github.io/2020-100
- Extended GTEA Gaze+: http://cbs.ic.gatech.edu/fpv

We use https://github.com/waitrysb/C3D_Feature_Extractor repostory for feature extraction. 
We extracted features from 7th layer

##Usage:
Update dataset paths in helper/io_utils.py 
```
mkdir runs && mkdir runs/gaze && mkdir runs/epic
mkdir models && mkdir models/epic && mkdir models/gaze

pip install -r requirements.txt
python tarn_trainer.py
```
##Contact:
Huseyin Coskun  
email: name.lastname@tum.de

##Citation:
Paper: Bishay, Mina, Georgios Zoumpourlis, and Ioannis Patras. "TARN: Temporal Attentive Relation Network for Few-Shot and Zero-Shot Action Recognition." Accepted for BMVC, 2019.
