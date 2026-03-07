---
tags:
  - TX
sources:
  - https://armbasedsolutions.com/blog-detail/computer-vision-software-difference-between-opencv-and-yolo
  - https://github.com/ultralytics/ultralytics
  - https://www.preprints.org/manuscript/202602.1844
  - https://github.com/lyuwenyu/RT-DETR?tab=readme-ov-file
  - https://link.springer.com/article/10.1007/s11119-025-10246-0
  - https://github.com/roboflow/rf-detr
  - https://odr.chalmers.se/bitstreams/eaabecde-c4e7-4d01-a606-0d54d5f24f59/download
---
OpenCV

RCNN ([Region Based Convolutional Neural Networks](https://en.wikipedia.org/wiki/Region_Based_Convolutional_Neural_Networks))
- a conduit à YOLO 11 d'après le papier cité par Claude Moulin
- R-CNN processes roughly 2,000 region proposals per image, feeding each proposal separately into the CNN for feature extraction and classification. Despite its high accuracy, running the CNN for each candidate region is quite **slow and computationally expensive**. **Fast R-CNN, in contrast to R-CNN**, produces a common feature map by feeding the image to the CNN once.
- **Faster R-CNN** enables the network itself to directly generate region proposals by using a Region Proposal Network (RPN) instead of Selective Search, which is the main bottleneck of **Fast R-CNN**. In this way, the method provides **significant improvements in both speed and accuracy**. However, due to the complexity of the architecture and the presence of the region proposal stage, the inference **speed of the model often remains insufficient** for real-time requirements.
- To overcome these limitations, one-stage object detection methods have been developed. Among single-stage approaches, the You Only Look Once (**YOLO**) family stands out as one of the most effective and widely used methods, balancing **speed and accuracy**.

Yolo
- Quelle version ? -> now yolo26 (yolo12 > yolo11)
	- Passés à 26 en 2026 pour correspondre aux années
- 


RT DETR (https://link.springer.com/article/10.1007/s11119-025-10246-0, 21 Mai 2025)
- RT-DETR-l is best suited for applications where false positives must be minimized
	- according to a paper benchmarking RT DETR-1 & YOLO9
	- contre des versions plus récentes de YOLO ?

[RF DETR](https://github.com/roboflow/rf-detr)
- RF-DETR consistently achieved higher accuracy, faster and smoother convergence, and greater robustness than YOLO11 variants. However, it requires more powerful hardware. (https://odr.chalmers.se/bitstreams/eaabecde-c4e7-4d01-a606-0d54d5f24f59/download)




diff entre openCV et Yolo
- **OpenCV** is a general-purpose computer vision library suitable for image processing and traditional vision tasks, with broad functionality but not focused on object detection.
- **YOLO** is an efficient object detection algorithm based on deep learning, designed for real-time object detection and suitable for complex scenarios.

Detectron : https://link.springer.com/article/10.1007/s11042-025-20647-y
- After thorough analysis, a thorough knowledge of the model’s potential has surfaced, and YOLOv8X, YOLOv8l, YOLOv8m, and YOLOv8s have proven their mettle with mAP50 scores ranging from 39.5% to an astounding 42.6%. EfficientDet demonstrated exceptional accuracy as well, exhibiting 89.5% precision and 63.6% recall. But the most impressive thing about this trip was how well the **Detectron 2** model performed. It was the top performer, attaining a significant **78%** mAP along with an exceptional 92.7% precision and a commendable 72.6% recall.



---

réu du 06/03
tenter à la fois yolo26, rt detr (dernière version ?) et rf detr ?

vidéo qu'on a déjà -> c'est Yolo11 qui avait été utilisé

certaines vidéos sont déjà annotées avec les déplacements et mouvements des intervenants (ex : se lever/s'asseoir, etc)

intérêt du temps réel dans la halle numérique 
- faire remonter en temps réel des informations au coach (ex : une personne monopolise la parole, est trop critique, ou au contraire n'intervient pas du tout, etc)

annoter manuellement frame par frame pour identifier les personnes 
- établir une stratégie pour l'identification
	1. chercher dans les papiers sur Yolo l'explication de l'association d'IDs aux personnes
	2. se baser uniquement sur les bounding box et faire l'interpolation nous-mêmes
		- tenter l'interpolation frame par frame ou bien sur des intervalles de temps + longs vu que les personnes bougent peu ?
		- 


Travailler sur la moyenne des embedings des images précédentes pour identifier une même personne sur une nouvelle image

tenter de faire tourner yolo 1 frame sur 2, 5, etc ?