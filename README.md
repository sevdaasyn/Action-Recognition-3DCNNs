## Action Recognition using 3DCNNs

By taking out these approaches, This project aiming to perform a video analyzer for kitchen activities 
cutting, baking, serving etc. Primary goal is extracting noun and verb couples by using pre-trained models
published by Epic-Kitchens team. This dataset is the largest dataset so far. 
It contains 11.5 million frames that is about kitchen activities. 

There are three models based on ResNet50, Temporal Segment Network, Temporal Relational Network and Multiscale Temporal Relational Network. 
My first goal was determining which model has highest perpormance. 

After determine a model that has the most accurate results, 
extracted action features by using I3D network, the team released Inception-v1 I3D models trained on the Kinetics dataset training split. 
This model finetuned and retrained on our dataset with extracted action features by using rgb samples. 
Purpose of training a network from scratch was improving pretrained model’s performances. 
To do this, these two model evaluated and output weights are associated to obtain better results.
