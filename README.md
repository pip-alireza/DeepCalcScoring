## Intro
This is the offial page for the paper "Automated Measurement of Vascular Calcification in Femoral Endarterectomy Patients using Deep Learning". 

This project leverages deep learning to automatically extract the vascular system in CT images of the human body and measure the calcium content in lower extremities, achieving #SOTA performance. This work enables monitoring calcification in the vascular system and detecting any atherosclerosis related condition which could potentially lead to heart attack or even leg amputation.

# Citation 
If you use this code or models in your scientific work, please cite the following paper:

Bagheri Rajeoni, A., Pederson, B., Clair, D. G., Lessner, S. M., & Valafar, H. (2023). Automated Measurement of Vascular Calcification in Femoral Endarterectomy Patients Using Deep Learning. Diagnostics, 13(21), 3363. https://doi.org/10.3390/diagnostics13213363



# Environment setup and training
1. Create the appropriate conda environment by running:

```bash
conda env create -f DeepCalc.yml
```

2. Use "train.py" to train the model with your data.
3. After training, you can utilize "Vascular extraction & calcium scoring.py" to extract the vascular system using the trained model and subsequently measure calcification within the vascular system.

Link to the paper: https://www.mdpi.com/2075-4418/13/21/3363

<img width="1432" alt="1" src="https://github.com/pip-alireza/DeepCalcScoring/assets/130691419/028751e0-1bea-47d8-b501-6f1fc8b1c54b">






# Results
Below is an illustration on how the model extracts the vascular system and calcifrication. Proceeding from left to right: input data from human CT images is received, the model then extracts the vascular system (middle image), subsequently tracking calcification within the vascular system (right image). Finally, it calculates the calcium score by summing up all identified calcifications.

![Projection of 6572np-2(2)](https://github.com/pip-alireza/DeepCalcScoring/assets/130691419/03aa5da8-9564-4e93-b9de-ca55ef59e3ee)
![Picture4(1)](https://github.com/pip-alireza/DeepCalcScoring/assets/130691419/a5174ae6-c23a-4774-999f-b0b09952236c)
![Picture5(3)](https://github.com/pip-alireza/DeepCalcScoring/assets/130691419/08e2a60d-cb8f-4e86-bc1c-b0f47d3e1296)
