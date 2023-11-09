# generation-classification-cnn

This multi-headed convolution neural network uses feature fusion to determine the generation of people aged 1-116. 
</br>
</br>
Images are copied and synthetically altered. One CNN makes a prediction on brightened images and another on darkened. The final CNN extracts features from the original image. 
</br>
</br>
Their predictions are fused, producing a more general and accurate model than one trained solely on the unaltered images.

The classes are:

- Gen Alpha
- Gen Z
- Millenial
- Gen X
- Baby Boomer
- Post War
- WWII

>[!Note]
>This repo is still a work in progress

Dataset:
https://susanqq.github.io/UTKFace/

Definition of generational groups:
https://www.beresfordresearch.com/age-range-by-generation/
