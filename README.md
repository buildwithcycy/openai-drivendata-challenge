# Aerial Imagery Classification
TL:DR Using convolutional neural networks to classify the building material of rooftops (aerial imagery from South America)

[**Dataset available on drivendata.org**](https://www.drivendata.org/competitions/58/disaster-response-roof-type/page/142/)

Our data consists of a set of overhead imagery taken using drones in seven locations across three countries (Colombia, Guatemala, and St Lucia) with labeled building footprints. Our goal is to classify each of the building footprints with the roof material type. 

**The Roof Material** type is associated with 5 classes:

*   Concrete Cement
*   Healthy Metal
*   Irregular Metal
*   Incomplete
*   Other

In this repository, we apply deep learning methods to classify the rooftop material of buildings located in **Borde Rural**, **Colombia**.

## Check out the [**Slides**](https://github.com/buildwithcycy/openai-drivendata-challenge/blob/master/aerial-imagery-classification-slides.pdf) to get an overview of the project.

## :information_desk_person: Project Motivation

In emergency situations, where time is of the essence, it can be critical to rapidly identify which houses have a higher likelihood of collapsing during a natural catastrophe such as a storm or an earthquake. By using deep learning, we can explore how to design a model which can be used to classify rooftop pictures taken by drones. This may help in identifying houses that are likely to collapse.


## :clipboard: Methods Used
1. Transfer Learning using fine tuning: VGG-16 <br>
 We use VGG-16 pre-trained on the ImageNet dataset as our base model and we add layers which constitute our head model.
 Using this method, we noticed that the training was slow as we were not using GPUs. Therefore, we adopted a different methodology.

2. Accelerated Transfer Learning with VGG-16 and a custom model <br>
  Just like Method 1, we use VGG-16 pre-trained on ImageNet. There are 18 pre-trained layers with the last one being layer 18 ,   a maxpooling2D layer. Since VGG-16 pre-trained layers are frozen, this means that the weights of those pre-trained layers are   not updated. This means that at every pass, these frozen layers would have guaranteed the same output for layer 18.<br>
  Therefore, instead of wasting our computing resources, we can pass the data once through the 18 pre-trained layers and save     the output of the layer 18 as our new dataset. <br>
  We feed our transformed data into on our custom model and make an inference using the custom model. This significantly cuts     down the training time. It also enables us to try more experiments by finetuning our hyperparameters. 


*Future Methods to Implement*:
  * Use area surrounding building by cropping larger images to make an inference
  * Use different neural networks (DenseNet, ResNet)

## :eight_spoked_asterisk: Running this program

1. Download the dataset for Borde Rural, Colombia which is available in this [Google Drive folder](https://drive.google.com/drive/folders/1aaBw9ImjmQ_WFaIIITBrgJtOSvgfh9tY?usp=sharing). Ensure the downloaded folder is named "borde_rural" and place it in the same folder as the rest of coding files that you got from this repository.
2. * Download all the files in this repository and run the file "main" using an IDE or terminal.<br>
  :name_badge: Ensure that you have correctly specified the location of your downloaded files in the program.


## :tea: Requirements
* Python 3
* Tensorflow 
* Keras
* Rasterio  For accessing geospatial raster data
* Geopandas For easily working with geospatial data 
* Pyproj    For cartographic projections and coordinate transformations
* No GPU required (For now)
* Your favorite playlist


## :smiley: Contribution
 * You can contribute by forking this repository, playing with the code, or adding your machine model. Any contributions you make are greatly appreciated.
 
## :pray: Acknowledgements
 * Special thanks to Hugo Larochelle for his amazing guidance on this project and providing useful advice on using transfer learning.
 * Certain figures in the slides are from the Deep Learning open source repository by Andrew Glassner.

