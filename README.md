# Herbarium-Field Features in Few-Shot Plant Identification with Triplet Loss

This repository contains the implementation codes to our cross-domain few-shot plant identification experiments. This work investigates the robustness of cross-domain plant features learned using the triplet loss metric learning approach (HFTL Network) compared to the supervised classification approach (OSM Network) under general and few-shot experimental settings. 

Detailed experiments show that the HFTL model outperformed the OSM model in the few-shot setting and achieved comparable results in the general experimental setting. 

In addition, the feature dictionary generation schemes composed of various herbarium field feature combinations we proposed boost our models’ performance significantly compared to a single feature type dictionary strategy.

![Overview](https://github.com/NeuonAI/hftl_fewshot/blob/51bfc5afde979c6286e9a8af2ca0d714ea01f735/static/overview_revised.jpg)
![Dictionary schemes](https://github.com/NeuonAI/hftl_fewshot/blob/42474222f539ed9b10c602b36c99e8fee8da735d/static/dictionary_schemes.png)

## T-SNE Visualizations of the HFTL Model (a) and the OSM Model (b)
![T-SNE Visualization](https://github.com/NeuonAI/hftl_fewshot/blob/51bfc5afde979c6286e9a8af2ca0d714ea01f735/static/tsne_608-5.jpg)
<p align="center"> Figure 1</p>

![T-SNE Visualization](https://github.com/NeuonAI/hftl_fewshot/blob/51bfc5afde979c6286e9a8af2ca0d714ea01f735/static/tsne_608-5_tiles.jpg)
<p align="center"> Figure 2</p>

Figure 1 and Figure 2 show the T-SNE visualizations constructed with the herbarium-field embeddings generated by the HFTL model and the OSM model.
The images are grouped according to how similar their embeddings are to one another. The closer the images are stacked together, the more similar their embeddings are to one another.

Figure 1 is built from 4,402 herbarium and field images (embeddings) of 261 species from the training dataset. We randomly selected two species from all the available family taxon to balance out the samples.
The images are accompanied by their respective species id and each color represents a unique species id.

Meanwhile, Figure 2 is built from 392 images of 25 species. We randomly selected 25 species from 25 different family taxon to allow a clearer observation of the visualizations.
The close-up images’ different border colors represent the different species. Additionally, the white numbered text in the images represent the species id number.

In our experiments, we find that the triplet model is able to further harness the similarity between different domain embeddings compared to the conventional supervised model.




## Research article
How Transferable are Herbarium-Field Features in Few-Shot Plant Identification with Triplet Loss?
[https://doi.org/10.1109/APSIPAASC58517.2023.10317564](https://doi.org/10.1109/APSIPAASC58517.2023.10317564)


## Requirements
- TensorFlow 1.12
- [TensorFlow-Slim library](https://github.com/tensorflow/models/tree/r1.12.0/research/slim)
- [Pre-trained models (Inception-ResNet-v2)](https://github.com/tensorflow/models/tree/r1.12.0/research/slim#pre-trained-models)


## Dataset
- [PlantCLEF 2021](https://www.aicrowd.com/challenges/lifeclef-2021-plant)
- Dataset label map
[labels_classid_map.csv](https://github.com/NeuonAI/hftl_fewshot/blob/6a2331faa3ba0ba733e4cf8ccdd86d54929852a6/labels_classid_map.csv)

## Scripts
**Training scripts**
- HFTL Network
  - [train_hftl.py](https://github.com/NeuonAI/hftl_fewshot/blob/51bfc5afde979c6286e9a8af2ca0d714ea01f735/scripts/train/train_hftl.py)
  - [network_module_HFTL.py](https://github.com/NeuonAI/hftl_fewshot/blob/51bfc5afde979c6286e9a8af2ca0d714ea01f735/scripts/train/network_module_HFTL.py)
  - [database_module_HFTL.py](https://github.com/NeuonAI/hftl_fewshot/blob/51bfc5afde979c6286e9a8af2ca0d714ea01f735/scripts/train/database_module_HFTL.py)
- OSM Network
  - [train_osm.py](https://github.com/NeuonAI/hftl_fewshot/blob/51bfc5afde979c6286e9a8af2ca0d714ea01f735/scripts/train/train_osm.py)
  - [network_module_OSM.py](https://github.com/NeuonAI/hftl_fewshot/blob/51bfc5afde979c6286e9a8af2ca0d714ea01f735/scripts/train/network_module_OSM.py)

**Validation scripts**
- HFTL Network
  - [construct_dictionary_HFTL_method1.py](https://github.com/NeuonAI/hftl_fewshot/blob/51bfc5afde979c6286e9a8af2ca0d714ea01f735/scripts/test/construct_dictionary_HFTL_method1.py)
  - [construct_dictionary_HFTL_method2_3_4.py](https://github.com/NeuonAI/hftl_fewshot/blob/51bfc5afde979c6286e9a8af2ca0d714ea01f735/scripts/test/construct_dictionary_HFTL_method2_3_4.py)
  - [validate_HFTL.py](https://github.com/NeuonAI/hftl_fewshot/blob/51bfc5afde979c6286e9a8af2ca0d714ea01f735/scripts/test/validate_HFTL.py)
- OSM Network
  - [construct_dictionary_OSM_method1.py](https://github.com/NeuonAI/hftl_fewshot/blob/51bfc5afde979c6286e9a8af2ca0d714ea01f735/scripts/test/construct_dictionary_OSM_method1.py)
  - [construct_dictionary_OSM_method2_3_4.py](https://github.com/NeuonAI/hftl_fewshot/blob/51bfc5afde979c6286e9a8af2ca0d714ea01f735/scripts/test/construct_dictionary_OSM_method2_3_4.py)
  - [validate_OSM.py](https://github.com/NeuonAI/hftl_fewshot/blob/51bfc5afde979c6286e9a8af2ca0d714ea01f735/scripts/test/validate_OSM.py)
  
## Lists
**Training lists**
- HFTL Model
  - [hftl_herbarium_train.txt](https://github.com/NeuonAI/hftl_fewshot/blob/51bfc5afde979c6286e9a8af2ca0d714ea01f735/lists/hftl_herbarium_train.txt)
  - [hftl_herbarium_validation.txt](https://github.com/NeuonAI/hftl_fewshot/blob/51bfc5afde979c6286e9a8af2ca0d714ea01f735/lists/hftl_herbarium_validation.txt)
  - [hftl_field_train.txt](https://github.com/NeuonAI/hftl_fewshot/blob/51bfc5afde979c6286e9a8af2ca0d714ea01f735/lists/hftl_field_train.txt)
  - [hftl_field_validation.txt](https://github.com/NeuonAI/hftl_fewshot/blob/51bfc5afde979c6286e9a8af2ca0d714ea01f735/lists/hftl_field_validation.txt)

- HFTL-435 Model
  - [hftl-435_herbarium_train.txt](https://github.com/NeuonAI/hftl_fewshot/blob/51bfc5afde979c6286e9a8af2ca0d714ea01f735/lists/hftl-435_herbarium_train.txt)
  - [hftl-435_herbarium_validation.txt](https://github.com/NeuonAI/hftl_fewshot/blob/51bfc5afde979c6286e9a8af2ca0d714ea01f735/lists/hftl-435_herbarium_validation.txt)
  - [hftl-435_field_train.txt](https://github.com/NeuonAI/hftl_fewshot/blob/51bfc5afde979c6286e9a8af2ca0d714ea01f735/lists/hftl-435_field_train.txt)
  - [hftl-435_field_validation.txt](https://github.com/NeuonAI/hftl_fewshot/blob/51bfc5afde979c6286e9a8af2ca0d714ea01f735/lists/hftl-435_field_validation.txt)

 - HFTL-435-5 Model
   - [hftl-435-5_herbarium_train.txt](https://github.com/NeuonAI/hftl_fewshot/blob/51bfc5afde979c6286e9a8af2ca0d714ea01f735/lists/hftl-435-5_herbarium_train.txt)
   - [hftl-435-5_herbarium_validation.txt](https://github.com/NeuonAI/hftl_fewshot/blob/51bfc5afde979c6286e9a8af2ca0d714ea01f735/lists/hftl-435-5_herbarium_validation.txt)
   - [hftl-435-5_field_train.txt](https://github.com/NeuonAI/hftl_fewshot/blob/51bfc5afde979c6286e9a8af2ca0d714ea01f735/lists/hftl-435-5_field_train.txt)
   - [hftl-435-5_field_validation.txt](https://github.com/NeuonAI/hftl_fewshot/blob/51bfc5afde979c6286e9a8af2ca0d714ea01f735/lists/hftl-435-5_field_validation.txt)

- HFTL-608 Model
  - [hftl-608_herbarium_train.txt](https://github.com/NeuonAI/hftl_fewshot/blob/51bfc5afde979c6286e9a8af2ca0d714ea01f735/lists/hftl-608_herbarium_train.txt)
  - [hftl-608_herbarium_validation.txt](https://github.com/NeuonAI/hftl_fewshot/blob/51bfc5afde979c6286e9a8af2ca0d714ea01f735/lists/hftl-608_herbarium_validation.txt)
  - [hftl-608_field_train.txt](https://github.com/NeuonAI/hftl_fewshot/blob/51bfc5afde979c6286e9a8af2ca0d714ea01f735/lists/hftl-608_field_train.txt)
  - [hftl-608_field_validation.txt](https://github.com/NeuonAI/hftl_fewshot/blob/51bfc5afde979c6286e9a8af2ca0d714ea01f735/lists/hftl-608_field_validation.txt)

- HFTL-608-5 Model
  - [hftl-608-5_herbarium_train.txt](https://github.com/NeuonAI/hftl_fewshot/blob/51bfc5afde979c6286e9a8af2ca0d714ea01f735/lists/hftl-608-5_herbarium_train.txt)
  - [hftl-608-5_herbarium_validation.txt](https://github.com/NeuonAI/hftl_fewshot/blob/51bfc5afde979c6286e9a8af2ca0d714ea01f735/lists/hftl-608-5_herbarium_validation.txt)
  - [hftl-608-5_field_train.txt](https://github.com/NeuonAI/hftl_fewshot/blob/51bfc5afde979c6286e9a8af2ca0d714ea01f735/lists/hftl-608-5_field_train.txt)
  - [hftl-608-5_field_validation.txt](https://github.com/NeuonAI/hftl_fewshot/blob/51bfc5afde979c6286e9a8af2ca0d714ea01f735/lists/hftl-608-5_field_validation.txt)

- OSM Model
  - [osm_train.txt](https://github.com/NeuonAI/hftl_fewshot/blob/51bfc5afde979c6286e9a8af2ca0d714ea01f735/lists/osm_train.txt)
  - [osm_validation.txt](https://github.com/NeuonAI/hftl_fewshot/blob/51bfc5afde979c6286e9a8af2ca0d714ea01f735/lists/osm_validation.txt)

- OSM-435 Model
  - [osm-435_train.txt](https://github.com/NeuonAI/hftl_fewshot/blob/51bfc5afde979c6286e9a8af2ca0d714ea01f735/lists/osm-435_train.txt)
  - [osm-435_validation.txt](https://github.com/NeuonAI/hftl_fewshot/blob/51bfc5afde979c6286e9a8af2ca0d714ea01f735/lists/osm-435_validation.txt)

- OSM-435-5 Model
  - [osm-435-5_train.txt](https://github.com/NeuonAI/hftl_fewshot/blob/51bfc5afde979c6286e9a8af2ca0d714ea01f735/lists/osm-435-5_train.txt)
  - [osm-435-5_validation.txt](https://github.com/NeuonAI/hftl_fewshot/blob/51bfc5afde979c6286e9a8af2ca0d714ea01f735/lists/osm-435-5_validation.txt)

- OSM-608 Model
  - [osm-608_train.txt](https://github.com/NeuonAI/hftl_fewshot/blob/51bfc5afde979c6286e9a8af2ca0d714ea01f735/lists/osm-608_train.txt)
  - [osm-608_validation.txt](https://github.com/NeuonAI/hftl_fewshot/blob/51bfc5afde979c6286e9a8af2ca0d714ea01f735/lists/osm-608_validation.txt)

- OSM-608-5 Model
  - [osm-608-5_train.txt](https://github.com/NeuonAI/hftl_fewshot/blob/51bfc5afde979c6286e9a8af2ca0d714ea01f735/lists/osm-608-5_train.txt)
  - [osm-608-5_validation.txt](https://github.com/NeuonAI/hftl_fewshot/blob/51bfc5afde979c6286e9a8af2ca0d714ea01f735/lists/osm-608-5_validation.txt)

**Testing lists**
- [test_seen.txt](https://github.com/NeuonAI/hftl_fewshot/blob/51bfc5afde979c6286e9a8af2ca0d714ea01f735/lists/test_seen.txt)
- [test_unseen.txt](https://github.com/NeuonAI/hftl_fewshot/blob/51bfc5afde979c6286e9a8af2ca0d714ea01f735/lists/test_unseen.txt)

### Herbarium-Field Feature Dictionary Generation Scheme Lists
Model               |  Method 1 |  Method 2 | Method 3 | Method 4 
:-------------------------|:-------------------------|:-------------------------|:-------------------------|:-------------------------
HFTL | [dictionary_method1.txt](https://github.com/NeuonAI/hftl_fewshot/blob/51bfc5afde979c6286e9a8af2ca0d714ea01f735/dictionary/dictionary_method1.txt) | [dictionary_method2.txt](https://github.com/NeuonAI/hftl_fewshot/blob/51bfc5afde979c6286e9a8af2ca0d714ea01f735/dictionary/dictionary_method2.txt) | [dictionary_method3_herbarium.txt](https://github.com/NeuonAI/hftl_fewshot/blob/51bfc5afde979c6286e9a8af2ca0d714ea01f735/dictionary/dictionary_method3_herbarium.txt) [dictionary_method3_field.txt](https://github.com/NeuonAI/hftl_fewshot/blob/51bfc5afde979c6286e9a8af2ca0d714ea01f735/dictionary/dictionary_method3_field.txt) | [dictionary_method4_herbarium.txt](https://github.com/NeuonAI/hftl_fewshot/blob/51bfc5afde979c6286e9a8af2ca0d714ea01f735/dictionary/dictionary_method4_herbarium.txt) [dictionary_method4_field.txt](https://github.com/NeuonAI/hftl_fewshot/blob/51bfc5afde979c6286e9a8af2ca0d714ea01f735/dictionary/dictionary_method4_field.txt)
HFTL-435 | [dictionary_method1.txt](https://github.com/NeuonAI/hftl_fewshot/blob/51bfc5afde979c6286e9a8af2ca0d714ea01f735/dictionary/dictionary_method1.txt) | [dictionary_method2.txt](https://github.com/NeuonAI/hftl_fewshot/blob/51bfc5afde979c6286e9a8af2ca0d714ea01f735/dictionary/dictionary_method2.txt) | [dictionary_method3_herbarium_435.txt](https://github.com/NeuonAI/hftl_fewshot/blob/51bfc5afde979c6286e9a8af2ca0d714ea01f735/dictionary/dictionary_method3_herbarium_435.txt) [dictionary_method3_field_435.txt](https://github.com/NeuonAI/hftl_fewshot/blob/51bfc5afde979c6286e9a8af2ca0d714ea01f735/dictionary/dictionary_method3_field_435.txt) | [dictionary_method4_herbarium_435.txt](https://github.com/NeuonAI/hftl_fewshot/blob/51bfc5afde979c6286e9a8af2ca0d714ea01f735/dictionary/dictionary_method4_herbarium_435.txt) [dictionary_method4_field_435.txt](https://github.com/NeuonAI/hftl_fewshot/blob/51bfc5afde979c6286e9a8af2ca0d714ea01f735/dictionary/dictionary_method4_field_435.txt)
HFTL-435-5 | [dictionary_method1.txt](https://github.com/NeuonAI/hftl_fewshot/blob/51bfc5afde979c6286e9a8af2ca0d714ea01f735/dictionary/dictionary_method1.txt) | [dictionary_method2.txt](https://github.com/NeuonAI/hftl_fewshot/blob/51bfc5afde979c6286e9a8af2ca0d714ea01f735/dictionary/dictionary_method2.txt) | [dictionary_method3_herbarium.txt](https://github.com/NeuonAI/hftl_fewshot/blob/51bfc5afde979c6286e9a8af2ca0d714ea01f735/dictionary/dictionary_method3_herbarium.txt) [dictionary_method3_field.txt](https://github.com/NeuonAI/hftl_fewshot/blob/51bfc5afde979c6286e9a8af2ca0d714ea01f735/dictionary/dictionary_method3_field.txt) |  [dictionary_method4_herbarium.txt](https://github.com/NeuonAI/hftl_fewshot/blob/51bfc5afde979c6286e9a8af2ca0d714ea01f735/dictionary/dictionary_method4_herbarium.txt) [dictionary_method4_field.txt](https://github.com/NeuonAI/hftl_fewshot/blob/51bfc5afde979c6286e9a8af2ca0d714ea01f735/dictionary/dictionary_method4_field.txt)
HFTL-608 | [dictionary_method1.txt](https://github.com/NeuonAI/hftl_fewshot/blob/51bfc5afde979c6286e9a8af2ca0d714ea01f735/dictionary/dictionary_method1.txt) | [dictionary_method2.txt](https://github.com/NeuonAI/hftl_fewshot/blob/51bfc5afde979c6286e9a8af2ca0d714ea01f735/dictionary/dictionary_method2.txt) | [dictionary_method3_herbarium_608.txt](https://github.com/NeuonAI/hftl_fewshot/blob/51bfc5afde979c6286e9a8af2ca0d714ea01f735/dictionary/dictionary_method3_herbarium_608.txt) [dictionary_method3_field_608.txt](https://github.com/NeuonAI/hftl_fewshot/blob/51bfc5afde979c6286e9a8af2ca0d714ea01f735/dictionary/dictionary_method3_field_608.txt) | [dictionary_method4_herbarium_608.txt](https://github.com/NeuonAI/hftl_fewshot/blob/51bfc5afde979c6286e9a8af2ca0d714ea01f735/dictionary/dictionary_method4_herbarium_608.txt) [dictionary_method4_field_608.txt](https://github.com/NeuonAI/hftl_fewshot/blob/51bfc5afde979c6286e9a8af2ca0d714ea01f735/dictionary/dictionary_method4_field_608.txt)
HFTL-608-5 | [dictionary_method1.txt](https://github.com/NeuonAI/hftl_fewshot/blob/51bfc5afde979c6286e9a8af2ca0d714ea01f735/dictionary/dictionary_method1.txt) | [dictionary_method2.txt](https://github.com/NeuonAI/hftl_fewshot/blob/51bfc5afde979c6286e9a8af2ca0d714ea01f735/dictionary/dictionary_method2.txt) | [dictionary_method3_herbarium.txt](https://github.com/NeuonAI/hftl_fewshot/blob/51bfc5afde979c6286e9a8af2ca0d714ea01f735/dictionary/dictionary_method3_herbarium.txt) [dictionary_method3_field.txt](https://github.com/NeuonAI/hftl_fewshot/blob/51bfc5afde979c6286e9a8af2ca0d714ea01f735/dictionary/dictionary_method3_field.txt) | [dictionary_method4_herbarium.txt](https://github.com/NeuonAI/hftl_fewshot/blob/51bfc5afde979c6286e9a8af2ca0d714ea01f735/dictionary/dictionary_method4_herbarium.txt) [dictionary_method4_field.txt](https://github.com/NeuonAI/hftl_fewshot/blob/51bfc5afde979c6286e9a8af2ca0d714ea01f735/dictionary/dictionary_method4_field.txt)
OSM | [dictionary_method1.txt](https://github.com/NeuonAI/hftl_fewshot/blob/51bfc5afde979c6286e9a8af2ca0d714ea01f735/dictionary/dictionary_method1.txt) | [dictionary_method2.txt](https://github.com/NeuonAI/hftl_fewshot/blob/51bfc5afde979c6286e9a8af2ca0d714ea01f735/dictionary/dictionary_method2.txt) | [dictionary_method3_herbarium.txt](https://github.com/NeuonAI/hftl_fewshot/blob/51bfc5afde979c6286e9a8af2ca0d714ea01f735/dictionary/dictionary_method3_herbarium.txt) [dictionary_method3_field.txt](https://github.com/NeuonAI/hftl_fewshot/blob/51bfc5afde979c6286e9a8af2ca0d714ea01f735/dictionary/dictionary_method3_field.txt) | [dictionary_method4_herbarium.txt](https://github.com/NeuonAI/hftl_fewshot/blob/51bfc5afde979c6286e9a8af2ca0d714ea01f735/dictionary/dictionary_method4_herbarium.txt) [dictionary_method4_field.txt](https://github.com/NeuonAI/hftl_fewshot/blob/51bfc5afde979c6286e9a8af2ca0d714ea01f735/dictionary/dictionary_method4_field.txt) 
OSM-435 | [dictionary_method1.txt](https://github.com/NeuonAI/hftl_fewshot/blob/51bfc5afde979c6286e9a8af2ca0d714ea01f735/dictionary/dictionary_method1.txt) | [dictionary_method2.txt](https://github.com/NeuonAI/hftl_fewshot/blob/51bfc5afde979c6286e9a8af2ca0d714ea01f735/dictionary/dictionary_method2.txt) | [dictionary_method3_herbarium_435.txt](https://github.com/NeuonAI/hftl_fewshot/blob/51bfc5afde979c6286e9a8af2ca0d714ea01f735/dictionary/dictionary_method3_herbarium_435.txt) [dictionary_method3_field_435.txt](https://github.com/NeuonAI/hftl_fewshot/blob/51bfc5afde979c6286e9a8af2ca0d714ea01f735/dictionary/dictionary_method3_field_435.txt) | [dictionary_method4_herbarium_435.txt](https://github.com/NeuonAI/hftl_fewshot/blob/51bfc5afde979c6286e9a8af2ca0d714ea01f735/dictionary/dictionary_method4_herbarium_435.txt) [dictionary_method4_field_435.txt](https://github.com/NeuonAI/hftl_fewshot/blob/51bfc5afde979c6286e9a8af2ca0d714ea01f735/dictionary/dictionary_method4_field_435.txt)
OSM-435-5 | [dictionary_method1.txt](https://github.com/NeuonAI/hftl_fewshot/blob/51bfc5afde979c6286e9a8af2ca0d714ea01f735/dictionary/dictionary_method1.txt) | [dictionary_method2.txt](https://github.com/NeuonAI/hftl_fewshot/blob/51bfc5afde979c6286e9a8af2ca0d714ea01f735/dictionary/dictionary_method2.txt) | [dictionary_method3_herbarium.txt](https://github.com/NeuonAI/hftl_fewshot/blob/51bfc5afde979c6286e9a8af2ca0d714ea01f735/dictionary/dictionary_method3_herbarium.txt) [dictionary_method3_field.txt](https://github.com/NeuonAI/hftl_fewshot/blob/51bfc5afde979c6286e9a8af2ca0d714ea01f735/dictionary/dictionary_method3_field.txt) | [dictionary_method4_herbarium.txt](https://github.com/NeuonAI/hftl_fewshot/blob/51bfc5afde979c6286e9a8af2ca0d714ea01f735/dictionary/dictionary_method4_herbarium.txt) [dictionary_method4_field.txt](https://github.com/NeuonAI/hftl_fewshot/blob/51bfc5afde979c6286e9a8af2ca0d714ea01f735/dictionary/dictionary_method4_field.txt)
OSM-608 | [dictionary_method1.txt](https://github.com/NeuonAI/hftl_fewshot/blob/51bfc5afde979c6286e9a8af2ca0d714ea01f735/dictionary/dictionary_method1.txt) | [dictionary_method2.txt](https://github.com/NeuonAI/hftl_fewshot/blob/51bfc5afde979c6286e9a8af2ca0d714ea01f735/dictionary/dictionary_method2.txt) | [dictionary_method3_herbarium_608.txt](https://github.com/NeuonAI/hftl_fewshot/blob/51bfc5afde979c6286e9a8af2ca0d714ea01f735/dictionary/dictionary_method3_herbarium_608.txt) [dictionary_method3_field_608.txt](https://github.com/NeuonAI/hftl_fewshot/blob/51bfc5afde979c6286e9a8af2ca0d714ea01f735/dictionary/dictionary_method3_field_608.txt) | [dictionary_method4_herbarium_608.txt](https://github.com/NeuonAI/hftl_fewshot/blob/51bfc5afde979c6286e9a8af2ca0d714ea01f735/dictionary/dictionary_method4_herbarium_608.txt) [dictionary_method4_field_608.txt](https://github.com/NeuonAI/hftl_fewshot/blob/51bfc5afde979c6286e9a8af2ca0d714ea01f735/dictionary/dictionary_method4_field_608.txt)
OSM-608-5 | [dictionary_method1.txt](https://github.com/NeuonAI/hftl_fewshot/blob/51bfc5afde979c6286e9a8af2ca0d714ea01f735/dictionary/dictionary_method1.txt) | [dictionary_method2.txt](https://github.com/NeuonAI/hftl_fewshot/blob/51bfc5afde979c6286e9a8af2ca0d714ea01f735/dictionary/dictionary_method2.txt) | [dictionary_method3_herbarium.txt](https://github.com/NeuonAI/hftl_fewshot/blob/51bfc5afde979c6286e9a8af2ca0d714ea01f735/dictionary/dictionary_method3_herbarium.txt) [dictionary_method3_field.txt](https://github.com/NeuonAI/hftl_fewshot/blob/51bfc5afde979c6286e9a8af2ca0d714ea01f735/dictionary/dictionary_method3_field.txt) | [dictionary_method4_herbarium.txt](https://github.com/NeuonAI/hftl_fewshot/blob/51bfc5afde979c6286e9a8af2ca0d714ea01f735/dictionary/dictionary_method4_herbarium.txt) [dictionary_method4_field.txt](https://github.com/NeuonAI/hftl_fewshot/blob/51bfc5afde979c6286e9a8af2ca0d714ea01f735/dictionary/dictionary_method4_field.txt)

 

 
