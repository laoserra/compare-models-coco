![Logo](9722_UBDC_logo.png)

# Evaluation of Object Detection Models

In this repo we present a tool to evaluate and compare object detection models using the coco metrics (https://cocodataset.org/#detection-eval) and coco libraries (https://github.com/cocodataset/cocoapi). This tool has been used by the Data Science Team to inform our decisions regarding the models to use for image detection projects within UBDC.

This script can be modified to analyse any number of models and any combination of object categories. The inputs to evaluate the models are a ground truth json file and a detection json file (one detection file per model in test) in the coco format (https://cocodataset.org/#format-data). The output is an Excel file with the results of the evaluation. 

To better understand the tool, we present here three json files we used as inputs. Important to note when producing detection files is the need to accommodate all detections, regardless their confidence level, since detections are ranked according to this value to produce the precision-recall curve. The average precision calculated corresponds to the area under the curve (AUC).
