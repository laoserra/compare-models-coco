from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
import pandas as pd

# Coco metrics (https://cocodataset.org/#detection-eval)
df = pd.DataFrame(
        {'Parameter': ['AP@[IoU=0.50:0.95|area=all|maxDets=100]', 
         'AP@[IoU=0.50|area=all|maxDets=100]', 
         'AP@[IoU=0.75|area=all|maxDets=100]',
         'AP@[IoU=0.50:0.95|area=small|maxDets=100]',
         'AP@[IoU=0.50:0.95|area=medium|maxDets=100]',
         'AP@[IoU=0.50:0.95|area=large|maxDets=100]',
         'AR@[IoU=0.50:0.95|area=all|maxDets=1]',
         'AR@[IoU=0.50:0.95|area=all|maxDets=10]',
         'AR@[IoU=0.50:0.95|area=all|maxDets=100]',
         'AR@[IoU=0.50:0.95|area=small|maxDets=100]',
         'AR@[IoU=0.50:0.95|area=medium|maxDets=100]',
         'AR@[IoU=0.50:0.95|area=large|maxDets=100]']}
) 

# my classes of objects
category_id = {"pedestrian": 1,
               "bus": 2,
               "van": 3,
               "lorry": 4,
               "car": 5,
               "taxi": 6,
               "cyclist": 7,
               "crowd": 8,
               "motorcycle": 9}

def get_key_from_value(val):
    return [key for key,value in category_id.items() 
            if value == val][0]

# my ground truth file
annFile = 'ground_truth_coco_format.json'
cocoGt = COCO(annFile)

# added zero to get results for all classes together
all_outputs = [0] + [val for val  in category_id.values()]


if __name__ == '__main__':

    for cat_id in all_outputs:

        # my detection files
        for model in (1, 3):
            resFile = f'detections_model_{model}_coco_format.json'

            cocoDt = cocoGt.loadRes(resFile)

            annType = 'bbox'

            # running evaluation
            cocoEval = COCOeval(cocoGt,cocoDt,annType)

            #to get overall results and per class of object
            if not cat_id:
                cocoEval.params.catIds = [val for val  in category_id.values()] 
            else:
                cocoEval.params.catIds = [cat_id] 

            cocoEval.evaluate()
            cocoEval.accumulate()
            cocoEval.summarize()

            data = np.round(cocoEval.stats, 3)

            if model == 1:
                df['model_frcnn'] = data
            else:
                df['model_yolov4'] = data


        # writing results to an Excel file
        if not cat_id:
            with pd.ExcelWriter('Validation_results.xlsx') as fp:
                df.to_excel(fp, sheet_name='overall', index=False)
        else:
            with pd.ExcelWriter('Validation_results.xlsx', mode='a') as fp:
                df.to_excel(fp, sheet_name=f'{get_key_from_value(cat_id)}', index=False)
