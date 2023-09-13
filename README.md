```
## Datasets
To train on the MVtec Anomaly Detection dataset [download](https://www.mvtec.com/company/research/datasets/mvtec-ad)
the data and extract it. The [Describable Textures dataset](https://www.robots.ox.ac.uk/~vgg/data/dtd/) was used as the anomaly source 
image set in most of the experiments in the paper. You can run the **download_dataset.sh** script from the project directory
to download the MVTec and the DTD datasets to the **datasets** folder in the project directory:
```
./scripts/download_dataset.sh
```
## Training
Pass the folder containing the training dataset to the **train_DRAEM.py** script as the --data_path argument and the
folder locating the anomaly source images as the --anomaly_source_path argument. 
The training script also requires the batch size (--bs), learning rate (--lr), epochs (--epochs), path to store checkpoints
(--checkpoint_path) and path to store logs (--log_path).
Example:

```
python train_DRAEM.py --gpu_id 0 --obj_id -1 --lr 0.0001 --bs 8 --epochs 700 --data_path ./datasets/mvtec/ --anomaly_source_path ./datasets/dtd/images/ --checkpoint_path ./checkpoints/ --log_path ./logs/

python train_DRAEM.py --gpu_id 0 --obj_id -1 --lr 0.0001 --bs 8 --epochs 700 --data_path /root/autodl-tmp/wyw/data/ --anomaly_source_path ./datasets/dtd/images/ --checkpoint_path ./checkpoints/ --log_path ./logs/

python train_DRAEM-A.py --gpu_id 0 --obj_id -1 --lr 0.0001 --bs 8 --epochs 700 --data_path /root/autodl-tmp/wyw/data/ --anomaly_source_path ./datasets/dtd/images/ --checkpoint_path ./checkpoints/ --log_path ./logs/
```  

The conda environement used in the project is decsribed in **requirements.txt**.

## Evaluating
The test script requires the --gpu_id arguments, the name of the checkpoint files (--base_model_name) for trained models, the 
location of the MVTec anomaly detection dataset (--data_path) and the folder where the checkpoint files are located (--checkpoint_path)
with pretrained models can be run with:

```
python test_DRAEM.py --gpu_id 0 --base_model_name "DRAEM_seg_large_ae_large_0.0001_800_bs8" --data_path ./datasets/mvtec/ --checkpoint_path ./checkpoints/DRAEM_checkpoints/
python test_DRAEM.py --gpu_id 0 --base_model_name "DRAEM_seg_large_ae_large_0.0001_800_bs8" --data_path /root/autodl-tmp/wyw/data/ --checkpoint_path ./checkpoints
```


