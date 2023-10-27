@echo off

:: Activate the virtual environment
E:\Projects\yang_proj\torch\venv\Scripts\activate.ps1

:: Run your Python scripts in a loop
python ddpm_mri2ct.py --run_name ddpm_monai_1027_1PC95 --mode testnifti --n_epochs 1 --normalize minmax --pad minimum --num_inference_steps 1000 --train_number 1 --val_number 1 --dataset_path E:\Projects\yang_proj\Task1\95 --GPU_ID 0 --center_crop 0 --pretrained_path E:\Projects\yang_proj\Diffusion-Models-pytorch\logs\ddpm_mri2ct_1016\models\model_259.pt

python ddpm_mri2ct.py --run_name ddpm_monai_1027_1PC93 --mode testnifti --n_epochs 1 --normalize minmax --pad minimum --num_inference_steps 1000 --train_number 1 --val_number 1 --dataset_path E:\Projects\yang_proj\Task1\93 --GPU_ID 0 --center_crop 0 --pretrained_path E:\Projects\yang_proj\Diffusion-Models-pytorch\logs\ddpm_mri2ct_1016\models\model_259.pt

python ddpm_mri2ct.py --run_name ddpm_monai_1027_1PC92 --mode testnifti --n_epochs 1 --normalize minmax --pad minimum --num_inference_steps 1000 --train_number 1 --val_number 1 --dataset_path E:\Projects\yang_proj\Task1\92 --GPU_ID 0 --center_crop 0 --pretrained_path E:\Projects\yang_proj\Diffusion-Models-pytorch\logs\ddpm_mri2ct_1016\models\model_259.pt

python ddpm_mri2ct.py --run_name ddpm_monai_1027_1PC88 --mode testnifti --n_epochs 1 --normalize minmax --pad minimum --num_inference_steps 1000 --train_number 1 --val_number 1 --dataset_path E:\Projects\yang_proj\Task1\88 --GPU_ID 0 --center_crop 0 --pretrained_path E:\Projects\yang_proj\Diffusion-Models-pytorch\logs\ddpm_mri2ct_1016\models\model_259.pt

python ddpm_mri2ct.py --run_name ddpm_monai_1027_1PC85 --mode testnifti --n_epochs 1 --normalize minmax --pad minimum --num_inference_steps 1000 --train_number 1 --val_number 1 --dataset_path E:\Projects\yang_proj\Task1\85 --GPU_ID 0 --center_crop 0 --pretrained_path E:\Projects\yang_proj\Diffusion-Models-pytorch\logs\ddpm_mri2ct_1016\models\model_259.pt

python ddpm_mri2ct.py --run_name ddpm_monai_1027_1PC84 --mode testnifti --n_epochs 1 --normalize minmax --pad minimum --num_inference_steps 1000 --train_number 1 --val_number 1 --dataset_path E:\Projects\yang_proj\Task1\84 --GPU_ID 0 --center_crop 0 --pretrained_path E:\Projects\yang_proj\Diffusion-Models-pytorch\logs\ddpm_mri2ct_1016\models\model_259.pt

python ddpm_mri2ct.py --run_name ddpm_monai_1027_1PC82 --mode testnifti --n_epochs 1 --normalize minmax --pad minimum --num_inference_steps 1000 --train_number 1 --val_number 1 --dataset_path E:\Projects\yang_proj\Task1\82 --GPU_ID 0 --center_crop 0 --pretrained_path E:\Projects\yang_proj\Diffusion-Models-pytorch\logs\ddpm_mri2ct_1016\models\model_259.pt

:: Deactivate the virtual environment
deactivate
