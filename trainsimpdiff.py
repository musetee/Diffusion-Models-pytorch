from simple_diffusion import simpleDifftrainer
from my_dataset import myslicesloader
device='cuda' if torch.cuda.is_available() else 'cpu'
# path to dataset
dataset_path=''
if __name__ == '__main__':
    model=simpleDifftrainer()
    train_volume_ds,_,train_loader,_,_ = myslicesloader(dataset_path,
                    normalize='zscore',
                    train_number=1,
                    val_number=1,
                    train_batch_size=8,
                    val_batch_size=1,
                    saved_name_train='./train_ds_2d.csv',
                    saved_name_val='./val_ds_2d.csv',
                    resized_size=(512,512,None),
                    div_size=(16,16,None),
                    ifcheck_volume=False,
                    ifcheck_sclices=False,)
    model.train(train_loader, 
                learning_rate=1e-3, 
                epoch_num=4,
                val_interval=1,
                batch_interval=10)