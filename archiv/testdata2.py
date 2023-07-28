from mydataloader.slice_loader import myslicesloader,len_patchloader

if __name__ == '__main__':
    dataset_path=r"F:\yang_Projects\Datasets\Task1\pelvis"
    #import math print(math.ceil(20/4))

    train_batch_size=8
    train_volume_ds,_,train_loader,_,_ = myslicesloader(dataset_path,
                    normalize='none',
                    train_number=4,
                    val_number=1,
                    train_batch_size=train_batch_size,
                    val_batch_size=1,
                    saved_name_train='./train_ds_2d.csv',
                    saved_name_val='./val_ds_2d.csv',
                    resized_size=(512, 512, None),
                    div_size=(16,16,None),
                    ifcheck_volume=False,
                    ifcheck_sclices=False,)
    slice_number,batch_number =len_patchloader(train_volume_ds,train_batch_size)
    # test data
    #iter_train = iter(train_loader)
    #batch = next(iter_train)
    #print(batch['image'].shape)
    
    # slice data
    #slice_number=sum(train_volume_ds[i]['image'].shape[-1] for i in range(len(train_volume_ds)))
    #print(slice_number)

    # test data loop
    
    #for i, batch in enumerate(train_loader):
    #    print(i, batch['image'].shape)