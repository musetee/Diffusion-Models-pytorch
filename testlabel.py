import torch
#labels = torch.arange(5).long()
labels = torch.tensor([100,101,102,103,104]).long()
print(labels)

manual_crop=[91,100]
print(manual_crop[0])

train_cond=[]
cond_file = r'F:\yang_Projects\Datasets\Task1\pelvis\1PA047\ct_slice_cond.csv'
with open(cond_file, 'r', newline='') as csvfile:
        train_crop=[]
        import csv
        csv_reader = csv.DictReader(csvfile)
        for row in csv_reader:
            # Append the value to the list
            if int(row['slice'])>=manual_crop[0] and int(row['slice'])<=manual_crop[1]:
                print(int(row['slice']))
                train_cond.append(int(row['slice']))



'''
            train_crop.append(int(row['slice']))  # Assuming each row contains only one value
        print(train_crop[manual_crop[0]:manual_crop[1]])
        train_cond = train_cond + train_crop[manual_crop[0]:manual_crop[1]]
print(train_cond)

    train_cond = []
    val_cond = []
    for cond_file in train_cond_src:
        with open(cond_file, 'r', newline='') as csvfile:
            import csv
            csv_reader = csv.DictReader(csvfile)
            for row in csv_reader:
                # Append the value to the list
                if len(manual_crop)==2:
                   if int(row['slice'])>=manual_crop[0] and int(row['slice'])<=manual_crop[1]:
                    train_cond.append(int(row['slice']))  # Assuming each row contains only one value
                else:
                    train_cond.append(int(row['slice']))  # Assuming each row contains only one value
    
    for cond_file in val_cond_src:
        with open(cond_file, 'r', newline='') as csvfile:
            import csv
            csv_reader = csv.DictReader(csvfile)
            for row in csv_reader:
                # Append the value to the list
                val_cond.append(int(row['slice']))
                

    # Load 2D slices for training
    for sample in train_data:
        train_ds_2d_image = LoadImage(image_only=True, ensure_channel_first=False, simple_keys=True)(sample)
        #train_ds_2d_image=DivisiblePadd(["image", "label"], (-1,batch_size), mode="minimum")(train_ds_2d_image)
        name = os.path.basename(os.path.dirname(sample))
        num_slices = train_ds_2d_image.shape[-1]
        #(train_ds_2d_image.shape)
        #print(num_slices)
        shape_list_train.append({'patient': name, 'shape': train_ds_2d_image.shape})
        for i in range(num_slices):
            train_ds_2d.append(train_ds_2d_image[:,:,i])
        all_slices_train += num_slices
    print(len(train_ds_2d))

    # Load 2D slices for validation
    for sample in val_data:
        val_ds_2d_image = LoadImage(image_only=True, ensure_channel_first=False, simple_keys=True)(sample)
        #val_ds_2d_image=DivisiblePadd(["image", "label"], (-1, batch_size), mode="minimum")(val_ds_2d_image)
        name = os.path.basename(os.path.dirname(sample))
        shape_list_val.append({'patient': name, 'shape': val_ds_2d_image.shape})
        num_slices = val_ds_2d_image.shape[-1]
        for i in range(num_slices):
            val_ds_2d.append(val_ds_2d_image[:,:,i])
        all_slices_val += num_slices

        '''