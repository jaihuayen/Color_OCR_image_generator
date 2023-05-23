python3 main.py --num_img 3 \
    --random_offset --blur --random_augmentation --fonts_path fonts/greek/ \
    --output_dir /mnt/mount_124/processed/paddleocrv3/training_set/folder1/ \
    --label_file /mnt/mount_124/processed/paddleocrv3/train.txt \
    --chars_file corpus_generator/greek_dict.txt \
    --corpus_path ./train_corpus/

python3 main.py --num_img 3 \
    --random_offset --blur --random_augmentation --fonts_path fonts/greek/ \
    --output_dir /mnt/mount_124/processed/paddleocrv3/validation_set/folder1/ \
    --label_file /mnt/mount_124/processed/paddleocrv3/validation.txt \
    --chars_file corpus_generator/greek_dict.txt \
    --corpus_path ./validation_corpus/