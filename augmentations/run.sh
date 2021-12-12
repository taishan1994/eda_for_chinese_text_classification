nohup python -u aug_for_file.py \
--input "../data/THUCNews/cnews.train.txt" \
--num_aug 4 \
--replace_ratio 0.3 \
--insert_ratio 0.3 \
--swap_ratio 0.3 \
--delete_prob 0.3 > eda.log 2>&1 &
