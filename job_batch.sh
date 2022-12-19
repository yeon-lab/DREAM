
start=0
end=`cat config.json | jq '.data_loader.args.num_folds'`
end=$((end-1))

for i in $(eval echo {$start..$end})
do
   python train.py --fold_id=$i --np_data_dir "data_npz/edf_20_fpzcz" --config "config.json"

done  
