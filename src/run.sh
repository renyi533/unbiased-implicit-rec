rm -rf ../logs*

i="0"

wait_function () {
    sleep 10
    for job in `jobs -p`
    do
        echo $job
        wait $job || let "FAIL+=1"
    done
}

while [ $i -lt 10 ]
do

for model in ipwbpr ipwbpr_opt0 ipwbpr_opt1 ipwbpr_opt2 ipwbpr_opt3 ubpr_nclip oracle mf rmf bpr ubpr
do
python main.py \
  $model \
  --eps 5.0 \
  --pow_list 0.5 1.0 2.0 3.0 4.0 \
  --iters 5 
done > run_$i.log 2>&1 &

wait_function

python visualize.py \
  --eps 5.0 \
  --pow_list 0.5 1.0 2.0 3.0 4.0

rm -rf ../logs_$i
mkdir -p ../logs_$i
cp -rf ../logs/*  ../logs_$i/*

i=$[$i+1]
done