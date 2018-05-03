filename=$1
fname_nosuffix=`echo $filename | awk -F '.' '{print | "cut -d '.' -f-"(NF-1)}'`
echo $fname_nosuffix
filedir=`echo $filename | awk -F '/' '{print | "cut -d '/' -f-"(NF-1)}'`
echo $filedir
/data/Experiments/caffe/tools/extra/parse_log.py $filename $filedir
for i in {0..7}; do 
    echo "start type "$i
    echo ${fname_nosuffix}_${i}.png
    python /data/Experiments/caffe/tools/extra/plot_training_log.py.example $i ${fname_nosuffix}_${i}.png $filename
    echo $i" done"
done
