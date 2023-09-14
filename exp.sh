num=(0 10000 20000 50000)
exp_object=("des_perf" "tv80" "wb_dma")
exp_timer=("ftask_only_partition_cost_not_reverse" "btask_only_partition_cost_not_reverse")

for obj in ${exp_object[@]}
do
  cd ./benchmark/$obj/
  rm *.txt
  cd ../../
done

for obj_timer in ${exp_timer[@]}
do
  cd ./ot/
  rm -rf timer/
  cp -r $obj_timer timer/
  cd ../
  for n in ${num[@]}
  do
    sed -i '' -e "s/size_t num_merge = 8888;/size_t num_merge = ${n};/g" "./ot/timer/timer.cpp"
    cd ./build
    make -j 16
    cd ../
    for obj in ${exp_object[@]}
    do
      cd ./benchmark/$obj/
      rm partition_${n}_$obj_timer.txt
      for i in 1 2 3
      do
        ../../bin/ot-shell < $obj.conf >> partition_${n}_$obj_timer.txt
      done
      cd ../../
    done
    sed -i '' -e "s/size_t num_merge = $n;/size_t num_merge = 8888;/g" "./ot/timer/timer.cpp"
  done
done

