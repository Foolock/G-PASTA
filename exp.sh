num=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20)
exp_object=("des_perf" "vga_lcd" "netcard_iccad")
exp_partition=("_partition_vivekDAG_GDCA_origin()" "_partition_vivekDAG_GDCA_cpu()" "_partition_vivekDAG_GDCA_gpu()" "_partition_vivekDAG_GDCA_gpu_deterministic()")


for obj in ${exp_object[@]}
do
  cd ./benchmark/$obj/
  rm *.txt
  rm -rf exp
  mkdir exp
  cd ../../
done

for par in ${exp_partition[@]}
do
  sed -i -e "s/_partition_vivekDAG_GDCA_origin();/${par};/g" "./ot/timer/timer.cpp"
  for n in ${num[@]}
  do
    sed -i -e "s/int partition_size = 1;/int partition_size = $n;/g" "./ot/timer/timer.hpp"
    cd ./build
    make -j 16 
    cd ../
    for obj in ${exp_object[@]}
    do
      cd ./benchmark/$obj/
      for i in 1 2 3 4 5 6 7 8 9 10
      do
        ../../bin/ot-shell < $obj.conf >> exp/${par}_$n.txt
      done
      cd ../../
    done
    sed -i -e "s/int partition_size = $n;/int partition_size = 1;/g" "./ot/timer/timer.hpp"
  done
  sed -i -e "s/${par};/_partition_vivekDAG_GDCA_origin();/g" "./ot/timer/timer.cpp"
done
