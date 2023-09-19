Replace Opentimer's timer for use

1. for vivek_int_reduced_update_cost_time and vivek_int_use_self_cost_only,the cost of ftask = 1, the cost of btask = 6. By doing this, partition actually choose more btasks to merge
2. for ftask_only_partition and btask_only_partition, the cost of all ftask(btask) are the same. 
3. currently all exp are based on btask_only_partition
4. all exps are run on my personal macbook 

9/14
1. created btask_only_partition_improved. Details in its folder readme.md

9/19
1. BUGs in btask_only_partition_GDCA

