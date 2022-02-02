#!/bin/sh

constant=20
for num_workers in 1 2 5 10
	do
		python distributed_sol.py --num_workers $num_workers --batch_size $((constant / num_workers))
	done
done
