expr='vectorized.mandel_set9(-2, 0.5, -1.25, 1.25, 1000, 1000, 80)'
for threads in 1 2 4 8 15
do
	printf "%3i threads \n" ${threads}
	NUMBA_NUM_THREADS=${threads} python3 -m timeit -s 'import vectorized' "$expr"
done
