1. implement GPU search for topological order -> queue atomic version
2. optimize GPU search for topological order -> use a centric vector
3. fixed BUGs on number of threads not enough for one level of BFS -> BUT still, haven't actually fixed it! Need to call kernel for one level multiple times if threads are not enough
The reason I am not doing this now is that current ctest cases won't have this problems.
4. implementing partition parallel
5. implementing latest cycle elimination idea, 1st. adjacent level, 2nd. shortest path, 3rd. smallest partition id.
